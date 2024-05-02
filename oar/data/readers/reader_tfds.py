""" Dataset reader that wraps TFDS datasets

Wraps many (most?) TFDS image-classification datasets
from https://github.com/tensorflow/datasets
https://www.tensorflow.org/datasets/catalog/overview#image_classification

Hacked together by / Copyright 2020 Ross Wightman
"""

import paddle
import math
import os
from typing import Optional
from PIL import Image

try:
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds

    try:
        tfds.even_splits("", 1, drop_remainder=False)
        has_buggy_even_splits = False
    except TypeError:
        print(
            "Warning: This version of tfds doesn't have the latest even_splits impl. Please update or use tfds-nightly for better fine-grained split behaviour."
        )
        has_buggy_even_splits = True
except ImportError as e:
    print(e)
    print(
        "Please install tensorflow_datasets package `pip install tensorflow-datasets`."
    )
    exit(1)
from .class_map import load_class_map
from .reader import Reader
from .shared_count import SharedCount

MAX_TP_SIZE = int(os.environ.get("TFDS_TP_SIZE", 8))
SHUFFLE_SIZE = int(os.environ.get("TFDS_SHUFFLE_SIZE", 8192))
PREFETCH_SIZE = int(os.environ.get("TFDS_PREFETCH_SIZE", 2048))


@tfds.decode.make_decoder()
def decode_example(serialized_image, feature, dct_method="INTEGER_ACCURATE"):
    return tf.image.decode_jpeg(serialized_image, channels=3, dct_method=dct_method)


def even_split_indices(split, n, num_samples):
    partitions = [round(i * num_samples / n) for i in range(n + 1)]
    return [f"{split}[{partitions[i]}:{partitions[i + 1]}]" for i in range(n)]


def get_class_labels(info):
    if "label" not in info.features:
        return {}
    class_label = info.features["label"]
    class_to_idx = {n: class_label.str2int(n) for n in class_label.names}
    return class_to_idx


class ReaderTfds(Reader):
    """Wrap Tensorflow Datasets for use in PyTorch

    There several things to be aware of:
      * To prevent excessive samples being dropped per epoch w/ distributed training or multiplicity of
         dataloader workers, the train iterator wraps to avoid returning partial batches that trigger drop_last
         https://github.com/pytorch/pytorch/issues/33413
      * With PyTorch IterableDatasets, each worker in each replica operates in isolation, the final batch
        from each worker could be a different size. For training this is worked around by option above, for
        validation extra samples are inserted iff distributed mode is enabled so that the batches being reduced
        across replicas are of same size. This will slightly alter the results, distributed validation will not be
        100% correct. This is similar to common handling in DistributedSampler for normal Datasets but a bit worse
        since there are up to N * J extra samples with IterableDatasets.
      * The sharding (splitting of dataset into TFRecord) files imposes limitations on the number of
        replicas and dataloader workers you can use. For really small datasets that only contain a few shards
        you may have to train non-distributed w/ 1-2 dataloader workers. This is likely not a huge concern as the
        benefit of distributed training or fast dataloading should be much less for small datasets.
      * This wrapper is currently configured to return individual, decompressed image samples from the TFDS
        dataset. The augmentation (transforms) and batching is still done in PyTorch. It would be possible
        to specify TF augmentation fn and return augmented batches w/ some modifications to other downstream
        components.

    """

    def __init__(
        self,
        root,
        name,
        split="train",
        class_map=None,
        is_training=False,
        batch_size=None,
        download=False,
        repeats=0,
        seed=42,
        input_name="image",
        input_img_mode="RGB",
        target_name="label",
        target_img_mode="",
        prefetch_size=None,
        shuffle_size=None,
        max_threadpool_size=None,
    ):
        """Tensorflow-datasets Wrapper

        Args:
            root: root data dir (ie your TFDS_DATA_DIR. not dataset specific sub-dir)
            name: tfds dataset name (eg `imagenet2012`)
            split: tfds dataset split (can use all TFDS split strings eg `train[:10%]`)
            is_training: training mode, shuffle enabled, dataset len rounded by batch_size
            batch_size: batch_size to use to unsure total samples % batch_size == 0 in training across all dis nodes
            download: download and build TFDS dataset if set, otherwise must use tfds CLI
            repeats: iterate through (repeat) the dataset this many times per iteration (once if 0 or 1)
            seed: common seed for shard shuffle across all distributed/worker instances
            input_name: name of Feature to return as data (input)
            input_img_mode: image mode if input is an image (currently PIL mode string)
            target_name: name of Feature to return as target (label)
            target_img_mode: image mode if target is an image (currently PIL mode string)
            prefetch_size: override default tf.data prefetch buffer size
            shuffle_size: override default tf.data shuffle buffer size
            max_threadpool_size: override default threadpool size for tf.data
        """
        super().__init__()
        self.root = root
        self.split = split
        self.is_training = is_training
        if self.is_training:
            assert (
                batch_size is not None
            ), "Must specify batch_size in training mode for reasonable behaviour w/ TFDS wrapper"
        self.batch_size = batch_size
        self.repeats = repeats
        self.common_seed = seed
        self.prefetch_size = prefetch_size or PREFETCH_SIZE
        self.shuffle_size = shuffle_size or SHUFFLE_SIZE
        self.max_threadpool_size = max_threadpool_size or MAX_TP_SIZE
        self.input_name = input_name
        self.input_img_mode = input_img_mode
        self.target_name = target_name
        self.target_img_mode = target_img_mode
        self.builder = tfds.builder(name, data_dir=root)
        if download:
            self.builder.download_and_prepare()
        self.remap_class = False
        if class_map:
            self.class_to_idx = load_class_map(class_map)
            self.remap_class = True
        else:
            self.class_to_idx = (
                get_class_labels(self.builder.info)
                if self.target_name == "label"
                else {}
            )
        self.split_info = self.builder.info.splits[split]
        self.num_samples = self.split_info.num_examples
        self.dist_rank = 0
        self.dist_num_replicas = 1
        if (
            paddle.distributed.is_available()
            and paddle.distributed.is_initialized()
            and paddle.distributed.get_world_size() > 1
        ):
            self.dist_rank = paddle.distributed.get_rank()
            self.dist_num_replicas = paddle.distributed.get_world_size()
        self.global_num_workers = 1
        self.num_workers = 1
        self.worker_info = None
        self.worker_seed = 0
        self.subsplit = None
        self.ds = None
        self.init_count = 0
        self.epoch_count = SharedCount()
        self.reinit_each_iter = self.is_training

    def set_epoch(self, count):
        self.epoch_count.value = count

    def set_loader_cfg(self, num_workers: Optional[int] = None):
        if self.ds is not None:
            return
        if num_workers is not None:
            self.num_workers = num_workers
            self.global_num_workers = self.dist_num_replicas * self.num_workers

    def _lazy_init(self):
        """Lazily initialize the dataset.

        This is necessary to init the Tensorflow dataset pipeline in the (dataloader) process that
        will be using the dataset instance. The __init__ method is called on the main process,
        this will be called in a dataloader worker process.

        NOTE: There will be problems if you try to re-use this dataset across different loader/worker
        instances once it has been initialized. Do not call any dataset methods that can call _lazy_init
        before it is passed to dataloader.
        """
        worker_info = paddle.io.get_worker_info()
        num_workers = 1
        global_worker_id = 0
        if worker_info is not None:
            self.worker_info = worker_info
            self.worker_seed = worker_info.seed
            self.num_workers = worker_info.num_workers
            self.global_num_workers = self.dist_num_replicas * self.num_workers
            global_worker_id = self.dist_rank * self.num_workers + worker_info.id
            """ Data sharding
            InputContext will assign subset of underlying TFRecord files to each 'pipeline' if used.
            My understanding is that using split, the underling TFRecord files will shuffle (shuffle_files=True)
            between the splits each iteration, but that understanding could be wrong.

            I am currently using a mix of InputContext shard assignment and fine-grained sub-splits for distributing
            the data across workers. For training InputContext is used to assign shards to nodes unless num_shards
            in dataset < total number of workers. Otherwise sub-split API is used for datasets without enough shards or
            for validation where we can't drop samples and need to avoid minimize uneven splits to avoid padding.
            """
            should_subsplit = self.global_num_workers > 1 and (
                self.split_info.num_shards < self.global_num_workers
                or not self.is_training
            )
            if should_subsplit:
                if has_buggy_even_splits:
                    if not isinstance(self.split_info, tfds.core.splits.SubSplitInfo):
                        subsplits = even_split_indices(
                            self.split, self.global_num_workers, self.num_samples
                        )
                        self.subsplit = subsplits[global_worker_id]
                else:
                    subsplits = tfds.even_splits(self.split, self.global_num_workers)
                    self.subsplit = subsplits[global_worker_id]
        input_context = None
        if self.global_num_workers > 1 and self.subsplit is None:
            input_context = tf.distribute.InputContext(
                num_input_pipelines=self.global_num_workers,
                input_pipeline_id=global_worker_id,
                num_replicas_in_sync=self.dist_num_replicas,
            )
        read_config = tfds.ReadConfig(
            shuffle_seed=self.common_seed + self.epoch_count.value,
            shuffle_reshuffle_each_iteration=True,
            input_context=input_context,
        )
        ds = self.builder.as_dataset(
            split=self.subsplit or self.split,
            shuffle_files=self.is_training,
            decoders=dict(image=decode_example()),
            read_config=read_config,
        )
        options = tf.data.Options()
        thread_member = (
            "threading" if hasattr(options, "threading") else "experimental_threading"
        )
        getattr(options, thread_member).private_threadpool_size = max(
            1, self.max_threadpool_size // self.num_workers
        )
        getattr(options, thread_member).max_intra_op_parallelism = 1
        ds = ds.with_options(options)
        if self.is_training or self.repeats > 1:
            ds = ds.repeat()
        if self.is_training:
            ds = ds.shuffle(
                min(self.num_samples, self.shuffle_size) // self.global_num_workers,
                seed=self.worker_seed,
            )
        ds = ds.prefetch(
            min(self.num_samples // self.global_num_workers, self.prefetch_size)
        )
        self.ds = tfds.as_numpy(ds)
        self.init_count += 1

    def _num_samples_per_worker(self):
        num_worker_samples = (
            max(1, self.repeats)
            * self.num_samples
            / max(self.global_num_workers, self.dist_num_replicas)
        )
        if self.is_training or self.dist_num_replicas > 1:
            num_worker_samples = math.ceil(num_worker_samples)
        if self.is_training and self.batch_size is not None:
            num_worker_samples = (
                math.ceil(num_worker_samples / self.batch_size) * self.batch_size
            )
        return int(num_worker_samples)

    def __iter__(self):
        if self.ds is None or self.reinit_each_iter:
            self._lazy_init()
        target_sample_count = self._num_samples_per_worker()
        sample_count = 0
        for sample in self.ds:
            input_data = sample[self.input_name]
            if self.input_img_mode:
                input_data = Image.fromarray(input_data, mode=self.input_img_mode)
            target_data = sample[self.target_name]
            if self.target_img_mode:
                target_data = Image.fromarray(target_data, mode=self.target_img_mode)
            elif self.remap_class:
                target_data = self.class_to_idx[target_data]
            yield input_data, target_data
            sample_count += 1
            if self.is_training and sample_count >= target_sample_count:
                break
        if (
            not self.is_training
            and self.dist_num_replicas > 1
            and self.subsplit is not None
            and 0 < sample_count < target_sample_count
        ):
            while sample_count < target_sample_count:
                yield input_data, target_data
                sample_count += 1

    def __len__(self):
        num_samples = self._num_samples_per_worker() * self.num_workers
        return num_samples

    def _filename(self, index, basename=False, absolute=False):
        assert False, "Not supported"

    def filenames(self, basename=False, absolute=False):
        """Return all filenames in dataset, overrides base"""
        if self.ds is None:
            self._lazy_init()
        names = []
        for sample in self.ds:
            if len(names) > self.num_samples:
                break
            if "file_name" in sample:
                name = sample["file_name"]
            elif "filename" in sample:
                name = sample["filename"]
            elif "id" in sample:
                name = sample["id"]
            else:
                assert False, "No supported name field present"
            names.append(name)
        return names
