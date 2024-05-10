""" Loader Factory, Fast Collate, CUDA Prefetcher

Prefetcher and Fast Collate inspired by NVIDIA APEX example at
https://github.com/NVIDIA/apex/commit/d5e2bb4bdeedd27b1dfaf5bb2b24d6c000dee9be#diff-cf86c282ff7fba81fad27a559379d5bf

Hacked together by / Copyright 2019, Ross Wightman
"""

import logging
import random
from contextlib import suppress
from functools import partial
from itertools import repeat
from typing import Callable
import numpy as np
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import paddle
from .dataset import IterableImageDataset
from .distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
from .random_erasing import RandomErasing
from .mixup import FastCollateMixup
from .transforms_factory import create_transform

_logger = logging.getLogger(__name__)


def fast_collate(batch):
    """A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = paddle.zeros(shape=flattened_batch_size, dtype="int64")
        tensor = paddle.zeros(
            shape=(flattened_batch_size, *batch[0][0][0].shape), dtype="uint8"
        )
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += paddle.to_tensor(data=batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = paddle.to_tensor(data=[b[1] for b in batch], dtype="int64")
        assert len(targets) == batch_size
        tensor = paddle.zeros(shape=(batch_size, *batch[0][0].shape), dtype="uint8")
        for i in range(batch_size):
            tensor[i] += paddle.to_tensor(data=batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], paddle.Tensor):
        targets = paddle.to_tensor(data=[b[1] for b in batch], dtype="int64")
        assert len(targets) == batch_size
        tensor = paddle.zeros(shape=(batch_size, *batch[0][0].shape), dtype="uint8")
        for i in range(batch_size):
            paddle.assign(batch[i][0], output=tensor[i])
        return tensor, targets
    else:
        assert False


def adapt_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) != n:
        x_mean = np.mean(x).item()
        x = (x_mean,) * n
        _logger.warning(
            f"Pretrained mean/std different shape than model, using avg value {x}."
        )
    else:
        assert len(x) == n, "normalization stats must match image channels"
    return x


class PrefetchLoader:

    def __init__(
        self,
        loader,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        channels=3,
        device=str("cuda").replace("cuda", "gpu"),
        img_dtype="float32",
        fp16=False,
        re_prob=0.0,
        re_mode="const",
        re_count=1,
        re_num_splits=0,
    ):
        mean = adapt_to_chs(mean, channels)
        std = adapt_to_chs(std, channels)
        normalization_shape = 1, channels, 1, 1
        self.loader = loader
        self.device = device
        if fp16:
            img_dtype = "float16"
        self.img_dtype = img_dtype
        self.mean = paddle.to_tensor(
            data=[(x * 255) for x in mean], dtype=img_dtype, place=device
        ).view(normalization_shape)
        self.std = paddle.to_tensor(
            data=[(x * 255) for x in std], dtype=img_dtype, place=device
        ).view(normalization_shape)
        if re_prob > 0.0:
            self.random_erasing = RandomErasing(
                probability=re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device=device,
            )
        else:
            self.random_erasing = None
        self.is_cuda = paddle.device.cuda.device_count() >= 1 and device.type == "cuda"

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = paddle.device.cuda.Stream()
            stream_context = partial(paddle.device.cuda.stream_guard, stream=stream)
        else:
            stream = None
            stream_context = suppress
        for next_input, next_target in self.loader:
            with stream_context():
                next_input = next_input.to(device=self.device, blocking=not True)
                next_target = next_target.to(device=self.device, blocking=not True)
                next_input = (
                    next_input.to(self.img_dtype)
                    .subtract_(y=paddle.to_tensor(self.mean))
                    .divide_(y=paddle.to_tensor(self.std))
                )
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input)
            if not first:
                yield input, target
            else:
                first = False
            if stream is not None:
                paddle.device.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target
        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def mixup_enabled(self):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            return self.loader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            self.loader.collate_fn.mixup_enabled = x


def _worker_init(worker_id, worker_seeding="all"):
    worker_info = paddle.io.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        paddle.seed(seed=seed)
        np.random.seed(seed % (2**32 - 1))
    else:
        assert worker_seeding in ("all", "part")
        if worker_seeding == "all":
            np.random.seed(worker_info.seed % (2**32 - 1))


def create_loader(
    dataset,
    input_size,
    batch_size,
    is_training=False,
    use_prefetcher=True,
    no_aug=False,
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_split=False,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    num_aug_repeats=0,
    num_aug_splits=0,
    interpolation="bilinear",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    num_workers=1,
    distributed=False,
    crop_pct=None,
    crop_mode=None,
    collate_fn=None,
    pin_memory=False,
    fp16=False,
    img_dtype="float32",
    device=str("cuda").replace("cuda", "gpu"),
    tf_preprocessing=False,
    use_multi_epochs_loader=False,
    persistent_workers=True,
    worker_seeding="all",
):
    re_num_splits = 0
    if re_split:
        re_num_splits = num_aug_splits or 2
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        crop_mode=crop_mode,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )
    if isinstance(dataset, IterableImageDataset):
        dataset.set_loader_cfg(num_workers=num_workers)
    sampler = None
    if distributed and not isinstance(dataset, paddle.io.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = paddle.io.DistributedBatchSampler(
                    dataset=dataset, shuffle=True, batch_size=1
                )
        else:
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert (
            num_aug_repeats == 0
        ), "RepeatAugment not currently supported in non-distributed or IterableDataset use"
    if collate_fn is None:
        collate_fn = (
            fast_collate
            if use_prefetcher
            else paddle.io.dataloader.collate.default_collate_fn
        )
    loader_class = paddle.io.DataLoader
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader
    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, paddle.io.IterableDataset)
        and sampler is None
        and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers,
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop("persistent_workers")
        loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.0
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_size[0],
            device=device,
            fp16=fp16,
            img_dtype=img_dtype,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
        )
    return loader


class MultiEpochsDataLoader(paddle.io.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return (
            len(self.sampler)
            if self.batch_sampler is None
            else len(self.batch_sampler.sampler)
        )

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
