""" Dataset Factory

Hacked together by / Copyright 2021, Ross Wightman
"""

import paddle
import os

try:
    has_places365 = True
except ImportError:
    has_places365 = False
try:
    has_inaturalist = True
except ImportError:
    has_inaturalist = False
try:
    has_qmnist = True
except ImportError:
    has_qmnist = False
try:
    has_imagenet = True
except ImportError:
    has_imagenet = False
from .dataset import IterableImageDataset, ImageDataset

_TORCH_BASIC_DS = dict(
    cifar10=CIFAR10,
    cifar100=CIFAR100,
    mnist=MNIST,
    kmnist=KMNIST,
    fashion_mnist=FashionMNIST,
)
_TRAIN_SYNONYM = dict(train=None, training=None)
_EVAL_SYNONYM = dict(val=None, valid=None, validation=None, eval=None, evaluation=None)


def _search_split(root, split):
    split_name = split.split("[")[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root

    def _try(syn):
        for s in syn:
            try_root = os.path.join(root, s)
            if os.path.exists(try_root):
                return try_root
        return root

    if split_name in _TRAIN_SYNONYM:
        root = _try(_TRAIN_SYNONYM)
    elif split_name in _EVAL_SYNONYM:
        root = _try(_EVAL_SYNONYM)
    return root


def create_dataset(
    name,
    root,
    split="validation",
    search_split=True,
    class_map=None,
    load_bytes=False,
    is_training=False,
    download=False,
    batch_size=None,
    seed=42,
    repeats=0,
    **kwargs,
):
    """Dataset factory method

    In parenthesis after each arg are the type of dataset supported for each arg, one of:
      * folder - default, timm folder (or tar) based ImageDataset
      * torch - paddle.vision based datasets
      * HFDS - Hugging Face Datasets
      * TFDS - Tensorflow-datasets wrapper in IterabeDataset interface via IterableImageDataset
      * WDS - Webdataset
      * all - any of the above

    Args:
        name: dataset name, empty is okay for folder based datasets
        root: root folder of dataset (all)
        split: dataset split (all)
        search_split: search for split specific child fold from root so one can specify
            `imagenet/` instead of `/imagenet/val`, etc on cmd line / config. (folder, torch/folder)
        class_map: specify class -> index mapping via text file or dict (folder)
        load_bytes: load data, return images as undecoded bytes (folder)
        download: download dataset if not present and supported (HFDS, TFDS, torch)
        is_training: create dataset in train mode, this is different from the split.
            For Iterable / TDFS it enables shuffle, ignored for other datasets. (TFDS, WDS)
        batch_size: batch size hint for (TFDS, WDS)
        seed: seed for iterable datasets (TFDS, WDS)
        repeats: dataset repeats per iteration i.e. epoch (TFDS, WDS)
        **kwargs: other args to pass to dataset

    Returns:
        Dataset object
    """
    name = name.lower()
    if name.startswith("torch/"):
        name = name.split("/", 2)[-1]
        torch_kwargs = dict(root=root, download=download, **kwargs)
        if name in _TORCH_BASIC_DS:
            ds_class = _TORCH_BASIC_DS[name]
            use_train = split in _TRAIN_SYNONYM
            ds = ds_class(train=use_train, **torch_kwargs)
        elif name == "inaturalist" or name == "inat":
            assert (
                has_inaturalist
            ), "Please update to PyTorch 1.10, paddle.vision 0.11+ for Inaturalist"
            target_type = "full"
            split_split = split.split("/")
            if len(split_split) > 1:
                target_type = split_split[0].split("_")
                if len(target_type) == 1:
                    target_type = target_type[0]
                split = split_split[-1]
            if split in _TRAIN_SYNONYM:
                split = "2021_train"
            elif split in _EVAL_SYNONYM:
                split = "2021_valid"
            ds = paddle.vision.datasets.INaturalist(
                version=split, target_type=target_type, **torch_kwargs
            )
        elif name == "places365":
            assert (
                has_places365
            ), "Please update to a newer PyTorch and paddle.vision for Places365 dataset."
            if split in _TRAIN_SYNONYM:
                split = "train-standard"
            elif split in _EVAL_SYNONYM:
                split = "val"
            ds = paddle.vision.datasets.Places365(split=split, **torch_kwargs)
        elif name == "qmnist":
            assert (
                has_qmnist
            ), "Please update to a newer PyTorch and paddle.vision for QMNIST dataset."
            use_train = split in _TRAIN_SYNONYM
            ds = paddle.vision.datasets.QMNIST(train=use_train, **torch_kwargs)
        elif name == "imagenet":
            assert (
                has_imagenet
            ), "Please update to a newer PyTorch and paddle.vision for ImageNet dataset."
            if split in _EVAL_SYNONYM:
                split = "val"
            ds = paddle.vision.datasets.ImageNet(split=split, **torch_kwargs)
        elif name == "image_folder" or name == "folder":
            if search_split and os.path.isdir(root):
                root = _search_split(root, split)
            ds = paddle.vision.datasets.ImageFolder(root, **kwargs)
        else:
            assert False, f"Unknown paddle.vision dataset {name}"
    elif name.startswith("hfds/"):
        ds = ImageDataset(root, reader=name, split=split, class_map=class_map, **kwargs)
    elif name.startswith("tfds/"):
        ds = IterableImageDataset(
            root,
            reader=name,
            split=split,
            class_map=class_map,
            is_training=is_training,
            download=download,
            batch_size=batch_size,
            repeats=repeats,
            seed=seed,
            **kwargs,
        )
    elif name.startswith("wds/"):
        ds = IterableImageDataset(
            root,
            reader=name,
            split=split,
            class_map=class_map,
            is_training=is_training,
            batch_size=batch_size,
            repeats=repeats,
            seed=seed,
            **kwargs,
        )
    else:
        if search_split and os.path.isdir(root):
            root = _search_split(root, split)
        ds = ImageDataset(
            root, reader=name, class_map=class_map, load_bytes=load_bytes, **kwargs
        )
    return ds
