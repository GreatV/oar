""" Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2019, Ross Wightman
"""

import paddle
import math
from oar.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    DEFAULT_CROP_PCT,
)
from oar.data.auto_augment import (
    rand_augment_transform,
    augment_and_mix_transform,
    auto_augment_transform,
)
from oar.data.transforms import (
    str_to_interp_mode,
    str_to_pil_interp,
    RandomResizedCropAndInterpolation,
    ResizeKeepRatio,
    CenterCropOrPad,
    ToNumpy,
)
from oar.data.random_erasing import RandomErasing


def transforms_noaug_train(
    img_size=224,
    interpolation="bilinear",
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    if interpolation == "random":
        interpolation = "bilinear"
    tfl = [
        paddle.vision.transforms.Resize(
            img_size, interpolation=str_to_interp_mode(interpolation)
        ),
        paddle.vision.transforms.CenterCrop(img_size),
    ]
    if use_prefetcher:
        tfl += [ToNumpy()]
    else:
        tfl += [
            paddle.vision.transforms.ToTensor(),
            paddle.vision.transforms.Normalize(
                mean=paddle.to_tensor(data=mean), std=paddle.to_tensor(data=std)
            ),
        ]
    return paddle.vision.transforms.Compose(tfl)


def transforms_imagenet_train(
    img_size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="random",
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    separate=False,
    force_color_jitter=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))
    primary_tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, ratio=ratio, interpolation=interpolation
        )
    ]
    if hflip > 0.0:
        primary_tfl += [paddle.vision.transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [paddle.vision.transforms.RandomVerticalFlip(p=vflip)]
    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str)
        disable_color_jitter = not (force_color_jitter or "3a" in auto_augment)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = str_to_pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith("augmix"):
            aa_params["translate_pct"] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    if color_jitter is not None and not disable_color_jitter:
        if isinstance(color_jitter, (list, tuple)):
            assert len(color_jitter) in (3, 4)
        else:
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [paddle.vision.transforms.ColorJitter(*color_jitter)]
    final_tfl = []
    if use_prefetcher:
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            paddle.vision.transforms.ToTensor(),
            paddle.vision.transforms.Normalize(
                mean=paddle.to_tensor(data=mean), std=paddle.to_tensor(data=std)
            ),
        ]
        if re_prob > 0.0:
            final_tfl.append(
                RandomErasing(
                    re_prob,
                    mode=re_mode,
                    max_count=re_count,
                    num_splits=re_num_splits,
                    device="cpu",
                )
            )
    if separate:
        return (
            paddle.vision.transforms.Compose(primary_tfl),
            paddle.vision.transforms.Compose(secondary_tfl),
            paddle.vision.transforms.Compose(final_tfl),
        )
    else:
        return paddle.vision.transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_imagenet_eval(
    img_size=224,
    crop_pct=None,
    crop_mode=None,
    interpolation="bilinear",
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        scale_size = tuple([math.floor(x / crop_pct) for x in img_size])
    else:
        scale_size = math.floor(img_size / crop_pct)
        scale_size = scale_size, scale_size
    if crop_mode == "squash":
        tfl = [
            paddle.vision.transforms.Resize(
                scale_size, interpolation=str_to_interp_mode(interpolation)
            ),
            paddle.vision.transforms.CenterCrop(img_size),
        ]
    elif crop_mode == "border":
        fill = [round(255 * v) for v in mean]
        tfl = [
            ResizeKeepRatio(scale_size, interpolation=interpolation, longest=1.0),
            CenterCropOrPad(img_size, fill=fill),
        ]
    else:
        if scale_size[0] == scale_size[1]:
            tfl = [
                paddle.vision.transforms.Resize(
                    scale_size[0], interpolation=str_to_interp_mode(interpolation)
                )
            ]
        else:
            tfl = [ResizeKeepRatio(scale_size)]
        tfl += [paddle.vision.transforms.CenterCrop(img_size)]
    if use_prefetcher:
        tfl += [ToNumpy()]
    else:
        tfl += [
            paddle.vision.transforms.ToTensor(),
            paddle.vision.transforms.Normalize(
                mean=paddle.to_tensor(data=mean), std=paddle.to_tensor(data=std)
            ),
        ]
    return paddle.vision.transforms.Compose(tfl)


def create_transform(
    input_size,
    is_training=False,
    use_prefetcher=False,
    no_aug=False,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="bilinear",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    crop_pct=None,
    crop_mode=None,
    tf_preprocessing=False,
    separate=False,
):
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from oar.data.tf_preprocessing import TfPreprocessTransform

        transform = TfPreprocessTransform(
            is_training=is_training, size=img_size, interpolation=interpolation
        )
    elif is_training and no_aug:
        assert not separate, "Cannot perform split augmentation with no_aug"
        transform = transforms_noaug_train(
            img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
        )
    elif is_training:
        transform = transforms_imagenet_train(
            img_size,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            separate=separate,
        )
    else:
        assert (
            not separate
        ), "Separate transforms not supported for validation preprocessing"
        transform = transforms_imagenet_eval(
            img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            crop_pct=crop_pct,
            crop_mode=crop_mode,
        )
    return transform
