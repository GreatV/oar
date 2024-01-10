""" Padding Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import paddle
from paddle.nn import functional as F
import math
from typing import List, Tuple


def _FUNCTIONAL_PAD(x, pad, mode="constant", value=0.0, data_format="NCHW"):
    if len(x.shape) * 2 == len(pad) and mode == "constant":
        pad = (
            paddle.to_tensor(pad, dtype="int32")
            .reshape((-1, 2))
            .flip([0])
            .flatten()
            .tolist()
        )
    return F.pad(x, pad, mode, value, data_format)


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
    return padding


def get_same_padding(x: int, kernel_size: int, stride: int, dilation: int):
    if isinstance(x, paddle.Tensor):
        return paddle.clip(
            x=((x / stride).ceil() - 1) * stride + (kernel_size - 1) * dilation + 1 - x,
            min=0,
        )
    else:
        return max(
            (math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x,
            0,
        )


def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and dilation * (kernel_size - 1) % 2 == 0


def pad_same_arg(
    input_size: List[int],
    kernel_size: List[int],
    stride: List[int],
    dilation: List[int] = (1, 1),
) -> List[int]:
    ih, iw = input_size
    kh, kw = kernel_size
    pad_h = get_same_padding(ih, kh, stride[0], dilation[0])
    pad_w = get_same_padding(iw, kw, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]


def pad_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    dilation: List[int] = (1, 1),
    value: float = 0,
):
    ih, iw = x.shape[-2:]
    pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    x = _FUNCTIONAL_PAD(
        pad=(pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        value=value,
        x=x,
    )
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == "same":
            if is_static_pad(kernel_size, **kwargs):
                padding = get_padding(kernel_size, **kwargs)
            else:
                padding = 0
                dynamic = True
        elif padding == "valid":
            padding = 0
        else:
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic
