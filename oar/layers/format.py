import paddle
from enum import Enum
from typing import Union


class Format(str, Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"
    NCL = "NCL"
    NLC = "NLC"


FormatT = Union[str, Format]


def get_spatial_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NLC:
        dim = (1,)
    elif fmt is Format.NCL:
        dim = (2,)
    elif fmt is Format.NHWC:
        dim = 1, 2
    else:
        dim = 2, 3
    return dim


def get_channel_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NHWC:
        dim = 3
    elif fmt is Format.NLC:
        dim = 2
    else:
        dim = 1
    return dim


def nchw_to(x: paddle.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.transpose(perm=[0, 2, 3, 1])
    elif fmt == Format.NLC:
        x = x.flatten(start_axis=2)
        perm_32 = list(range(x.ndim))
        perm_32[1] = 2
        perm_32[2] = 1
        x = x.transpose(perm=perm_32)
    elif fmt == Format.NCL:
        x = x.flatten(start_axis=2)
    return x


def nhwc_to(x: paddle.Tensor, fmt: Format):
    if fmt == Format.NCHW:
        x = x.transpose(perm=[0, 3, 1, 2])
    elif fmt == Format.NLC:
        x = x.flatten(start_axis=1, stop_axis=2)
    elif fmt == Format.NCL:
        x = x.flatten(start_axis=1, stop_axis=2)
        perm_33 = list(range(x.ndim))
        perm_33[1] = 2
        perm_33[2] = 1
        x = x.transpose(perm=perm_33)
    return x
