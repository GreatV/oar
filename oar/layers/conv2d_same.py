""" Conv2d w/ Same Padding

Hacked together by / Copyright 2020 Ross Wightman
"""
import paddle
from paddle import nn
from paddle.nn import functional as F
from typing import Tuple, Optional
from .config import is_exportable, is_scriptable
from .padding import pad_same, pad_same_arg, get_padding_value

_USE_EXPORT_CONV = False


def conv2d_same(
    x,
    weight: paddle.Tensor,
    bias: Optional[paddle.Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=(0, 0),
        dilation=dilation,
        groups=groups,
    )


class Conv2dSame(nn.Conv2D):
    """Tensorflow like 'SAME' convolution wrapper for 2D convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )

    def forward(self, x):
        return conv2d_same(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv2dSameExport(nn.Conv2D):
    """ONNX export friendly Tensorflow like 'SAME' convolution wrapper for 2D convolutions

    NOTE: This does not currently work with torch.jit.script
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dSameExport, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )
        self.pad = None
        self.pad_input_size = 0, 0

    def forward(self, x):
        input_size = x.shape[-2:]
        if self.pad is None:
            pad_arg = pad_same_arg(
                input_size, self.weight.shape[-2:], self.stride, self.dilation
            )
            self.pad = nn.ZeroPad2D(padding=pad_arg)
            self.pad_input_size = input_size
        x = self.pad(x)
        return F.conv2d(
            x=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop("padding", "")
    kwargs.setdefault("bias", False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        if _USE_EXPORT_CONV and is_exportable():
            assert not is_scriptable()
            return Conv2dSameExport(in_chs, out_chs, kernel_size, **kwargs)
        else:
            return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2D(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
