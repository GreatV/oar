""" AvgPool2d w/ Same Padding

Hacked together by / Copyright 2020 Ross Wightman
"""

import paddle
from typing import List, Tuple, Optional
from .helpers import to_2tuple
from .padding import pad_same, get_padding_value


def avg_pool2d_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int] = (0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
):
    x = pad_same(x, kernel_size, stride)
    return paddle.nn.functional.avg_pool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=(0, 0),
        ceil_mode=ceil_mode,
        x=x,
        exclusive=not count_include_pad,
    )


class AvgPool2dSame(paddle.nn.AvgPool2D):
    """Tensorflow like 'SAME' wrapper for 2D average pooling"""

    def __init__(
        self,
        kernel_size: int,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(
            kernel_size, stride, (0, 0), ceil_mode, count_include_pad
        )

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return paddle.nn.functional.avg_pool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            x=x,
            exclusive=not self.count_include_pad,
        )


def max_pool2d_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int] = (0, 0),
    dilation: List[int] = (1, 1),
    ceil_mode: bool = False,
):
    x = pad_same(x, kernel_size, stride, value=-float("inf"))
    return paddle.nn.functional.max_pool2d(
        x, kernel_size, stride, (0, 0), dilation, ceil_mode
    )


class MaxPool2dSame(paddle.nn.MaxPool2D):
    """Tensorflow like 'SAME' wrapper for 2D max pooling"""

    def __init__(
        self, kernel_size: int, stride=None, padding=0, dilation=1, ceil_mode=False
    ):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        super(MaxPool2dSame, self).__init__(
            kernel_size, stride, (0, 0), dilation, ceil_mode
        )

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride, value=-float("inf"))
        return paddle.nn.functional.max_pool2d(
            x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode
        )


def create_pool2d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop("padding", "")
    padding, is_dynamic = get_padding_value(
        padding, kernel_size, stride=stride, **kwargs
    )
    if is_dynamic:
        if pool_type == "avg":
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == "max":
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
    elif pool_type == "avg":
        return paddle.nn.AvgPool2D(
            kernel_size, stride=stride, padding=padding, **kwargs
        )
    elif pool_type == "max":
        return paddle.nn.MaxPool2D(
            kernel_size, stride=stride, padding=padding, **kwargs
        )
    else:
        assert False, f"Unsupported pool type {pool_type}"
