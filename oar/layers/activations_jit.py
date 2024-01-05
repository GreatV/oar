""" Activations

A collection of jit-scripted activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

All jit scripted activations are lacking in-place variations on purpose, scripted kernel fusion does not
currently work across in-place op boundaries, thus performance is equal to or less than the non-scripted
versions if they contain in-place ops.

Hacked together by / Copyright 2020 Ross Wightman
"""

import paddle
from paddle import nn
from paddle.nn import functional as F


@paddle.jit.to_static
def swish_jit(x: paddle.Tensor, inplace: bool = False) -> paddle.Tensor:
    """Swish - Described in: https://arxiv.org/abs/1710.05941"""
    return x.multiply(x.sigmoid())


@paddle.jit.to_static
def mish_jit(x: paddle.Tensor, _inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681"""
    return x.multiply(F.softplus(x).tanh())


class SwishJit(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(SwishJit, self).__init__()

    def forward(self, x: paddle.Tensor):
        return swish_jit(x)


class MishJit(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(MishJit, self).__init__()

    def forward(self, x: paddle.Tensor):
        return mish_jit(x)


@paddle.jit.to_static
def hard_sigmoid_jit(x: paddle.Tensor, inplace: bool = False):
    # return F.relu6(x + 3.) / 6.
    return (x + 3).clip(min=0, max=6).div(6.0)  # clamp seems ever so slightly faster?


class HardSigmoidJit(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(HardSigmoidJit, self).__init__()

    def forward(self, x: paddle.Tensor):
        return hard_sigmoid_jit(x)


@paddle.jit.to_static
def hard_swish_jit(x: paddle.Tensor, inplace: bool = False):
    # return x * (F.relu6(x + 3.) / 6)
    return x * (x + 3).clip(min=0, max=6).div(
        6.0
    )  # clamp seems ever so slightly faster?


class HardSwishJit(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(HardSwishJit, self).__init__()

    def forward(self, x: paddle.Tensor):
        return hard_swish_jit(x)


@paddle.jit.to_static
def hard_mish_jit(x: paddle.Tensor, inplace: bool = False):
    """Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    return 0.5 * x * (x + 2).clip(min=0, max=2)


class HardMishJit(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(HardMishJit, self).__init__()

    def forward(self, x: paddle.Tensor):
        return hard_mish_jit(x)
