""" Activations

A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

Hacked together by / Copyright 2020 Ross Wightman
"""

import paddle
from paddle import nn
from paddle.nn import functional as F


def swish(x: paddle.Tensor, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941"""
    return x.multiply_(x.sigmoid()) if inplace else x.multiply(x.sigmoid())


class Swish(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


def mish(x: paddle.Tensor, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x.multiply(F.softplus(x).tanh())


class Mish(nn.Layer):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681"""

    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()

    def forward(self, x: paddle.Tensor):
        return mish(x)


def sigmoid(x: paddle.Tensor, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()


# PyTorch has this, but not with a consistent inplace argument interface
class Sigmoid(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x: paddle.Tensor):
        return x.sigmoid_() if self.inplace else x.sigmoid()


def tanh(x: paddle.Tensor, inplace: bool = False):
    return x.tanh_() if inplace else x.tanh()


# PyTorch has this, but not with a consistent inplace argument interface
class Tanh(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x: paddle.Tensor):
        return x.tanh_() if self.inplace else x.tanh()


def hard_swish(x: paddle.Tensor, inplace: bool = False):
    inner = F.relu6(x + 3.0).div_(6.0)
    return x.multiply_(inner) if inplace else x.multiply(inner)


class HardSwish(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x: paddle.Tensor):
        return hard_swish(x, self.inplace)


def hard_sigmoid(x: paddle.Tensor, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clip_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class HardSigmoid(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x: paddle.Tensor):
        return hard_sigmoid(x, self.inplace)


def hard_mish(x: paddle.Tensor, inplace: bool = False):
    """Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.multiply_(0.5 * (x + 2).clip(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clip(min=0, max=2)


class HardMish(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    def forward(self, x: paddle.Tensor):
        return hard_mish(x, self.inplace)


class PReLU(nn.PReLU):
    """Applies PReLU (w/ dummy inplace arg)"""

    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, inplace: bool = False
    ) -> None:
        super(PReLU, self).__init__(num_parameters=num_parameters, init=init)

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return F.prelu(input, self.weight)


def gelu(x: paddle.Tensor, inplace: bool = False) -> paddle.Tensor:
    return F.gelu(x)


class GELU(nn.Layer):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)"""

    def __init__(self, inplace: bool = False):
        super(GELU, self).__init__()

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return F.gelu(input)


def gelu_tanh(x: paddle.Tensor, inplace: bool = False) -> paddle.Tensor:
    return F.gelu(x, approximate="tanh")


class GELUTanh(nn.Layer):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)"""

    def __init__(self, inplace: bool = False):
        super(GELUTanh, self).__init__()

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return F.gelu(input, approximate="tanh")


def quick_gelu(x: paddle.Tensor, inplace: bool = False) -> paddle.Tensor:
    return x * F.sigmoid(1.702 * x)


class QuickGELU(nn.Layer):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)"""

    def __init__(self, inplace: bool = False):
        super(QuickGELU, self).__init__()

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return quick_gelu(input)
