""" Normalization layers and wrappers

Norm layer definitions that support fast norm and consistent channel arg order (always first arg).

Hacked together by / Copyright 2022 Ross Wightman
"""
import paddle
from paddle import nn
from paddle.nn import functional as F
import numbers
from typing import Tuple
from .fast_norm import is_fast_norm, fast_group_norm, fast_layer_norm, fast_rms_norm


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, eps=1e-05, affine=True):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)
        self.fast_norm = is_fast_norm()

    def forward(self, x):
        if self.fast_norm:
            return fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            return F.group_norm(
                x, self.num_groups, self.weight, self.bias, self.eps
            )


class GroupNorm1(nn.GroupNorm):
    """Group Normalization with 1 group.
    Input: tensor in shape [B, C, *]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
        self.fast_norm = is_fast_norm()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.fast_norm:
            return fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            return F.group_norm(
                x, self.num_groups, self.weight, self.bias, self.eps
            )


class LayerNorm(nn.LayerNorm):
    """LayerNorm w/ fast norm option"""

    def __init__(self, num_channels, eps=1e-06, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self._fast_norm:
            x = fast_layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        else:
            x = F.layer_norm(
                x=x,
                normalized_shape=self.normalized_shape,
                weight=self.weight,
                bias=self.bias,
                epsilon=self.eps,
            )
        return x


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial NCHW tensors"""

    def __init__(self, num_channels, eps=1e-06, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = x.transpose(perm=[0, 2, 3, 1])
        if self._fast_norm:
            x = fast_layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        else:
            x = F.layer_norm(
                x=x,
                normalized_shape=self.normalized_shape,
                weight=self.weight,
                bias=self.bias,
                epsilon=self.eps,
            )
        x = x.transpose(perm=[0, 3, 1, 2])
        return x


def _is_contiguous(tensor: paddle.Tensor) -> bool:
    return True


@paddle.jit.to_static
def _layer_norm_cf(
    x: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor, eps: float
):
    s, u = tuple(
        [
            paddle.var(x, axis=1, unbiased=False, keepdim=True),
            paddle.mean(x, axis=1, keepdim=True),
        ]
    )
    x = (x - u) * paddle.rsqrt(x=s + eps)
    x = x * weight[:, None, None] + bias[:, None, None]
    return x


def _layer_norm_cf_sqm(
    x: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor, eps: float
):
    u = x.mean(axis=1, keepdim=True)
    s = ((x * x).mean(axis=1, keepdim=True) - u * u).clip(min=0)
    x = (x - u) * paddle.rsqrt(x=s + eps)
    x = x * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
    return x


class LayerNormExp2d(nn.LayerNorm):
    """LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).

    Experimental implementation w/ manual norm for tensors non-contiguous tensors.

    This improves throughput in some scenarios (tested on Ampere GPU), esp w/ channels_last
    layout. However, benefits are not always clear and can perform worse on other GPUs.
    """

    def __init__(self, num_channels, eps=1e-06):
        super().__init__(num_channels, eps=eps)

    def forward(self, x) -> paddle.Tensor:
        if _is_contiguous(x):
            x = F.layer_norm(
                x=x.transpose(perm=[0, 2, 3, 1]),
                normalized_shape=self.normalized_shape,
                weight=self.weight,
                bias=self.bias,
                epsilon=self.eps,
            ).transpose(perm=[0, 3, 1, 2])
        else:
            x = _layer_norm_cf(x, self.weight, self.bias, self.eps)
        return x


class RmsNorm(nn.Layer):
    """RmsNorm w/ fast (apex) norm if available"""

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self, channels, eps=1e-06, affine=True, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = affine
        if self.elementwise_affine:
            out_35 = paddle.create_parameter(
                shape=paddle.empty(self.normalized_shape, **factory_kwargs).shape,
                dtype=paddle.empty(self.normalized_shape, **factory_kwargs)
                .numpy()
                .dtype,
                default_initializer=nn.initializer.Assign(
                    paddle.empty(self.normalized_shape, **factory_kwargs)
                ),
            )
            out_35.stop_gradient = not True
            self.weight = out_35
        else:
            self.add_parameter(name="weight", parameter=None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init_Constant = nn.initializer.Constant(value=1.0)
            init_Constant(self.weight)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = fast_rms_norm(x, self.normalized_shape, self.weight, self.eps)
        return x
