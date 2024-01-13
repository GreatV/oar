""" 'Fast' Normalization Functions

For GroupNorm and LayerNorm these functions bypass typical AMP upcast to float32.

Additionally, for LayerNorm, the APEX fused LN is used if available (which also does not upcast)

Hacked together by / Copyright 2022 Ross Wightman
"""
import paddle
from paddle.nn import functional as F
from typing import List, Optional


_USE_FAST_NORM = False


def is_fast_norm():
    return _USE_FAST_NORM


def set_fast_norm(enable=True):
    global _USE_FAST_NORM
    _USE_FAST_NORM = enable


def fast_group_norm(
    x: paddle.Tensor,
    num_groups: int,
    weight: Optional[paddle.Tensor] = None,
    bias: Optional[paddle.Tensor] = None,
    eps: float = 1e-05,
) -> paddle.Tensor:
    if not paddle.in_dynamic_mode():
        return F.group_norm(x, num_groups, weight, bias, eps)
    with paddle.amp.auto_cast(enable=False):
        return F.group_norm(x, num_groups, weight, bias, eps)


def fast_layer_norm(
    x: paddle.Tensor,
    normalized_shape: List[int],
    weight: Optional[paddle.Tensor] = None,
    bias: Optional[paddle.Tensor] = None,
    eps: float = 1e-05,
) -> paddle.Tensor:
    if not paddle.in_dynamic_mode():
        return F.layer_norm(
            x=x,
            normalized_shape=normalized_shape,
            weight=weight,
            bias=bias,
            epsilon=eps,
        )
    with paddle.amp.auto_cast(enable=False):
        return F.layer_norm(
            x=x,
            normalized_shape=normalized_shape,
            weight=weight,
            bias=bias,
            epsilon=eps,
        )


def rms_norm(
    x: paddle.Tensor,
    normalized_shape: List[int],
    weight: Optional[paddle.Tensor] = None,
    eps: float = 1e-05,
):
    norm_ndim = len(normalized_shape)
    if not paddle.in_dynamic_mode():
        assert norm_ndim == 1
        v = paddle.var(x=x, axis=-1).unsqueeze(axis=-1)
    else:
        dims = tuple(range(-1, -norm_ndim - 1, -1))
        v = paddle.var(x=x, axis=dims, keepdim=True)
    x = x * paddle.rsqrt(x=v + eps)
    if weight is not None:
        x = x * weight
    return x


def fast_rms_norm(
    x: paddle.Tensor,
    normalized_shape: List[int],
    weight: Optional[paddle.Tensor] = None,
    eps: float = 1e-05,
) -> paddle.Tensor:
    if not paddle.in_dynamic_mode():
        return rms_norm(x, normalized_shape, weight, eps)
    return rms_norm(x, normalized_shape, weight, eps)
