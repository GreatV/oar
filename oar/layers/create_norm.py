""" Norm Layer Factory

Create norm modules by string (to mirror create_act and creat_norm-act fns)

Copyright 2022 Ross Wightman
"""
import paddle
import functools
import types
from typing import Type
from .norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d, RmsNorm

_NORM_MAP = dict(
    batchnorm=paddle.nn.BatchNorm2D,
    batchnorm2d=paddle.nn.BatchNorm2D,
    batchnorm1d=paddle.nn.BatchNorm1D,
    groupnorm=GroupNorm,
    groupnorm1=GroupNorm1,
    layernorm=LayerNorm,
    layernorm2d=LayerNorm2d,
    rmsnorm=RmsNorm,
)
_NORM_TYPES = {m for n, m in _NORM_MAP.items()}


def create_norm_layer(layer_name, num_features, **kwargs):
    layer = get_norm_layer(layer_name)
    layer_instance = layer(num_features, **kwargs)
    return layer_instance


def get_norm_layer(norm_layer):
    if norm_layer is None:
        return None
    assert isinstance(norm_layer, (type, str, types.FunctionType, functools.partial))
    norm_kwargs = {}
    if isinstance(norm_layer, functools.partial):
        norm_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func
    if isinstance(norm_layer, str):
        if not norm_layer:
            return None
        layer_name = norm_layer.replace("_", "")
        norm_layer = _NORM_MAP[layer_name]
    else:
        norm_layer = norm_layer
    if norm_kwargs:
        norm_layer = functools.partial(norm_layer, **norm_kwargs)
    return norm_layer
