""" Filter Response Norm in PyTorch

Based on `Filter Response Normalization Layer` - https://arxiv.org/abs/1911.09737

Hacked together by / Copyright 2021 Ross Wightman
"""
import paddle
from paddle import nn
from .create_act import create_act_layer
from .trace_utils import _assert


def inv_instance_rms(x, eps: float = 1e-05):
    rms = (
        x.square()
        .astype(dtype="float32")
        .mean(axis=(2, 3), keepdim=True)
        .add(eps)
        .rsqrt()
        .to(x.dtype)
    )
    return rms.expand(shape=x.shape)


class FilterResponseNormTlu2d(nn.Layer):
    def __init__(self, num_features, apply_act=True, eps=1e-05, rms=True, **_):
        super(FilterResponseNormTlu2d, self).__init__()
        self.apply_act = apply_act
        self.rms = rms
        self.eps = eps
        out_48 = paddle.create_parameter(
            shape=paddle.ones(shape=num_features).shape,
            dtype=paddle.ones(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=num_features)),
        )
        out_48.stop_gradient = not True
        self.weight = out_48
        out_49 = paddle.create_parameter(
            shape=paddle.zeros(shape=num_features).shape,
            dtype=paddle.zeros(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=num_features)),
        )
        out_49.stop_gradient = not True
        self.bias = out_49
        out_50 = paddle.create_parameter(
            shape=paddle.zeros(shape=num_features).shape,
            dtype=paddle.zeros(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=num_features)),
        )
        out_50.stop_gradient = not True
        self.tau = out_50 if apply_act else None
        self.reset_parameters()

    def reset_parameters(self):
        init_Constant = nn.initializer.Constant(value=1.0)
        init_Constant(self.weight)
        init_Constant = nn.initializer.Constant(value=0.0)
        init_Constant(self.bias)
        if self.tau is not None:
            init_Constant = nn.initializer.Constant(value=0.0)
            init_Constant(self.tau)

    def forward(self, x):
        _assert(x.dim() == 4, "expected 4D input")
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        x = x * inv_instance_rms(x, self.eps)
        x = x * self.weight.view(v_shape).to(dtype=x_dtype) + self.bias.view(
            v_shape
        ).to(dtype=x_dtype)
        return (
            paddle.maximum(x=x, y=self.tau.reshape(v_shape).to(dtype=x_dtype))
            if self.tau is not None
            else x
        )


class FilterResponseNormAct2d(nn.Layer):
    def __init__(
        self,
        num_features,
        apply_act=True,
        act_layer=nn.ReLU,
        inplace=None,
        rms=True,
        eps=1e-05,
        **_
    ):
        super(FilterResponseNormAct2d, self).__init__()
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer, inplace=inplace)
        else:
            self.act = nn.Identity()
        self.rms = rms
        self.eps = eps
        out_51 = paddle.create_parameter(
            shape=paddle.ones(shape=num_features).shape,
            dtype=paddle.ones(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=num_features)),
        )
        out_51.stop_gradient = not True
        self.weight = out_51
        out_52 = paddle.create_parameter(
            shape=paddle.zeros(shape=num_features).shape,
            dtype=paddle.zeros(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=num_features)),
        )
        out_52.stop_gradient = not True
        self.bias = out_52
        self.reset_parameters()

    def reset_parameters(self):
        init_Constant = nn.initializer.Constant(value=1.0)
        init_Constant(self.weight)
        init_Constant = nn.initializer.Constant(value=0.0)
        init_Constant(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, "expected 4D input")
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        x = x * inv_instance_rms(x, self.eps)
        x = x * self.weight.view(v_shape).to(dtype=x_dtype) + self.bias.view(
            v_shape
        ).to(dtype=x_dtype)
        return self.act(x)
