""" EvoNorm in PyTorch

Based on `Evolving Normalization-Activation Layers` - https://arxiv.org/abs/2004.02967
@inproceedings{NEURIPS2020,
 author = {Liu, Hanxiao and Brock, Andy and Simonyan, Karen and Le, Quoc},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {13539--13550},
 publisher = {Curran Associates, Inc.},
 title = {Evolving Normalization-Activation Layers},
 url = {https://proceedings.neurips.cc/paper/2020/file/9d4c03631b8b0c85ae08bf05eda37d0f-Paper.pdf},
 volume = {33},
 year = {2020}
}

An attempt at getting decent performing EvoNorms running in PyTorch.
While faster than other PyTorch impl, still quite a ways off the built-in BatchNorm
in terms of memory usage and throughput on GPUs.

I'm testing these modules on TPU w/ PyTorch XLA. Promising start but
currently working around some issues with builtin torch/tensor.var/std. Unlike
GPU, similar train speeds for EvoNormS variants and BatchNorm.

Hacked together by / Copyright 2020 Ross Wightman
"""
import paddle
from paddle import nn
from typing import Sequence, Union
from .create_act import create_act_layer
from .trace_utils import _assert


def instance_std(x, eps: float = 1e-05):
    std = (
        x.astype(dtype="float32")
        .var(axis=(2, 3), unbiased=False, keepdim=True)
        .add(eps)
        .sqrt()
        .to(x.dtype)
    )
    return std.expand(shape=x.shape)


def instance_std_tpu(x, eps: float = 1e-05):
    std = manual_var(x, dim=(2, 3)).add(eps).sqrt()
    return std.expand(shape=x.shape)


def instance_rms(x, eps: float = 1e-05):
    rms = (
        x.astype(dtype="float32")
        .square()
        .mean(axis=(2, 3), keepdim=True)
        .add(eps)
        .sqrt()
        .to(x.dtype)
    )
    return rms.expand(shape=x.shape)


def manual_var(x, dim: Union[int, Sequence[int]], diff_sqm: bool = False):
    xm = x.mean(axis=dim, keepdim=True)
    if diff_sqm:
        var = ((x * x).mean(axis=dim, keepdim=True) - xm * xm).clip(min=0)
    else:
        var = ((x - xm) * (x - xm)).mean(axis=dim, keepdim=True)
    return var


def group_std(x, groups: int = 32, eps: float = 1e-05, flatten: bool = False):
    B, C, H, W = x.shape
    x_dtype = x.dtype
    _assert(C % groups == 0, "")
    if flatten:
        x = x.reshape(B, groups, -1)
        std = (
            x.astype(dtype="float32")
            .var(axis=2, unbiased=False, keepdim=True)
            .add(eps)
            .sqrt()
            .to(x_dtype)
        )
    else:
        x = x.reshape(B, groups, C // groups, H, W)
        std = (
            x.astype(dtype="float32")
            .var(axis=(2, 3, 4), unbiased=False, keepdim=True)
            .add(eps)
            .sqrt()
            .to(x_dtype)
        )
    return std.expand(shape=x.shape).reshape(B, C, H, W)


def group_std_tpu(
    x,
    groups: int = 32,
    eps: float = 1e-05,
    diff_sqm: bool = False,
    flatten: bool = False,
):
    B, C, H, W = x.shape
    _assert(C % groups == 0, "")
    if flatten:
        x = x.reshape(B, groups, -1)
        var = manual_var(x, dim=-1, diff_sqm=diff_sqm)
    else:
        x = x.reshape(B, groups, C // groups, H, W)
        var = manual_var(x, dim=(2, 3, 4), diff_sqm=diff_sqm)
    return var.add(eps).sqrt().expand(shape=x.shape).reshape(B, C, H, W)


def group_rms(x, groups: int = 32, eps: float = 1e-05):
    B, C, H, W = x.shape
    _assert(C % groups == 0, "")
    x_dtype = x.dtype
    x = x.reshape(B, groups, C // groups, H, W)
    rms = (
        x.astype(dtype="float32")
        .square()
        .mean(axis=(2, 3, 4), keepdim=True)
        .add(eps)
        .sqrt_()
        .to(x_dtype)
    )
    return rms.expand(shape=x.shape).reshape(B, C, H, W)


class EvoNorm2dB0(nn.Layer):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=0.001, **_):
        super().__init__()
        self.apply_act = apply_act
        self.momentum = momentum
        self.eps = eps
        out_14 = paddle.create_parameter(
            shape=paddle.ones(shape=num_features).shape,
            dtype=paddle.ones(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=num_features)),
        )
        out_14.stop_gradient = not True
        self.weight = out_14
        out_15 = paddle.create_parameter(
            shape=paddle.zeros(shape=num_features).shape,
            dtype=paddle.zeros(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=num_features)),
        )
        out_15.stop_gradient = not True
        self.bias = out_15
        out_16 = paddle.create_parameter(
            shape=paddle.ones(shape=num_features).shape,
            dtype=paddle.ones(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=num_features)),
        )
        out_16.stop_gradient = not True
        self.v = out_16 if apply_act else None
        self.register_buffer(name="running_var", tensor=paddle.ones(shape=num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init_Constant = nn.initializer.Constant(value=1.0)
        init_Constant(self.weight)
        init_Constant = nn.initializer.Constant(value=0.0)
        init_Constant(self.bias)
        if self.v is not None:
            init_Constant = nn.initializer.Constant(value=1.0)
            init_Constant(self.v)

    def forward(self, x):
        _assert(x.dim() == 4, "expected 4D input")
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        if self.v is not None:
            if self.training:
                var = x.astype(dtype="float32").var(axis=(0, 2, 3), unbiased=False)
                n = x.size / x.shape[1]
                paddle.assign(
                    self.running_var * (1 - self.momentum)
                    + var.detach() * self.momentum * (n / (n - 1)),
                    output=self.running_var,
                )
            else:
                var = self.running_var
            left = var.add(self.eps).sqrt_().to(x_dtype).view(v_shape).expand_as(y=x)
            v = self.v.to(x_dtype).view(v_shape)
            right = x * v + instance_std(x, self.eps)
            x = x / left.max(right)
        return x * self.weight.to(x_dtype).view(v_shape) + self.bias.to(x_dtype).view(
            v_shape
        )


class EvoNorm2dB1(nn.Layer):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-05, **_):
        super().__init__()
        self.apply_act = apply_act
        self.momentum = momentum
        self.eps = eps
        out_17 = paddle.create_parameter(
            shape=paddle.ones(shape=num_features).shape,
            dtype=paddle.ones(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=num_features)),
        )
        out_17.stop_gradient = not True
        self.weight = out_17
        out_18 = paddle.create_parameter(
            shape=paddle.zeros(shape=num_features).shape,
            dtype=paddle.zeros(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=num_features)),
        )
        out_18.stop_gradient = not True
        self.bias = out_18
        self.register_buffer(name="running_var", tensor=paddle.ones(shape=num_features))
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
        if self.apply_act:
            if self.training:
                var = x.astype(dtype="float32").var(axis=(0, 2, 3), unbiased=False)
                n = x.size / x.shape[1]
                paddle.assign(
                    self.running_var * (1 - self.momentum)
                    + var.detach().to(self.running_var.dtype)
                    * self.momentum
                    * (n / (n - 1)),
                    output=self.running_var,
                )
            else:
                var = self.running_var
            var = var.to(x_dtype).view(v_shape)
            left = var.add(self.eps).sqrt_()
            right = (x + 1) * instance_rms(x, self.eps)
            x = x / left.max(right)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(
            x_dtype
        )


class EvoNorm2dB2(nn.Layer):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-05, **_):
        super().__init__()
        self.apply_act = apply_act
        self.momentum = momentum
        self.eps = eps
        out_19 = paddle.create_parameter(
            shape=paddle.ones(shape=num_features).shape,
            dtype=paddle.ones(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=num_features)),
        )
        out_19.stop_gradient = not True
        self.weight = out_19
        out_20 = paddle.create_parameter(
            shape=paddle.zeros(shape=num_features).shape,
            dtype=paddle.zeros(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=num_features)),
        )
        out_20.stop_gradient = not True
        self.bias = out_20
        self.register_buffer(name="running_var", tensor=paddle.ones(shape=num_features))
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
        if self.apply_act:
            if self.training:
                var = x.astype(dtype="float32").var(axis=(0, 2, 3), unbiased=False)
                n = x.size / x.shape[1]
                paddle.assign(
                    self.running_var * (1 - self.momentum)
                    + var.detach().to(self.running_var.dtype)
                    * self.momentum
                    * (n / (n - 1)),
                    output=self.running_var,
                )
            else:
                var = self.running_var
            var = var.to(x_dtype).view(v_shape)
            left = var.add(self.eps).sqrt_()
            right = instance_rms(x, self.eps) - x
            x = x / left.max(right)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(
            x_dtype
        )


class EvoNorm2dS0(nn.Layer):
    def __init__(
        self, num_features, groups=32, group_size=None, apply_act=True, eps=1e-05, **_
    ):
        super().__init__()
        self.apply_act = apply_act
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        out_21 = paddle.create_parameter(
            shape=paddle.ones(shape=num_features).shape,
            dtype=paddle.ones(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=num_features)),
        )
        out_21.stop_gradient = not True
        self.weight = out_21
        out_22 = paddle.create_parameter(
            shape=paddle.zeros(shape=num_features).shape,
            dtype=paddle.zeros(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=num_features)),
        )
        out_22.stop_gradient = not True
        self.bias = out_22
        out_23 = paddle.create_parameter(
            shape=paddle.ones(shape=num_features).shape,
            dtype=paddle.ones(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=num_features)),
        )
        out_23.stop_gradient = not True
        self.v = out_23 if apply_act else None
        self.reset_parameters()

    def reset_parameters(self):
        init_Constant = nn.initializer.Constant(value=1.0)
        init_Constant(self.weight)
        init_Constant = nn.initializer.Constant(value=0.0)
        init_Constant(self.bias)
        if self.v is not None:
            init_Constant = nn.initializer.Constant(value=1.0)
            init_Constant(self.v)

    def forward(self, x):
        _assert(x.dim() == 4, "expected 4D input")
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        if self.v is not None:
            v = self.v.view(v_shape).to(x_dtype)
            x = x * (x * v).sigmoid() / group_std(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(
            x_dtype
        )


class EvoNorm2dS0a(EvoNorm2dS0):
    def __init__(
        self, num_features, groups=32, group_size=None, apply_act=True, eps=0.001, **_
    ):
        super().__init__(
            num_features,
            groups=groups,
            group_size=group_size,
            apply_act=apply_act,
            eps=eps,
        )

    def forward(self, x):
        _assert(x.dim() == 4, "expected 4D input")
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        d = group_std(x, self.groups, self.eps)
        if self.v is not None:
            v = self.v.view(v_shape).to(x_dtype)
            x = x * (x * v).sigmoid()
        x = x / d
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(
            x_dtype
        )


class EvoNorm2dS1(nn.Layer):
    def __init__(
        self,
        num_features,
        groups=32,
        group_size=None,
        apply_act=True,
        act_layer=None,
        eps=1e-05,
        **_
    ):
        super().__init__()
        act_layer = act_layer or nn.Silu
        self.apply_act = apply_act
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer)
        else:
            self.act = nn.Identity()
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.pre_act_norm = False
        out_24 = paddle.create_parameter(
            shape=paddle.ones(shape=num_features).shape,
            dtype=paddle.ones(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=num_features)),
        )
        out_24.stop_gradient = not True
        self.weight = out_24
        out_25 = paddle.create_parameter(
            shape=paddle.zeros(shape=num_features).shape,
            dtype=paddle.zeros(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=num_features)),
        )
        out_25.stop_gradient = not True
        self.bias = out_25
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
        if self.apply_act:
            x = self.act(x) / group_std(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(
            x_dtype
        )


class EvoNorm2dS1a(EvoNorm2dS1):
    def __init__(
        self,
        num_features,
        groups=32,
        group_size=None,
        apply_act=True,
        act_layer=None,
        eps=0.001,
        **_
    ):
        super().__init__(
            num_features,
            groups=groups,
            group_size=group_size,
            apply_act=apply_act,
            act_layer=act_layer,
            eps=eps,
        )

    def forward(self, x):
        _assert(x.dim() == 4, "expected 4D input")
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        x = self.act(x) / group_std(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(
            x_dtype
        )


class EvoNorm2dS2(nn.Layer):
    def __init__(
        self,
        num_features,
        groups=32,
        group_size=None,
        apply_act=True,
        act_layer=None,
        eps=1e-05,
        **_
    ):
        super().__init__()
        act_layer = act_layer or nn.Silu
        self.apply_act = apply_act
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer)
        else:
            self.act = nn.Identity()
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        out_26 = paddle.create_parameter(
            shape=paddle.ones(shape=num_features).shape,
            dtype=paddle.ones(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=num_features)),
        )
        out_26.stop_gradient = not True
        self.weight = out_26
        out_27 = paddle.create_parameter(
            shape=paddle.zeros(shape=num_features).shape,
            dtype=paddle.zeros(shape=num_features).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=num_features)),
        )
        out_27.stop_gradient = not True
        self.bias = out_27
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
        if self.apply_act:
            x = self.act(x) / group_rms(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(
            x_dtype
        )


class EvoNorm2dS2a(EvoNorm2dS2):
    def __init__(
        self,
        num_features,
        groups=32,
        group_size=None,
        apply_act=True,
        act_layer=None,
        eps=0.001,
        **_
    ):
        super().__init__(
            num_features,
            groups=groups,
            group_size=group_size,
            apply_act=apply_act,
            act_layer=act_layer,
            eps=eps,
        )

    def forward(self, x):
        _assert(x.dim() == 4, "expected 4D input")
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        x = self.act(x) / group_rms(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(
            x_dtype
        )
