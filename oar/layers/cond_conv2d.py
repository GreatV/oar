""" PyTorch Conditionally Parameterized Convolution (CondConv)

Paper: CondConv: Conditionally Parameterized Convolutions for Efficient Inference
(https://arxiv.org/abs/1904.04971)

Hacked together by / Copyright 2020 Ross Wightman
"""
import paddle
from paddle import nn
from paddle.nn import functional as F
import math
from functools import partial
import numpy as np
from .helpers import to_2tuple
from .conv2d_same import conv2d_same
from .padding import get_padding_value


def get_condconv_initializer(initializer, num_experts, expert_shape):
    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (
            len(weight.shape) != 2
            or weight.shape[0] != num_experts
            or weight.shape[1] != num_params
        ):
            raise ValueError(
                "CondConv variables must have shape [num_experts, num_params]"
            )
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))

    return condconv_initializer


class CondConv2d(nn.Layer):
    """Conditionally Parameterized Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """

    __constants__ = ["in_channels", "out_channels", "dynamic_padding"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="",
        dilation=1,
        groups=1,
        bias=False,
        num_experts=4,
    ):
        super(CondConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        padding_val, is_padding_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation
        )
        self.dynamic_padding = is_padding_dynamic
        self.padding = to_2tuple(padding_val)
        self.dilation = to_2tuple(dilation)
        self.groups = groups
        self.num_experts = num_experts
        self.weight_shape = (
            self.out_channels,
            self.in_channels // self.groups,
        ) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        out_43 = paddle.create_parameter(
            shape=paddle.empty(shape=[self.num_experts, weight_num_param]).shape,
            dtype=paddle.empty(shape=[self.num_experts, weight_num_param])
            .numpy()
            .dtype,
            default_initializer=nn.initializer.Assign(
                paddle.empty(shape=[self.num_experts, weight_num_param])
            ),
        )
        out_43.stop_gradient = not True
        self.weight = out_43
        if bias:
            self.bias_shape = (self.out_channels,)
            out_44 = paddle.create_parameter(
                shape=paddle.empty(shape=[self.num_experts, self.out_channels]).shape,
                dtype=paddle.empty(shape=[self.num_experts, self.out_channels])
                .numpy()
                .dtype,
                default_initializer=nn.initializer.Assign(
                    paddle.empty(shape=[self.num_experts, self.out_channels])
                ),
            )
            out_44.stop_gradient = not True
            self.bias = out_44
        else:
            self.add_parameter(name="bias", parameter=None)
        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(
            partial(nn.initializer.KaimingUniform, a=math.sqrt(5)),
            self.num_experts,
            self.weight_shape,
        )
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.initializer.Uniform, a=-bound, b=bound),
                self.num_experts,
                self.bias_shape,
            )
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = paddle.matmul(x=routing_weights, y=self.weight)
        new_weight_shape = (
            B * self.out_channels,
            self.in_channels // self.groups,
        ) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = paddle.matmul(x=routing_weights, y=self.bias)
            bias = bias.view(B * self.out_channels)
        x = x.reshape(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * B,
            )
        else:
            out = F.conv2d(
                x=x,
                weight=weight,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * B,
            )
        out = out.transpose(perm=[1, 0, 2, 3]).view(
            B, self.out_channels, out.shape[-2], out.shape[-1]
        )
        return out
