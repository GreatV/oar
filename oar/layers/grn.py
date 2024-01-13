""" Global Response Normalization Module

Based on the GRN layer presented in
`ConvNeXt-V2 - Co-designing and Scaling ConvNets with Masked Autoencoders` - https://arxiv.org/abs/2301.00808

This implementation
* works for both NCHW and NHWC tensor layouts
* uses affine param names matching existing torch norm layers
* slightly improves eager mode performance via fused addcmul

Hacked together by / Copyright 2023 Ross Wightman
"""
import paddle
from paddle import nn
from paddle.nn import initializer as init


class GlobalResponseNorm(nn.Layer):
    """Global Response Normalization layer"""

    def __init__(self, dim, eps=1e-06, channels_last=True):
        super().__init__()
        self.eps = eps
        if channels_last:
            self.spatial_dim = 1, 2
            self.channel_dim = -1
            self.wb_shape = 1, 1, 1, -1
        else:
            self.spatial_dim = 2, 3
            self.channel_dim = 1
            self.wb_shape = 1, -1, 1, 1
        out_28 = paddle.create_parameter(
            shape=paddle.zeros(shape=dim).shape,
            dtype=paddle.zeros(shape=dim).numpy().dtype,
            default_initializer=init.Assign(paddle.zeros(shape=dim)),
        )
        out_28.stop_gradient = not True
        self.weight = out_28
        out_29 = paddle.create_parameter(
            shape=paddle.zeros(shape=dim).shape,
            dtype=paddle.zeros(shape=dim).numpy().dtype,
            default_initializer=init.Assign(paddle.zeros(shape=dim)),
        )
        out_29.stop_gradient = not True
        self.bias = out_29

    def forward(self, x):
        x_g = x.norm(p=2, axis=self.spatial_dim, keepdim=True)
        x_n = x_g / (x_g.mean(axis=self.channel_dim, keepdim=True) + self.eps)
        return x + paddle.add(
            self.bias.reshape(self.wb_shape),
            1 * self.weight.reshape(self.wb_shape) * (x * x_n),
        )
