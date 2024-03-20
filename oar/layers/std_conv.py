"""Convolution with Weight Standardization (StdConv and ScaledStdConv)

StdConv:
@article{weightstandardization,
  author    = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Yuille},
  title     = {Weight Standardization},
  journal   = {arXiv preprint arXiv:1903.10520},
  year      = {2019},
}
Code: https://github.com/joe-siyuan-qiao/WeightStandardization

ScaledStdConv:
Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
    - https://arxiv.org/abs/2101.08692
Official Deepmind JAX code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

Hacked together by / copyright Ross Wightman, 2021.
"""

import paddle
from paddle import nn
from paddle.nn import functional as F
from .padding import get_padding, get_padding_value, pad_same


class StdConv2d(nn.Conv2D):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
        self,
        in_channel,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=False,
        eps=1e-06,
    ):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channel,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(
            x=self.weight.reshape(1, self.out_channels, -1),
            running_mean=None,
            running_var=None,
            training=True,
            epsilon=self.eps,
            momentum=1 - 0.0,
        ).reshape(self.weight.shape)
        x = F.conv2d(
            x=x,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return x


class StdConv2dSame(nn.Conv2D):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for ViT Hybrid model.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
        self,
        in_channel,
        out_channels,
        kernel_size,
        stride=1,
        padding="SAME",
        dilation=1,
        groups=1,
        bias=False,
        eps=1e-06,
    ):
        padding, is_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation
        )
        super().__init__(
            in_channel,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.same_pad = is_dynamic
        self.eps = eps

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = F.batch_norm(
            x=self.weight.reshape(1, self.out_channels, -1),
            running_mean=None,
            running_var=None,
            training=True,
            epsilon=self.eps,
            momentum=1 - 0.0,
        ).reshape(self.weight.shape)
        x = F.conv2d(
            x=x,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return x


class ScaledStdConv2d(nn.Conv2D):
    """Conv2d layer with Scaled Weight Standardization.

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=True,
        gamma=1.0,
        eps=1e-06,
        gain_init=1.0,
    ):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        out_31 = paddle.create_parameter(
            shape=paddle.full(
                shape=(self.out_channels, 1, 1, 1), fill_value=gain_init
            ).shape,
            dtype=paddle.full(shape=(self.out_channels, 1, 1, 1), fill_value=gain_init)
            .numpy()
            .dtype,
            default_initializer=nn.initializer.Assign(
                paddle.full(shape=(self.out_channels, 1, 1, 1), fill_value=gain_init)
            ),
        )
        out_31.stop_gradient = not True
        self.gain = out_31
        self.scale = gamma * self.weight[0].size ** -0.5
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(
            x=self.weight.reshape(1, self.out_channels, -1),
            running_mean=None,
            running_var=None,
            weight=(self.gain * self.scale).view(-1),
            training=True,
            epsilon=self.eps,
            momentum=1 - 0.0,
        ).reshape(self.weight.shape)
        return F.conv2d(
            x=x,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class ScaledStdConv2dSame(nn.Conv2D):
    """Conv2d layer with Scaled Weight Standardization and Tensorflow-like SAME padding support

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="SAME",
        dilation=1,
        groups=1,
        bias=True,
        gamma=1.0,
        eps=1e-06,
        gain_init=1.0,
    ):
        padding, is_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation
        )
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        out_32 = paddle.create_parameter(
            shape=paddle.full(
                shape=(self.out_channels, 1, 1, 1), fill_value=gain_init
            ).shape,
            dtype=paddle.full(shape=(self.out_channels, 1, 1, 1), fill_value=gain_init)
            .numpy()
            .dtype,
            default_initializer=nn.initializer.Assign(
                paddle.full(shape=(self.out_channels, 1, 1, 1), fill_value=gain_init)
            ),
        )
        out_32.stop_gradient = not True
        self.gain = out_32
        self.scale = gamma * self.weight[0].size ** -0.5
        self.same_pad = is_dynamic
        self.eps = eps

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = F.batch_norm(
            x=self.weight.reshape(1, self.out_channels, -1),
            running_mean=None,
            running_var=None,
            weight=(self.gain * self.scale).view(-1),
            training=True,
            epsilon=self.eps,
            momentum=1 - 0.0,
        ).reshape(self.weight.shape)
        return F.conv2d(
            x=x,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
