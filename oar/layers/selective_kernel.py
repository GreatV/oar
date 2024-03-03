""" Selective Kernel Convolution/Attention

Paper: Selective Kernel Networks (https://arxiv.org/abs/1903.06586)

Hacked together by / Copyright 2020 Ross Wightman
"""

import paddle
from .conv_bn_act import ConvNormActAa
from .helpers import make_divisible
from .trace_utils import _assert


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis] // num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


def _kernel_valid(k):
    if isinstance(k, (list, tuple)):
        for ki in k:
            return _kernel_valid(ki)
    assert k >= 3 and k % 2


class SelectiveKernelAttn(paddle.nn.Layer):

    def __init__(
        self,
        channels,
        num_paths=2,
        attn_channels=32,
        act_layer=paddle.nn.ReLU,
        norm_layer=paddle.nn.BatchNorm2D,
    ):
        """Selective Kernel Attention Module

        Selective Kernel attention mechanism factored out into its own module.

        """
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.fc_reduce = paddle.nn.Conv2D(
            in_channels=channels,
            out_channels=attn_channels,
            kernel_size=1,
            bias_attr=False,
        )
        self.bn = norm_layer(attn_channels)
        self.act = act_layer(inplace=True)
        self.fc_select = paddle.nn.Conv2D(
            in_channels=attn_channels,
            out_channels=channels * num_paths,
            kernel_size=1,
            bias_attr=False,
        )

    def forward(self, x):
        _assert(x.shape[1] == self.num_paths, "")
        x = x.sum(axis=1).mean(axis=(2, 3), keepdim=True)
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        B, C, H, W = x.shape
        x = x.view(B, self.num_paths, C // self.num_paths, H, W)
        x = paddle.nn.functional.softmax(x=x, axis=1)
        return x


class SelectiveKernel(paddle.nn.Layer):

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=None,
        stride=1,
        dilation=1,
        groups=1,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=8,
        keep_3x3=True,
        split_input=True,
        act_layer=paddle.nn.ReLU,
        norm_layer=paddle.nn.BatchNorm2D,
        aa_layer=None,
        drop_layer=None,
    ):
        """Selective Kernel Convolution Module

        As described in Selective Kernel Networks (https://arxiv.org/abs/1903.06586) with some modifications.

        Largest change is the input split, which divides the input channels across each convolution path, this can
        be viewed as a grouping of sorts, but the output channel counts expand to the module level value. This keeps
        the parameter count from ballooning when the convolutions themselves don't have groups, but still provides
        a noteworthy increase in performance over similar param count models without this attention layer. -Ross W

        Args:
            in_channels (int):  module input (feature) channel count
            out_channels (int):  module output (feature) channel count
            kernel_size (int, list): kernel size for each convolution branch
            stride (int): stride for convolutions
            dilation (int): dilation for module as a whole, impacts dilation of each branch
            groups (int): number of groups for each branch
            rd_ratio (int, float): reduction factor for attention features
            keep_3x3 (bool): keep all branch convolution kernels as 3x3, changing larger kernels for dilations
            split_input (bool): split input channels evenly across each convolution branch, keeps param count lower,
                can be viewed as grouping by path, output expands to module out_channels count
            act_layer (nn.Module): activation layer to use
            norm_layer (nn.Module): batchnorm/norm layer to use
            aa_layer (nn.Module): anti-aliasing module
            drop_layer (nn.Module): spatial drop module in convs (drop block, etc)
        """
        super(SelectiveKernel, self).__init__()
        out_channels = out_channels or in_channels
        kernel_size = kernel_size or [3, 5]
        _kernel_valid(kernel_size)
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [(dilation * (k - 1) // 2) for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        self.num_paths = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_input = split_input
        if self.split_input:
            assert in_channels % self.num_paths == 0
            in_channels = in_channels // self.num_paths
        groups = min(out_channels, groups)
        conv_kwargs = dict(
            stride=stride,
            groups=groups,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_layer=drop_layer,
        )
        self.paths = paddle.nn.LayerList(
            sublayers=[
                ConvNormActAa(
                    in_channels, out_channels, kernel_size=k, dilation=d, **conv_kwargs
                )
                for k, d in zip(kernel_size, dilation)
            ]
        )
        attn_channels = rd_channels or make_divisible(
            out_channels * rd_ratio, divisor=rd_divisor
        )
        self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)

    def forward(self, x):
        if self.split_input:
            x_split = split(
                x=x, num_or_sections=self.in_channels // self.num_paths, axis=1
            )
            x_paths = [op(x_split[i]) for i, op in enumerate(self.paths)]
        else:
            x_paths = [op(x) for op in self.paths]
        x = paddle.stack(x=x_paths, axis=1)
        x_attn = self.attn(x)
        x = x * x_attn
        x = paddle.sum(x=x, axis=1)
        return x
