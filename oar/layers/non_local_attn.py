""" Bilinear-Attention-Transform and Non-Local Attention

Paper: `Non-Local Neural Networks With Grouped Bilinear Attentional Transforms`
    - https://openaccess.thecvf.com/content_CVPR_2020/html/Chi_Non-Local_Neural_Networks_With_Grouped_Bilinear_Attentional_Transforms_CVPR_2020_paper.html
Adapted from original code: https://github.com/BA-Transform/BAT-Image-Classification
"""

import paddle
from paddle import nn
from paddle.nn import functional as F
from .conv_bn_act import ConvNormAct
from .helpers import make_divisible
from .trace_utils import _assert


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis] // num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


class NonLocalAttn(nn.Layer):
    """Spatial NL block for image classification.

    This was adapted from https://github.com/BA-Transform/BAT-Image-Classification
    Their NonLocal impl inspired by https://github.com/facebookresearch/video-nonlocal-net.
    """

    def __init__(
        self,
        in_channels,
        use_scale=True,
        rd_ratio=1 / 8,
        rd_channels=None,
        rd_divisor=8,
        **kwargs
    ):
        super(NonLocalAttn, self).__init__()
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.scale = in_channels**-0.5 if use_scale else 1.0
        self.t = nn.Conv2D(
            in_channels=in_channels,
            out_channels=rd_channels,
            kernel_size=1,
            stride=1,
            bias_attr=True,
        )
        self.p = nn.Conv2D(
            in_channels=in_channels,
            out_channels=rd_channels,
            kernel_size=1,
            stride=1,
            bias_attr=True,
        )
        self.g = nn.Conv2D(
            in_channels=in_channels,
            out_channels=rd_channels,
            kernel_size=1,
            stride=1,
            bias_attr=True,
        )
        self.z = nn.Conv2D(
            in_channels=rd_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias_attr=True,
        )
        self.norm = nn.BatchNorm2D(num_features=in_channels)
        self.reset_parameters()

    def forward(self, x):
        shortcut = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        B, C, H, W = t.shape
        t = t.view(B, C, -1).transpose(perm=[0, 2, 1])
        p = p.view(B, C, -1)
        g = g.view(B, C, -1).transpose(perm=[0, 2, 1])
        att = paddle.bmm(x=t, y=p) * self.scale
        att = F.softmax(x=att, axis=2)
        x = paddle.bmm(x=att, y=g)
        x = x.transpose(perm=[0, 2, 1]).reshape(B, C, H, W)
        x = self.z(x)
        x = self.norm(x) + shortcut
        return x

    def reset_parameters(self):
        for name, m in self.named_sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if len(list(m.parameters())) > 1:
                    nn.initializer.Constant(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2D):
                nn.initializer.Constant(m.weight, 0)
                nn.initializer.Constant(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.initializer.Constant(m.weight, 0)
                nn.initializer.Constant(m.bias, 0)


class BilinearAttnTransform(nn.Layer):

    def __init__(
        self,
        in_channels,
        block_size,
        groups,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2D,
    ):
        super(BilinearAttnTransform, self).__init__()
        self.conv1 = ConvNormAct(
            in_channels, groups, 1, act_layer=act_layer, norm_layer=norm_layer
        )
        self.conv_p = nn.Conv2D(
            in_channels=groups,
            out_channels=block_size * block_size * groups,
            kernel_size=(block_size, 1),
        )
        self.conv_q = nn.Conv2D(
            in_channels=groups,
            out_channels=block_size * block_size * groups,
            kernel_size=(1, block_size),
        )
        self.conv2 = ConvNormAct(
            in_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer
        )
        self.block_size = block_size
        self.groups = groups
        self.in_channels = in_channels

    def resize_mat(self, x, t: int):
        B, C, block_size, block_size1 = x.shape
        _assert(block_size == block_size1, "")
        if t <= 1:
            return x
        x = x.view(B * C, -1, 1, 1)
        x = x * paddle.eye(num_rows=t, num_columns=t, dtype=x.dtype)
        x = x.view(B * C, block_size, block_size, t, t)
        x = paddle.concat(x=split(x=x, num_or_sections=1, axis=1), axis=3)
        x = paddle.concat(x=split(x=x, num_or_sections=1, axis=2), axis=4)
        x = x.view(B, C, block_size * t, block_size * t)
        return x

    def forward(self, x):
        _assert(x.shape[-1] % self.block_size == 0, "")
        _assert(x.shape[-2] % self.block_size == 0, "")
        B, C, H, W = x.shape
        out = self.conv1(x)
        rp = F.adaptive_max_pool2d(x=out, output_size=(self.block_size, 1))
        cp = F.adaptive_max_pool2d(x=out, output_size=(1, self.block_size))
        p = (
            self.conv_p(rp)
            .view(B, self.groups, self.block_size, self.block_size)
            .sigmoid()
        )
        q = (
            self.conv_q(cp)
            .view(B, self.groups, self.block_size, self.block_size)
            .sigmoid()
        )
        p = p / p.sum(axis=3, keepdim=True)
        q = q / q.sum(axis=2, keepdim=True)
        p = p.view(B, self.groups, 1, self.block_size, self.block_size).expand(
            shape=[
                x.shape[0],
                self.groups,
                C // self.groups,
                self.block_size,
                self.block_size,
            ]
        )
        p = p.view(B, C, self.block_size, self.block_size)
        q = q.view(B, self.groups, 1, self.block_size, self.block_size).expand(
            shape=[
                x.shape[0],
                self.groups,
                C // self.groups,
                self.block_size,
                self.block_size,
            ]
        )
        q = q.view(B, C, self.block_size, self.block_size)
        p = self.resize_mat(p, H // self.block_size)
        q = self.resize_mat(q, W // self.block_size)
        y = p.matmul(y=x)
        y = y.matmul(y=q)
        y = self.conv2(y)
        return y


class BatNonLocalAttn(nn.Layer):
    """BAT
    Adapted from: https://github.com/BA-Transform/BAT-Image-Classification
    """

    def __init__(
        self,
        in_channels,
        block_size=7,
        groups=2,
        rd_ratio=0.25,
        rd_channels=None,
        rd_divisor=8,
        drop_rate=0.2,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2D,
        **_
    ):
        super().__init__()
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.conv1 = ConvNormAct(
            in_channels, rd_channels, 1, act_layer=act_layer, norm_layer=norm_layer
        )
        self.ba = BilinearAttnTransform(
            rd_channels, block_size, groups, act_layer=act_layer, norm_layer=norm_layer
        )
        self.conv2 = ConvNormAct(
            rd_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer
        )
        self.dropout = nn.Dropout2D(p=drop_rate)

    def forward(self, x):
        xl = self.conv1(x)
        y = self.ba(xl)
        y = self.conv2(y)
        y = self.dropout(y)
        return y + x
