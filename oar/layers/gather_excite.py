""" Gather-Excite Attention Block

Paper: `Gather-Excite: Exploiting Feature Context in CNNs` - https://arxiv.org/abs/1810.12348

Official code here, but it's only partial impl in Caffe: https://github.com/hujie-frank/GENet

I've tried to support all of the extent both w/ and w/o params. I don't believe I've seen another
impl that covers all of the cases.

NOTE: extent=0 + extra_params=False is equivalent to Squeeze-and-Excitation

Hacked together by / Copyright 2021 Ross Wightman
"""
from paddle import nn
from paddle.nn import functional as F
import math
from .create_act import create_act_layer, get_act_layer
from .create_conv2d import create_conv2d
from .helpers import make_divisible
from .mlp import ConvMlp


class GatherExcite(nn.Layer):
    """Gather-Excite Attention Module"""

    def __init__(
        self,
        channels,
        feat_size=None,
        extra_params=False,
        extent=0,
        use_mlp=True,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=1,
        add_maxpool=False,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2D,
        gate_layer="sigmoid",
    ):
        super(GatherExcite, self).__init__()
        self.add_maxpool = add_maxpool
        act_layer = get_act_layer(act_layer)
        self.extent = extent
        if extra_params:
            self.gather = nn.Sequential()
            if extent == 0:
                assert (
                    feat_size is not None
                ), "spatial feature size must be specified for global extent w/ params"
                self.gather.add_sublayer(
                    name="conv1",
                    sublayer=create_conv2d(
                        channels,
                        channels,
                        kernel_size=feat_size,
                        stride=1,
                        depthwise=True,
                    ),
                )
                if norm_layer:
                    self.gather.add_sublayer(
                        name=f"norm1",
                        sublayer=nn.BatchNorm2D(num_features=channels),
                    )
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    self.gather.add_sublayer(
                        name=f"conv{i + 1}",
                        sublayer=create_conv2d(
                            channels, channels, kernel_size=3, stride=2, depthwise=True
                        ),
                    )
                    if norm_layer:
                        self.gather.add_sublayer(
                            name=f"norm{i + 1}",
                            sublayer=nn.BatchNorm2D(num_features=channels),
                        )
                    if i != num_conv - 1:
                        self.gather.add_sublayer(
                            name=f"act{i + 1}", sublayer=act_layer(inplace=True)
                        )
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent
        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        self.mlp = (
            ConvMlp(channels, rd_channels, act_layer=act_layer)
            if use_mlp
            else nn.Identity()
        )
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        size = x.shape[-2:]
        if self.gather is not None:
            x_ge = self.gather(x)
        elif self.extent == 0:
            x_ge = x.mean(dim=(2, 3), keepdims=True)
            if self.add_maxpool:
                x_ge = 0.5 * x_ge + 0.5 * x.amax(axis=(2, 3), keepdim=True)
        else:
            x_ge = F.avg_pool2d(
                kernel_size=self.gk,
                stride=self.gs,
                padding=self.gk // 2,
                x=x,
                exclusive=not False,
            )
            if self.add_maxpool:
                x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(
                    x=x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2
                )
        x_ge = self.mlp(x_ge)
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x=x_ge, size=size)
        return x * self.gate(x_ge)
