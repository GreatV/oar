""" Global Context Attention Block

Paper: `GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond`
    - https://arxiv.org/abs/1904.11492

Official code consulted as reference: https://github.com/xvjiarui/GCNet

Hacked together by / Copyright 2021 Ross Wightman
"""
from paddle import nn
from paddle.nn import functional as F
from .create_act import create_act_layer, get_act_layer
from .helpers import make_divisible
from .mlp import ConvMlp
from .norm import LayerNorm2d


class GlobalContext(nn.Layer):
    def __init__(
        self,
        channels,
        use_attn=True,
        fuse_add=False,
        fuse_scale=True,
        init_last_zero=False,
        rd_ratio=1.0 / 8,
        rd_channels=None,
        rd_divisor=1,
        act_layer=nn.ReLU,
        gate_layer="sigmoid",
    ):
        super(GlobalContext, self).__init__()
        act_layer = get_act_layer(act_layer)
        self.conv_attn = (
            nn.Conv2D(
                in_channels=channels, out_channels=1, kernel_size=1, bias_attr=True
            )
            if use_attn
            else None
        )
        if rd_channels is None:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        if fuse_add:
            self.mlp_add = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_scale = None
        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            init_KaimingNormal = nn.initializer.KaimingNormal(nonlinearity="relu")
            init_KaimingNormal(self.conv_attn.weight)
        if self.mlp_add is not None:
            init_Constant = nn.initializer.Constant(value=0.0)
            init_Constant(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)
            attn = F.softmax(x=attn, axis=-1).unsqueeze(axis=3)
            context = x.reshape(B, C, H * W).unsqueeze(axis=1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(axis=(2, 3), keepdim=True)
        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x
        return x
