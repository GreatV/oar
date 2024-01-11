""" Bottleneck Self Attention (Bottleneck Transformers)

Paper: `Bottleneck Transformers for Visual Recognition` - https://arxiv.org/abs/2101.11605

@misc{2101.11605,
Author = {Aravind Srinivas and Tsung-Yi Lin and Niki Parmar and Jonathon Shlens and Pieter Abbeel and Ashish Vaswani},
Title = {Bottleneck Transformers for Visual Recognition},
Year = {2021},
}

Based on ref gist at: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2

This impl is a WIP but given that it is based on the ref gist likely not too far off.

Hacked together by / Copyright 2021 Ross Wightman
"""
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as init
from typing import List
from .helpers import to_2tuple, make_divisible
from .weight_init import trunc_normal_
from .trace_utils import _assert


def _FUNCTIONAL_PAD(x, pad, mode="constant", value=0.0, data_format="NCHW"):
    if len(x.shape) * 2 == len(pad) and mode == "constant":
        pad = (
            paddle.to_tensor(pad, dtype="int32")
            .reshape((-1, 2))
            .flip([0])
            .flatten()
            .tolist()
        )
    return F.pad(x, pad, mode, value, data_format)


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis] // num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


def rel_logits_1d(q, rel_k, permute_mask: List[int]):
    """Compute relative logits along one dimension

    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    Args:
        q: (batch, heads, height, width, dim)
        rel_k: (2 * width - 1, dim)
        permute_mask: permute output dim according to this
    """
    B, H, W, dim = q.shape
    x = rel_k
    perm_6 = list(range(x.ndim))
    perm_6[-1] = -2
    perm_6[-2] = -1
    x = q @ x.transpose(perm=perm_6)
    x = x.reshape(-1, W, 2 * W - 1)
    x_pad = _FUNCTIONAL_PAD(pad=[0, 1], x=x).flatten(start_axis=1)
    x_pad = _FUNCTIONAL_PAD(pad=[0, W - 1], x=x_pad)
    x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x = x_pad[:, :W, W - 1 :]
    x = x.reshape(B, H, 1, W, W).expand(shape=[-1, -1, H, -1, -1])
    return x.transpose(perm=permute_mask)


class PosEmbedRel(nn.Layer):
    """Relative Position Embedding
    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925
    """

    def __init__(self, feat_size, dim_head, scale):
        super().__init__()
        self.height, self.width = to_2tuple(feat_size)
        self.dim_head = dim_head
        out_33 = paddle.create_parameter(
            shape=(paddle.randn(shape=[self.height * 2 - 1, dim_head]) * scale).shape,
            dtype=(paddle.randn(shape=[self.height * 2 - 1, dim_head]) * scale)
            .numpy()
            .dtype,
            default_initializer=init.Assign(
                paddle.randn(shape=[self.height * 2 - 1, dim_head]) * scale
            ),
        )
        out_33.stop_gradient = not True
        self.height_rel = out_33
        out_34 = paddle.create_parameter(
            shape=(paddle.randn(shape=[self.width * 2 - 1, dim_head]) * scale).shape,
            dtype=(paddle.randn(shape=[self.width * 2 - 1, dim_head]) * scale)
            .numpy()
            .dtype,
            default_initializer=init.Assign(
                paddle.randn(shape=[self.width * 2 - 1, dim_head]) * scale
            ),
        )
        out_34.stop_gradient = not True
        self.width_rel = out_34

    def forward(self, q):
        B, HW, _ = q.shape
        q = q.reshape(B, self.height, self.width, -1)
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))
        x = q
        perm_7 = list(range(x.ndim))
        perm_7[1] = 2
        perm_7[2] = 1
        q = x.transpose(perm=perm_7)
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))
        rel_logits = rel_logits_h + rel_logits_w
        rel_logits = rel_logits.reshape(B, HW, HW)
        return rel_logits


class BottleneckAttn(nn.Layer):
    """Bottleneck Attention
    Paper: `Bottleneck Transformers for Visual Recognition` - https://arxiv.org/abs/2101.11605

    The internal dimensions of the attention module are controlled by the interaction of several arguments.
      * the output dimension of the module is specified by dim_out, which falls back to input dim if not set
      * the value (v) dimension is set to dim_out // num_heads, the v projection determines the output dim
      * the query and key (qk) dimensions are determined by
        * num_heads * dim_head if dim_head is not None
        * num_heads * (dim_out * attn_ratio // num_heads) if dim_head is None
      * as seen above, attn_ratio determines the ratio of q and k relative to the output if dim_head not used

    Args:
        dim (int): input dimension to the module
        dim_out (int): output dimension of the module, same as dim if not set
        stride (int): output stride of the module, avg pool used if stride == 2 (default: 1).
        num_heads (int): parallel attention heads (default: 4)
        dim_head (int): dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool): add bias to q, k, and v projections
        scale_pos_embed (bool): scale the position embedding as well as Q @ K
    """

    def __init__(
        self,
        dim,
        dim_out=None,
        feat_size=None,
        stride=1,
        num_heads=4,
        dim_head=None,
        qk_ratio=1.0,
        qkv_bias=False,
        scale_pos_embed=False,
    ):
        super().__init__()
        assert (
            feat_size is not None
        ), "A concrete feature size matching expected input (H, W) is required"
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        self.num_heads = num_heads
        self.dim_head_qk = (
            dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        )
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk**-0.5
        self.scale_pos_embed = scale_pos_embed
        self.qkv = nn.Conv2D(
            in_channels=dim,
            out_channels=self.dim_out_qk * 2 + self.dim_out_v,
            kernel_size=1,
            bias_attr=qkv_bias,
        )
        self.pos_embed = PosEmbedRel(
            feat_size, dim_head=self.dim_head_qk, scale=self.scale
        )
        self.pool = (
            nn.AvgPool2D(kernel_size=2, stride=2, exclusive=False)
            if stride == 2
            else nn.Identity()
        )
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.qkv.weight, std=self.qkv.weight.shape[1] ** -0.5)
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.pos_embed.height, "")
        _assert(W == self.pos_embed.width, "")
        x = self.qkv(x)
        q, k, v = split(
            x=x,
            num_or_sections=[self.dim_out_qk, self.dim_out_qk, self.dim_out_v],
            axis=1,
        )
        x = q.reshape(B * self.num_heads, self.dim_head_qk, -1)
        perm_8 = list(range(x.ndim))
        perm_8[-1] = -2
        perm_8[-2] = -1
        q = x.transpose(perm=perm_8)
        k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)
        x = v.reshape(B * self.num_heads, self.dim_head_v, -1)
        perm_9 = list(range(x.ndim))
        perm_9[-1] = -2
        perm_9[-2] = -1
        v = x.transpose(perm=perm_9)
        if self.scale_pos_embed:
            attn = (q @ k + self.pos_embed(q)) * self.scale
        else:
            attn = q @ k * self.scale + self.pos_embed(q)
        attn = F.softmax(attn, axis=-1)
        x = attn @ v
        perm_10 = list(range(x.ndim))
        perm_10[-1] = -2
        perm_10[-2] = -1
        out = x.transpose(perm=perm_10).reshape(B, self.dim_out_v, H, W)
        out = self.pool(out)
        return out
