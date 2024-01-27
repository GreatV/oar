""" Halo Self Attention

Paper: `Scaling Local Self-Attention for Parameter Efficient Visual Backbones`
    - https://arxiv.org/abs/2103.12731

@misc{2103.12731,
Author = {Ashish Vaswani and Prajit Ramachandran and Aravind Srinivas and Niki Parmar and Blake Hechtman and
    Jonathon Shlens},
Title = {Scaling Local Self-Attention for Parameter Efficient Visual Backbones},
Year = {2021},
}

Status:
This impl is a WIP, there is no official ref impl and some details in paper weren't clear to me.
The attention mechanism works but it's slow as implemented.

Hacked together by / Copyright 2021 Ross Wightman
"""
import paddle
from paddle import nn
from paddle.nn import functional as F
from typing import List
from .helpers import make_divisible
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
        q: (batch, height, width, dim)
        rel_k: (2 * window - 1, dim)
        permute_mask: permute output dim according to this
    """
    B, H, W, dim = q.shape
    rel_size = rel_k.shape[0]
    win_size = (rel_size + 1) // 2
    x = rel_k
    perm_14 = list(range(x.ndim))
    perm_14[-1] = -2
    perm_14[-2] = -1
    x = q @ x.transpose(perm=perm_14)
    x = x.reshape(-1, W, rel_size)
    x_pad = _FUNCTIONAL_PAD(pad=[0, 1], x=x).flatten(start_axis=1)
    x_pad = _FUNCTIONAL_PAD(pad=[0, rel_size - W], x=x_pad)
    x_pad = x_pad.reshape(-1, W + 1, rel_size)
    x = x_pad[:, :W, win_size - 1 :]
    x = x.reshape(B, H, 1, W, win_size).expand(shape=[-1, -1, win_size, -1, -1])
    return x.transpose(perm=permute_mask)


class PosEmbedRel(nn.Layer):
    """Relative Position Embedding
    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    """

    def __init__(self, block_size, win_size, dim_head, scale):
        """
        Args:
            block_size (int): block size
            win_size (int): neighbourhood window size
            dim_head (int): attention head dim
            scale (float): scale factor (for init)
        """
        super().__init__()
        self.block_size = block_size
        self.dim_head = dim_head
        out_38 = paddle.create_parameter(
            shape=(paddle.randn(shape=[win_size * 2 - 1, dim_head]) * scale).shape,
            dtype=(paddle.randn(shape=[win_size * 2 - 1, dim_head]) * scale)
            .numpy()
            .dtype,
            default_initializer=nn.initializer.Assign(
                paddle.randn(shape=[win_size * 2 - 1, dim_head]) * scale
            ),
        )
        out_38.stop_gradient = not True
        self.height_rel = out_38
        out_39 = paddle.create_parameter(
            shape=(paddle.randn(shape=[win_size * 2 - 1, dim_head]) * scale).shape,
            dtype=(paddle.randn(shape=[win_size * 2 - 1, dim_head]) * scale)
            .numpy()
            .dtype,
            default_initializer=nn.initializer.Assign(
                paddle.randn(shape=[win_size * 2 - 1, dim_head]) * scale
            ),
        )
        out_39.stop_gradient = not True
        self.width_rel = out_39

    def forward(self, q):
        B, BB, HW, _ = q.shape
        q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))
        x = q
        perm_15 = list(range(x.ndim))
        perm_15[1] = 2
        perm_15[2] = 1
        q = x.transpose(perm=perm_15)
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))
        rel_logits = rel_logits_h + rel_logits_w
        rel_logits = rel_logits.reshape(B, BB, HW, -1)
        return rel_logits


class HaloAttn(nn.Layer):
    """Halo Attention

    Paper: `Scaling Local Self-Attention for Parameter Efficient Visual Backbones`
        - https://arxiv.org/abs/2103.12731

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
        feat_size (Tuple[int, int]): size of input feature_map (not used, for arg compat with bottle/lambda)
        stride: output stride of the module, query downscaled if > 1 (default: 1).
        num_heads: parallel attention heads (default: 8).
        dim_head: dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        block_size (int): size of blocks. (default: 8)
        halo_size (int): size of halo overlap. (default: 3)
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool) : add bias to q, k, and v projections
        avg_down (bool): use average pool downsample instead of strided query blocks
        scale_pos_embed (bool): scale the position embedding as well as Q @ K
    """

    def __init__(
        self,
        dim,
        dim_out=None,
        feat_size=None,
        stride=1,
        num_heads=8,
        dim_head=None,
        block_size=8,
        halo_size=3,
        qk_ratio=1.0,
        qkv_bias=False,
        avg_down=False,
        scale_pos_embed=False,
    ):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        assert stride in (1, 2)
        self.num_heads = num_heads
        self.dim_head_qk = (
            dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        )
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk**-0.5
        self.scale_pos_embed = scale_pos_embed
        self.block_size = self.block_size_ds = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2
        self.block_stride = 1
        use_avg_pool = False
        if stride > 1:
            use_avg_pool = avg_down or block_size % stride != 0
            self.block_stride = 1 if use_avg_pool else stride
            self.block_size_ds = self.block_size // self.block_stride
        self.q = nn.Conv2D(
            in_channels=dim,
            out_channels=self.dim_out_qk,
            kernel_size=1,
            stride=self.block_stride,
            bias_attr=qkv_bias,
        )
        self.kv = nn.Conv2D(
            in_channels=dim,
            out_channels=self.dim_out_qk + self.dim_out_v,
            kernel_size=1,
            bias_attr=qkv_bias,
        )
        self.pos_embed = PosEmbedRel(
            block_size=self.block_size_ds,
            win_size=self.win_size,
            dim_head=self.dim_head_qk,
            scale=self.scale,
        )
        self.pool = (
            nn.AvgPool2D(kernel_size=2, stride=2, exclusive=False)
            if use_avg_pool
            else nn.Identity()
        )
        self.reset_parameters()

    def reset_parameters(self):
        std = self.q.weight.shape[1] ** -0.5
        trunc_normal_(self.q.weight, std=std)
        trunc_normal_(self.kv.weight, std=std)
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H % self.block_size == 0, "")
        _assert(W % self.block_size == 0, "")
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks
        q = self.q(x)
        q = q.reshape(
            -1,
            self.dim_head_qk,
            num_h_blocks,
            self.block_size_ds,
            num_w_blocks,
            self.block_size_ds,
        ).transpose(perm=[0, 1, 3, 5, 2, 4])
        x = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks)
        perm_16 = list(range(x.ndim))
        perm_16[1] = 3
        perm_16[3] = 1
        q = x.transpose(perm=perm_16)
        kv = self.kv(x)
        kv = _FUNCTIONAL_PAD(
            pad=[self.halo_size, self.halo_size, self.halo_size, self.halo_size], x=kv
        )
        kv = (
            kv.unfold(axis=2, size=self.win_size, step=self.block_size)
            .unfold(axis=3, size=self.win_size, step=self.block_size)
            .reshape(
                B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1
            )
            .transpose(perm=[0, 2, 3, 1])
        )
        k, v = split(x=kv, num_or_sections=[self.dim_head_qk, self.dim_head_v], axis=-1)
        if self.scale_pos_embed:
            x = k
            perm_17 = list(range(x.ndim))
            perm_17[-1] = -2
            perm_17[-2] = -1
            attn = (q @ x.transpose(perm=perm_17) + self.pos_embed(q)) * self.scale
        else:
            x = k
            perm_18 = list(range(x.ndim))
            perm_18[-1] = -2
            perm_18[-2] = -1
            attn = q @ x.transpose(perm=perm_18) * self.scale + self.pos_embed(q)
        attn = F.softmax(attn, axis=-1)
        x = attn @ v
        perm_19 = list(range(x.ndim))
        perm_19[1] = 3
        perm_19[3] = 1
        out = x.transpose(perm=perm_19)
        out = out.reshape(
            -1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks
        )
        out = out.transpose(perm=[0, 3, 1, 4, 2]).view(
            B, self.dim_out_v, H // self.block_stride, W // self.block_stride
        )
        out = self.pool(out)
        return out


""" Three alternatives for overlapping windows.

`.unfold().unfold()` is same speed as stride tricks with similar clarity as F.unfold()

    if is_xla:
        # This code achieves haloing on PyTorch XLA with reasonable runtime trade-off, it is
        # EXTREMELY slow for backward on a GPU though so I need a way of selecting based on environment.
        WW = self.win_size ** 2
        pw = torch.eye(WW, dtype=x.dtype, device=x.device).reshape(WW, 1, self.win_size, self.win_size)
        kv = F.conv2d(kv.reshape(-1, 1, H, W), pw, stride=self.block_size, padding=self.halo_size)
    elif self.stride_tricks:
        kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size]).contiguous()
        kv = kv.as_strided((
            B, self.dim_out_qk + self.dim_out_v, self.win_size, self.win_size, num_h_blocks, num_w_blocks),
            stride=(kv.stride(0), kv.stride(1), kv.shape[-1], 1, self.block_size * kv.shape[-1], self.block_size))
    else:
        kv = F.unfold(kv, kernel_size=self.win_size, stride=self.block_size, padding=self.halo_size)

    kv = kv.reshape(
       B * self.num_heads, self.dim_head_qk + self.dim_head_v, -1, num_blocks).transpose(1, 3)
"""
