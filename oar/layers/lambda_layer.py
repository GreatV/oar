""" Lambda Layer

Paper: `LambdaNetworks: Modeling Long-Range Interactions Without Attention`
    - https://arxiv.org/abs/2102.08602

@misc{2102.08602,
Author = {Irwan Bello},
Title = {LambdaNetworks: Modeling Long-Range Interactions Without Attention},
Year = {2021},
}

Status:
This impl is a WIP. Code snippets in the paper were used as reference but
good chance some details are missing/wrong.

I've only implemented local lambda conv based pos embeddings.

For a PyTorch impl that includes other embedding options checkout
https://github.com/lucidrains/lambda-networks

Hacked together by / Copyright 2021 Ross Wightman
"""
import paddle
from paddle import nn
from paddle.nn import functional as F
from .helpers import to_2tuple, make_divisible
from .weight_init import trunc_normal_


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis] // num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


def rel_pos_indices(size):
    size = to_2tuple(size)
    pos = paddle.stack(
        x=paddle.meshgrid(paddle.arange(end=size[0]), paddle.arange(end=size[1]))
    ).flatten(start_axis=1)
    rel_pos = pos[:, None, :] - pos[:, :, None]
    rel_pos[0] += size[0] - 1
    rel_pos[1] += size[1] - 1
    return rel_pos


class LambdaLayer(nn.Layer):
    """Lambda Layer

    Paper: `LambdaNetworks: Modeling Long-Range Interactions Without Attention`
        - https://arxiv.org/abs/2102.08602

    NOTE: intra-depth parameter 'u' is fixed at 1. It did not appear worth the complexity to add.

    The internal dimensions of the lambda module are controlled via the interaction of several arguments.
      * the output dimension of the module is specified by dim_out, which falls back to input dim if not set
      * the value (v) dimension is set to dim_out // num_heads, the v projection determines the output dim
      * the query (q) and key (k) dimension are determined by
        * dim_head = (dim_out * attn_ratio // num_heads) if dim_head is None
        * q = num_heads * dim_head, k = dim_head
      * as seen above, attn_ratio determines the ratio of q and k relative to the output if dim_head not set

    Args:
        dim (int): input dimension to the module
        dim_out (int): output dimension of the module, same as dim if not set
        feat_size (Tuple[int, int]): size of input feature_map for relative pos variant H, W
        stride (int): output stride of the module, avg pool used if stride == 2
        num_heads (int): parallel attention heads.
        dim_head (int): dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        r (int): local lambda convolution radius. Use lambda conv if set, else relative pos if not. (default: 9)
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool): add bias to q, k, and v projections
    """

    def __init__(
        self,
        dim,
        dim_out=None,
        feat_size=None,
        stride=1,
        num_heads=4,
        dim_head=16,
        r=9,
        qk_ratio=1.0,
        qkv_bias=False,
    ):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0, " should be divided by num_heads"
        self.dim_qk = (
            dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        )
        self.num_heads = num_heads
        self.dim_v = dim_out // num_heads
        self.qkv = nn.Conv2D(
            in_channels=dim,
            out_channels=num_heads * self.dim_qk + self.dim_qk + self.dim_v,
            kernel_size=1,
            bias_attr=qkv_bias,
        )
        self.norm_q = nn.BatchNorm2D(num_features=num_heads * self.dim_qk)
        self.norm_v = nn.BatchNorm2D(num_features=self.dim_v)
        if r is not None:
            self.conv_lambda = nn.Conv3D(
                in_channels=1,
                out_channels=self.dim_qk,
                kernel_size=(r, r, 1),
                padding=(r // 2, r // 2, 0),
            )
            self.pos_emb = None
            self.rel_pos_indices = None
        else:
            assert feat_size is not None
            feat_size = to_2tuple(feat_size)
            rel_size = [(2 * s - 1) for s in feat_size]
            self.conv_lambda = None
            out_47 = paddle.create_parameter(
                shape=paddle.zeros(shape=[rel_size[0], rel_size[1], self.dim_qk]).shape,
                dtype=paddle.zeros(shape=[rel_size[0], rel_size[1], self.dim_qk])
                .numpy()
                .dtype,
                default_initializer=nn.initializer.Assign(
                    paddle.zeros(shape=[rel_size[0], rel_size[1], self.dim_qk])
                ),
            )
            out_47.stop_gradient = not True
            self.pos_emb = out_47
            self.register_buffer(
                name="rel_pos_indices",
                tensor=rel_pos_indices(feat_size),
                persistable=False,
            )
        self.pool = (
            nn.AvgPool2D(kernel_size=2, stride=2, exclusive=False)
            if stride == 2
            else nn.Identity()
        )
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.qkv.weight, std=self.qkv.weight.shape[1] ** -0.5)
        if self.conv_lambda is not None:
            trunc_normal_(self.conv_lambda.weight, std=self.dim_qk**-0.5)
        if self.pos_emb is not None:
            trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        M = H * W
        qkv = self.qkv(x)
        q, k, v = split(
            x=qkv,
            num_or_sections=[self.num_heads * self.dim_qk, self.dim_qk, self.dim_v],
            axis=1,
        )
        x = self.norm_q(q).reshape(B, self.num_heads, self.dim_qk, M)
        perm_27 = list(range(x.ndim))
        perm_27[-1] = -2
        perm_27[-2] = -1
        q = x.transpose(perm=perm_27)
        x = self.norm_v(v).reshape(B, self.dim_v, M)
        perm_28 = list(range(x.ndim))
        perm_28[-1] = -2
        perm_28[-2] = -1
        v = x.transpose(perm=perm_28)
        k = F.softmax(x=k.reshape(B, self.dim_qk, M), axis=-1)
        content_lam = k @ v
        content_out = q @ content_lam.unsqueeze(axis=1)
        if self.pos_emb is None:
            position_lam = self.conv_lambda(v.reshape(B, 1, H, W, self.dim_v))
            x = position_lam.reshape(B, 1, self.dim_qk, H * W, self.dim_v)
            perm_29 = list(range(x.ndim))
            perm_29[2] = 3
            perm_29[3] = 2
            position_lam = x.transpose(perm=perm_29)
        else:
            pos_emb = self.pos_emb[
                self.rel_pos_indices[0], self.rel_pos_indices[1]
            ].expand(shape=[B, -1, -1, -1])
            x = pos_emb
            perm_30 = list(range(x.ndim))
            perm_30[-1] = -2
            perm_30[-2] = -1
            position_lam = (x.transpose(perm=perm_30) @ v.unsqueeze(axis=1)).unsqueeze(
                axis=1
            )
        position_out = (q.unsqueeze(axis=-2) @ position_lam).squeeze(axis=-2)
        x = content_out + position_out
        perm_31 = list(range(x.ndim))
        perm_31[-1] = -2
        perm_31[-2] = -1
        out = x.transpose(perm=perm_31).reshape(B, C, H, W)
        out = self.pool(out)
        return out
