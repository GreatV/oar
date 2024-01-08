import paddle
import paddle.nn as nn
from paddle.nn import initializer as init
from paddle.nn import functional as F
from typing import Optional
from .config import use_fused_attn
from .mlp import Mlp
from .weight_init import trunc_normal_tf_


class AttentionPoolLatent(nn.Layer):
    """Attention pooling w/ latent query"""

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        embed_dim: int = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        latent_len: int = 1,
        latent_dim: int = None,
        pos_embed: str = "",
        pool_type: str = "token",
        norm_layer: Optional[nn.Layer] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn()
        if pos_embed == "abs":
            spatial_len = self.feat_size
            out_36 = paddle.create_parameter(
                shape=paddle.zeros(shape=[spatial_len, in_features]).shape,
                dtype=paddle.zeros(shape=[spatial_len, in_features]).numpy().dtype,
                default_initializer=init.Assign(
                    paddle.zeros(shape=[spatial_len, in_features])
                ),
            )
            out_36.stop_gradient = not True
            self.pos_embed = out_36
        else:
            self.pos_embed = None
        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        out_37 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, self.latent_len, embed_dim]).shape,
            dtype=paddle.zeros(shape=[1, self.latent_len, embed_dim]).numpy().dtype,
            default_initializer=init.Assign(
                paddle.zeros(shape=[1, self.latent_len, embed_dim])
            ),
        )
        out_37.stop_gradient = not True
        self.latent = out_37
        self.q = nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias_attr=qkv_bias
        )
        self.kv = nn.Linear(
            in_features=embed_dim, out_features=embed_dim * 2, bias_attr=qkv_bias
        )
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.proj_drop = nn.Dropout(p=drop)
        self.norm = (
            norm_layer(out_features) if norm_layer is not None else nn.Identity()
        )
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio))
        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
        trunc_normal_tf_(self.latent, std=self.latent_dim**-0.5)

    def forward(self, x):
        B, N, C = x.shape
        if self.pos_embed is not None:
            x = x + self.pos_embed.unsqueeze(axis=0).to(x.dtype)
        q_latent = self.latent.expand(shape=[B, -1, -1])
        x = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim)
        perm_11 = list(range(x.ndim))
        perm_11[1] = 2
        perm_11[2] = 1
        q = x.transpose(perm=perm_11)
        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .transpose(perm=[2, 0, 3, 1, 4])
        )
        k, v = kv.unbind(axis=0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            x = k
            perm_12 = list(range(x.ndim))
            perm_12[-2] = -1
            perm_12[-1] = -2
            attn = q @ x.transpose(perm=perm_12)
            attn = F.softmax(attn, axis=-1)
            x = attn @ v
        x = x
        perm_13 = list(range(x.ndim))
        perm_13[1] = 2
        perm_13[2] = 1
        x = x.transpose(perm=perm_13).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + self.mlp(self.norm(x))
        if self.pool == "token":
            x = x[:, 0]
        elif self.pool == "avg":
            x = x.mean(axis=1)
        return x
