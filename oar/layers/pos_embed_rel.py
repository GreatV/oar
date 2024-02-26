""" Relative position embedding modules and functions

Hacked together by / Copyright 2022 Ross Wightman
"""

import paddle
import math
import os
from typing import Optional, Tuple
from .interpolate import RegularGridInterpolator
from .mlp import Mlp
from .weight_init import trunc_normal_

_USE_SCIPY = int(os.environ.get("TIMM_USE_SCIPY_INTERP", 0)) > 0


def _FUNCTIONAL_PAD(x, pad, mode="constant", value=0.0, data_format="NCHW"):
    if len(x.shape) * 2 == len(pad) and mode == "constant":
        pad = (
            paddle.to_tensor(pad, dtype="int32")
            .reshape((-1, 2))
            .flip([0])
            .flatten()
            .tolist()
        )
    return paddle.nn.functional.pad(x, pad, mode, value, data_format)


def gen_relative_position_index(
    q_size: Tuple[int, int],
    k_size: Optional[Tuple[int, int]] = None,
    class_token: bool = False,
) -> paddle.Tensor:
    assert k_size is None, "Different q & k sizes not currently supported"
    coords = paddle.stack(
        x=paddle.meshgrid([paddle.arange(end=q_size[0]), paddle.arange(end=q_size[1])])
    ).flatten(start_axis=1)
    relative_coords = coords[:, :, None] - coords[:, None, :]
    relative_coords = relative_coords.transpose(perm=[1, 2, 0])
    relative_coords[:, :, 0] += q_size[0] - 1
    relative_coords[:, :, 1] += q_size[1] - 1
    relative_coords[:, :, 0] *= 2 * q_size[1] - 1
    num_relative_distance = (2 * q_size[0] - 1) * (2 * q_size[1] - 1)
    relative_position_index = relative_coords.sum(axis=-1)
    if class_token:
        relative_position_index = _FUNCTIONAL_PAD(
            pad=[1, 0, 1, 0], x=relative_position_index
        )
        relative_position_index[0, 0:] = num_relative_distance
        relative_position_index[0:, 0] = num_relative_distance + 1
        relative_position_index[0, 0] = num_relative_distance + 2
    return relative_position_index


def resize_rel_pos_bias_table_simple(
    rel_pos_bias, new_window_size: Tuple[int, int], new_bias_shape: Tuple[int, ...]
):
    dst_size = new_window_size[0] * 2 - 1, new_window_size[1] * 2 - 1
    if rel_pos_bias.ndim == 3:
        _, dst_h, dst_w = new_bias_shape
        num_attn_heads, src_h, src_w = rel_pos_bias.shape
        assert dst_h == dst_size[0] and dst_w == dst_size[1]
        if src_h != dst_h or src_w != dst_w:
            rel_pos_bias = paddle.nn.functional.interpolate(
                x=rel_pos_bias.unsqueeze(axis=0),
                size=dst_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
    else:
        assert rel_pos_bias.ndim == 2
        dst_num_pos, _ = new_bias_shape
        src_num_pos, num_attn_heads = rel_pos_bias.shape
        num_extra_tokens = dst_num_pos - dst_size[0] * dst_size[1]
        src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
        src_size = src_size, src_size
        if src_size[0] != dst_size[0] or src_size[1] != dst_size[1]:
            if num_extra_tokens:
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
            else:
                extra_tokens = None
            x = rel_pos_bias
            perm_25 = list(range(x.ndim))
            perm_25[1] = 0
            perm_25[0] = 1
            rel_pos_bias = (
                paddle.nn.functional.interpolate(
                    x=x.transpose(perm=perm_25).reshape(
                        (1, -1, src_size[0], src_size[1])
                    ),
                    size=dst_size,
                    mode="bicubic",
                    align_corners=False,
                )
                .view(-1, dst_num_pos - num_extra_tokens)
                .transpose(0, 1)
            )
            if extra_tokens is not None:
                rel_pos_bias = paddle.concat(x=(rel_pos_bias, extra_tokens), axis=0)
    return rel_pos_bias


def resize_rel_pos_bias_table_levit(
    position_bias_table,
    new_size,
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """
    Resample relative position bias table suggested in LeVit
    Adapted from: https://github.com/microsoft/Cream/blob/main/TinyViT/utils.py
    """
    L1, nH1 = position_bias_table.shape
    L2, nH2 = new_size
    assert nH1 == nH2
    if L1 != L2:
        orig_dtype = position_bias_table.dtype
        position_bias_table = position_bias_table.astype(dtype="float32")
        S1 = int(L1**0.5)
        S2 = int(L2**0.5)
        relative_position_bias_table_resized = paddle.nn.functional.interpolate(
            position_bias_table.transpose(perm=[1, 0]).view(1, nH1, S1, S1),
            size=(S2, S2),
            mode=interpolation,
            antialias=antialias,
        )
        relative_position_bias_table_resized = (
            relative_position_bias_table_resized.view(nH2, L2).transpose(perm=[1, 0])
        )
        relative_position_bias_table_resized.to(orig_dtype)
        return relative_position_bias_table_resized
    else:
        return position_bias_table


def resize_rel_pos_bias_table(
    rel_pos_bias, new_window_size: Tuple[int, int], new_bias_shape: Tuple[int, ...]
):
    """Resize relative position bias table using more advanced interpolation.

    Modified from code in Microsoft Unilm (https://github.com/microsoft/unilm) repo (BeiT, BeiT-v2, etc).

    https://github.com/microsoft/unilm/blob/5255d52de86dad642810f5849dd357769346c1d7/beit/run_class_finetuning.py#L351

    Args:
        rel_pos_bias:
        new_window_size:
        new_bias_shape:

    Returns:

    """
    if _USE_SCIPY:
        from scipy import interpolate
    dst_size = new_window_size[0] * 2 - 1, new_window_size[1] * 2 - 1
    if rel_pos_bias.ndim == 3:
        num_extra_tokens = 0
        _, dst_h, dst_w = new_bias_shape
        assert dst_h == dst_size[0] and dst_w == dst_size[1]
        num_attn_heads, src_h, src_w = rel_pos_bias.shape
        src_size = src_h, src_w
        has_flat_shape = False
    else:
        assert rel_pos_bias.ndim == 2
        dst_num_pos, _ = new_bias_shape
        src_num_pos, num_attn_heads = rel_pos_bias.shape
        num_extra_tokens = dst_num_pos - dst_size[0] * dst_size[1]
        src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
        src_size = src_size, src_size
        has_flat_shape = True
    if src_size[0] != dst_size[0] or src_size[1] != dst_size[1]:
        if num_extra_tokens:
            extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
            rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
        else:
            extra_tokens = None

        def geometric_progression(a, r, n):
            return a * (1.0 - r**n) / (1.0 - r)

        def _calc(src, dst):
            left, right = 1.01, 1.5
            while right - left > 1e-06:
                q = (left + right) / 2.0
                gp = geometric_progression(1, q, src // 2)
                if gp > dst // 2:
                    right = q
                else:
                    left = q
            dis = []
            cur = 1
            for i in range(src // 2):
                dis.append(cur)
                cur += q ** (i + 1)
            r_ids = [(-_) for _ in reversed(dis)]
            return r_ids + [0] + dis

        y = _calc(src_size[0], dst_size[0])
        x = _calc(src_size[1], dst_size[1])
        yx = [paddle.to_tensor(data=y), paddle.to_tensor(data=x)]
        ty = dst_size[0] // 2.0
        tx = dst_size[1] // 2.0
        dy = paddle.arange(start=-ty, end=ty + 0.1, step=1.0)
        dx = paddle.arange(start=-tx, end=tx + 0.1, step=1.0)
        dyx = paddle.meshgrid([dy, dx])
        all_rel_pos_bias = []
        for i in range(num_attn_heads):
            if has_flat_shape:
                z = (
                    rel_pos_bias[:, i]
                    .view(src_size[0], src_size[1])
                    .astype(dtype="float32")
                )
            else:
                z = rel_pos_bias[i, :, :].astype(dtype="float32")
            if _USE_SCIPY:
                f = interpolate.interp2d(x, y, z.numpy(), kind="cubic")
                r = paddle.to_tensor(data=f(dx, dy), dtype="float32").to(
                    rel_pos_bias.place
                )
            else:
                f = RegularGridInterpolator(yx, z)
                r = f(dyx).to(rel_pos_bias.place)
            if has_flat_shape:
                r = r.view(-1, 1)
            all_rel_pos_bias.append(r)
        if has_flat_shape:
            rel_pos_bias = paddle.concat(x=all_rel_pos_bias, axis=-1)
        else:
            rel_pos_bias = paddle.concat(x=all_rel_pos_bias, axis=0)
        if extra_tokens is not None:
            assert has_flat_shape
            rel_pos_bias = paddle.concat(x=(rel_pos_bias, extra_tokens), axis=0)
    return rel_pos_bias


class RelPosBias(paddle.nn.Layer):
    """Relative Position Bias
    Adapted from Swin-V1 relative position bias impl, modularized.
    """

    def __init__(self, window_size, num_heads, prefix_tokens=0):
        super().__init__()
        assert prefix_tokens <= 1
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.bias_shape = (self.window_area + prefix_tokens,) * 2 + (num_heads,)
        num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1
        ) + 3 * prefix_tokens
        out_45 = paddle.create_parameter(
            shape=paddle.zeros(shape=[num_relative_distance, num_heads]).shape,
            dtype=paddle.zeros(shape=[num_relative_distance, num_heads]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.zeros(shape=[num_relative_distance, num_heads])
            ),
        )
        out_45.stop_gradient = not True
        self.relative_position_bias_table = out_45
        self.register_buffer(
            name="relative_position_index",
            tensor=gen_relative_position_index(
                self.window_size, class_token=prefix_tokens > 0
            ).view(-1),
            persistable=False,
        )
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_bias(self) -> paddle.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index
        ]
        relative_position_bias = relative_position_bias.view(self.bias_shape).transpose(
            perm=[2, 0, 1]
        )
        return relative_position_bias.unsqueeze(axis=0)

    def forward(self, attn, shared_rel_pos: Optional[paddle.Tensor] = None):
        return attn + self.get_bias()


def gen_relative_log_coords(
    win_size: Tuple[int, int],
    pretrained_win_size: Tuple[int, int] = (0, 0),
    mode="swin",
):
    assert mode in ("swin", "cr")
    relative_coords_h = paddle.arange(
        start=-(win_size[0] - 1), end=win_size[0], dtype="float32"
    )
    relative_coords_w = paddle.arange(
        start=-(win_size[1] - 1), end=win_size[1], dtype="float32"
    )
    relative_coords_table = paddle.stack(
        x=paddle.meshgrid([relative_coords_h, relative_coords_w])
    )
    relative_coords_table = relative_coords_table.transpose(perm=[1, 2, 0])
    if mode == "swin":
        if pretrained_win_size[0] > 0:
            relative_coords_table[:, :, 0] /= pretrained_win_size[0] - 1
            relative_coords_table[:, :, 1] /= pretrained_win_size[1] - 1
        else:
            relative_coords_table[:, :, 0] /= win_size[0] - 1
            relative_coords_table[:, :, 1] /= win_size[1] - 1
        relative_coords_table *= 8
        relative_coords_table = (
            paddle.sign(x=relative_coords_table)
            * paddle.log2(x=1.0 + relative_coords_table.abs())
            / math.log2(8)
        )
    else:
        relative_coords_table = paddle.sign(x=relative_coords_table) * paddle.log(
            x=1.0 + relative_coords_table.abs()
        )
    return relative_coords_table


class RelPosMlp(paddle.nn.Layer):
    """Log-Coordinate Relative Position MLP
    Based on ideas presented in Swin-V2 paper (https://arxiv.org/abs/2111.09883)

    This impl covers the 'swin' implementation as well as two timm specific modes ('cr', and 'rw')
    """

    def __init__(
        self,
        window_size,
        num_heads=8,
        hidden_dim=128,
        prefix_tokens=0,
        mode="cr",
        pretrained_window_size=(0, 0),
    ):
        super().__init__()
        self.window_size = window_size
        self.window_area = self.window_size[0] * self.window_size[1]
        self.prefix_tokens = prefix_tokens
        self.num_heads = num_heads
        self.bias_shape = (self.window_area,) * 2 + (num_heads,)
        if mode == "swin":
            self.bias_act = paddle.nn.Sigmoid()
            self.bias_gain = 16
            mlp_bias = True, False
        else:
            self.bias_act = paddle.nn.Identity()
            self.bias_gain = None
            mlp_bias = True
        self.mlp = Mlp(
            2,
            hidden_features=hidden_dim,
            out_features=num_heads,
            act_layer=paddle.nn.ReLU,
            bias=mlp_bias,
            drop=(0.125, 0.0),
        )
        self.register_buffer(
            name="relative_position_index",
            tensor=gen_relative_position_index(window_size).view(-1),
            persistable=False,
        )
        self.register_buffer(
            name="rel_coords_log",
            tensor=gen_relative_log_coords(
                window_size, pretrained_window_size, mode=mode
            ),
            persistable=False,
        )

    def get_bias(self) -> paddle.Tensor:
        relative_position_bias = self.mlp(self.rel_coords_log)
        if self.relative_position_index is not None:
            relative_position_bias = relative_position_bias.view(-1, self.num_heads)[
                self.relative_position_index
            ]
            relative_position_bias = relative_position_bias.view(self.bias_shape)
        relative_position_bias = relative_position_bias.transpose(perm=[2, 0, 1])
        relative_position_bias = self.bias_act(relative_position_bias)
        if self.bias_gain is not None:
            relative_position_bias = self.bias_gain * relative_position_bias
        if self.prefix_tokens:
            relative_position_bias = _FUNCTIONAL_PAD(
                pad=[self.prefix_tokens, 0, self.prefix_tokens, 0],
                x=relative_position_bias,
            )
        return relative_position_bias.unsqueeze(axis=0)

    def forward(self, attn, shared_rel_pos: Optional[paddle.Tensor] = None):
        return attn + self.get_bias()


def generate_lookup_tensor(length: int, max_relative_position: Optional[int] = None):
    """Generate a one_hot lookup tensor to reindex embeddings along one dimension.

    Args:
        length: the length to reindex to.
        max_relative_position: the maximum relative position to consider.
            Relative position embeddings for distances above this threshold
            are zeroed out.
    Returns:
        a lookup Tensor of size [length, length, vocab_size] that satisfies
            ret[n,m,v] = 1{m - n + max_relative_position = v}.
    """
    if max_relative_position is None:
        max_relative_position = length - 1
    vocab_size = 2 * max_relative_position + 1
    ret = paddle.zeros(shape=[length, length, vocab_size])
    for i in range(length):
        for x in range(length):
            v = x - i + max_relative_position
            if abs(x - i) > max_relative_position:
                continue
            ret[i, x, v] = 1
    return ret


def reindex_2d_einsum_lookup(
    relative_position_tensor,
    height: int,
    width: int,
    height_lookup: paddle.Tensor,
    width_lookup: paddle.Tensor,
) -> paddle.Tensor:
    """Reindex 2d relative position bias with 2 independent einsum lookups.

    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py

    Args:
        relative_position_tensor: tensor of shape
            [..., vocab_height, vocab_width, ...].
        height: height to reindex to.
        width: width to reindex to.
        height_lookup: one-hot height lookup
        width_lookup: one-hot width lookup
    Returns:
        reindexed_tensor: a Tensor of shape
            [..., height * width, height * width, ...]
    """
    reindexed_tensor = paddle.einsum(
        "nhw,ixh->nixw", relative_position_tensor, height_lookup
    )
    reindexed_tensor = paddle.einsum("nixw,jyw->nijxy", reindexed_tensor, width_lookup)
    area = height * width
    return reindexed_tensor.reshape(relative_position_tensor.shape[0], area, area)


class RelPosBiasTf(paddle.nn.Layer):
    """Relative Position Bias Impl (Compatible with Tensorflow MaxViT models)
    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py
    """

    def __init__(self, window_size, num_heads, prefix_tokens=0):
        super().__init__()
        assert prefix_tokens <= 1
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.num_heads = num_heads
        vocab_height = 2 * window_size[0] - 1
        vocab_width = 2 * window_size[1] - 1
        self.bias_shape = self.num_heads, vocab_height, vocab_width
        out_46 = paddle.create_parameter(
            shape=paddle.zeros(shape=self.bias_shape).shape,
            dtype=paddle.zeros(shape=self.bias_shape).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.zeros(shape=self.bias_shape)
            ),
        )
        out_46.stop_gradient = not True
        self.relative_position_bias_table = out_46
        self.register_buffer(
            name="height_lookup",
            tensor=generate_lookup_tensor(window_size[0]),
            persistable=False,
        )
        self.register_buffer(
            name="width_lookup",
            tensor=generate_lookup_tensor(window_size[1]),
            persistable=False,
        )
        self.init_weights()

    def init_weights(self):
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.relative_position_bias_table)

    def get_bias(self) -> paddle.Tensor:
        return reindex_2d_einsum_lookup(
            self.relative_position_bias_table,
            self.window_size[0],
            self.window_size[1],
            self.height_lookup,
            self.width_lookup,
        )

    def forward(self, attn, shared_rel_pos: Optional[paddle.Tensor] = None):
        return attn + self.get_bias()
