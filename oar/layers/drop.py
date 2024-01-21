import paddle
from paddle import nn
from paddle.nn import functional as F

""" DropBlock, DropPath

PyTorch implementations of DropBlock and DropPath (Stochastic Depth) regularization layers.

Papers:
DropBlock: A regularization method for convolutional networks (https://arxiv.org/abs/1810.12890)

Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)

Code:
DropBlock impl inspired by two Tensorflow impl that I liked:
 - https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L74
 - https://github.com/clovaai/assembled-cnn/blob/master/nets/blocks.py

Hacked together by / Copyright 2020 Ross Wightman
"""


def drop_block_2d(
    x,
    drop_prob: float = 0.1,
    block_size: int = 7,
    gamma_scale: float = 1.0,
    with_noise: bool = False,
    inplace: bool = False,
    batchwise: bool = False,
):
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = (
        gamma_scale
        * drop_prob
        * total_size
        / clipped_block_size**2
        / ((W - block_size + 1) * (H - block_size + 1))
    )
    w_i, h_i = paddle.meshgrid(
        paddle.arange(end=W).to(x.place), paddle.arange(end=H).to(x.place)
    )
    valid_block = (
        (w_i >= clipped_block_size // 2)
        & (w_i < W - (clipped_block_size - 1) // 2)
        & ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    )
    valid_block = paddle.reshape(x=valid_block, shape=(1, 1, H, W)).to(dtype=x.dtype)
    if batchwise:
        uniform_noise = paddle.rand(shape=(1, C, H, W), dtype=x.dtype)
    else:
        uniform_noise = paddle.rand(shape=x.shape, dtype=x.dtype)
    block_mask = (2 - gamma - valid_block + uniform_noise >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        x=-block_mask,
        kernel_size=clipped_block_size,
        stride=1,
        padding=clipped_block_size // 2,
    )
    if with_noise:
        normal_noise = (
            paddle.randn(shape=(1, C, H, W), dtype=x.dtype)
            if batchwise
            else paddle.randn(shape=x.shape, dtype=x.dtype)
        )
        if inplace:
            x.multiply_(y=paddle.to_tensor(block_mask)).add_(
                y=paddle.to_tensor(normal_noise * (1 - block_mask))
            )
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (
            block_mask.size / block_mask.to(dtype="float32").sum().add(1e-07)
        ).to(x.dtype)
        if inplace:
            x.multiply_(y=paddle.to_tensor(block_mask * normalize_scale))
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(
    x: paddle.Tensor,
    drop_prob: float = 0.1,
    block_size: int = 7,
    gamma_scale: float = 1.0,
    with_noise: bool = False,
    inplace: bool = False,
):
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = (
        gamma_scale
        * drop_prob
        * total_size
        / clipped_block_size**2
        / ((W - block_size + 1) * (H - block_size + 1))
    )
    block_mask = paddle.empty_like(x=x).bernoulli_(gamma)
    block_mask = F.max_pool2d(
        x=block_mask.to(x.dtype),
        kernel_size=clipped_block_size,
        stride=1,
        padding=clipped_block_size // 2,
    )
    if with_noise:
        normal_noise = paddle.empty_like(x=x).normal_()
        if inplace:
            x.multiply_(y=paddle.to_tensor(1.0 - block_mask)).add_(
                y=paddle.to_tensor(normal_noise * block_mask)
            )
        else:
            x = x * (1.0 - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (
            block_mask.size / block_mask.to(dtype="float32").sum().add(1e-06)
        ).to(dtype=x.dtype)
        if inplace:
            x.multiply_(y=paddle.to_tensor(block_mask * normalize_scale))
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Layer):
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(
        self,
        drop_prob: float = 0.1,
        block_size: int = 7,
        gamma_scale: float = 1.0,
        with_noise: bool = False,
        inplace: bool = False,
        batchwise: bool = False,
        fast: bool = True,
    ):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(
                x,
                self.drop_prob,
                self.block_size,
                self.gamma_scale,
                self.with_noise,
                self.inplace,
            )
        else:
            return drop_block_2d(
                x,
                self.drop_prob,
                self.block_size,
                self.gamma_scale,
                self.with_noise,
                self.inplace,
                self.batchwise,
            )


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = paddle.empty(shape=shape, dtype=x.dtype).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.divide_(y=paddle.to_tensor(keep_prob))
    return x * random_tensor


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"
