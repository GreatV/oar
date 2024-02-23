import paddle
from typing import Optional, Tuple, Union


class PatchDropout(paddle.nn.Layer):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(
        self,
        prob: float = 0.5,
        num_prefix_tokens: int = 1,
        ordered: bool = False,
        return_indices: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens
        self.ordered = ordered
        self.return_indices = return_indices

    def forward(
        self, x
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, Optional[paddle.Tensor]]]:
        if not self.training or self.prob == 0.0:
            if self.return_indices:
                return x, None
            return x
        if self.num_prefix_tokens:
            prefix_tokens, x = (
                x[:, : self.num_prefix_tokens],
                x[:, self.num_prefix_tokens :],
            )
        else:
            prefix_tokens = None
        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1.0 - self.prob)))
        keep_indices = paddle.argsort(x=paddle.randn(shape=[B, L]), axis=-1)[
            :, :num_keep
        ]
        if self.ordered:
            keep_indices = (
                paddle.sort(x=keep_indices, axis=-1),
                paddle.argsort(x=keep_indices, axis=-1),
            )[0]
        x = x.take_along_axis(
            axis=1,
            indices=keep_indices.unsqueeze(axis=-1).expand(
                shape=(-1, -1) + x.shape[2:]
            ),
        )
        if prefix_tokens is not None:
            x = paddle.concat(x=(prefix_tokens, x), axis=1)
        if self.return_indices:
            return x, keep_indices
        return x
