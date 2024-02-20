import paddle
import paddle.nn.functional as F
from paddle import nn as nn

class Linear(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Wraps paddle.nn.Linear to support AMP usage by manually casting
    weight & bias to input.dtype to work around an issue w/ paddle.matmul in this use case.
    """
    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        if self.bias is not None:
            bias = self.bias.astype(dtype=input.dtype)
        else:
            bias = None
        return F.linear(input, self.weight.astype(dtype=input.dtype), bias=bias)
