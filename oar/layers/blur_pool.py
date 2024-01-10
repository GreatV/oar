"""
BlurPool layer inspired by
 - Kornia's Max_BlurPool2d
 - Making Convolutional Networks Shift-Invariant Again :cite:`zhang2019shiftinvar`

Hacked together by Chris Ha and Ross Wightman
"""
import paddle
from paddle.nn import functional as F
from paddle import nn
import numpy as np
from .padding import get_padding


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


class BlurPool2d(nn.Layer):
    """Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling

    Args:
        channels = Number of input channels
        filt_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride

    Returns:
        torch.Tensor: the transformed tensor.
    """

    def __init__(self, channels, filt_size=3, stride=2) -> None:
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        self.padding = [get_padding(filt_size, stride, dilation=1)] * 4
        coeffs = paddle.to_tensor(
            data=(np.poly1d((0.5, 0.5)) ** (self.filt_size - 1)).coeffs.astype(
                np.float32
            )
        )
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :].repeat(
            self.channels, 1, 1, 1
        )
        self.register_buffer(name="filt", tensor=blur_filter, persistable=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = _FUNCTIONAL_PAD(pad=self.padding, mode="reflect", x=x)
        return F.conv2d(
            x=x, weight=self.filt, stride=self.stride, groups=self.channels
        )
