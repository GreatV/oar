import paddle.nn as nn


class SpaceToDepth(nn.Layer):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape([N, C, H // self.bs, self.bs, W // self.bs, self.bs])
        x = x.transpose([0, 3, 5, 1, 2, 4])
        x = x.reshape([N, C * self.bs * self.bs, H // self.bs, W // self.bs])
        return x


class DepthToSpace(nn.Layer):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape([N, self.bs, self.bs, C // (self.bs**2), H, W])
        x = x.transpose([0, 3, 4, 1, 5, 2])
        x = x.reshape([N, C // (self.bs**2), H * self.bs, W * self.bs])
        return x
