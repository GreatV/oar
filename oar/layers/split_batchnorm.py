""" Split BatchNorm

A PyTorch BatchNorm layer that splits input batch into N equal parts and passes each through
a separate BN layer. The first split is passed through the parent BN layers with weight/bias
keys the same as the original BN. All other splits pass through BN sub-layers under the '.aux_bn'
namespace.

This allows easily removing the auxiliary BN layers after training to efficiently
achieve the 'Auxiliary BatchNorm' as described in the AdvProp Paper, section 4.2,
'Disentangled Learning via An Auxiliary BN'

Hacked together by / Copyright 2020 Ross Wightman
"""

import paddle
import paddle.nn as nn


class SplitBatchNorm2D(nn.BatchNorm2D):
    def __init__(
        self,
        num_features,
        momentum=0.9,
        epsilon=1e-5,
        weight_attr=None,
        bias_attr=None,
        data_format="NCHW",
        num_splits=2,
    ):
        super().__init__(
            num_features, momentum, epsilon, weight_attr, bias_attr, data_format
        )
        assert (
            num_splits > 1
        ), "Should have at least one aux BN layer (num_splits at least 2)"
        self.num_splits = num_splits
        self.aux_bn = nn.LayerList(
            [
                nn.BatchNorm2D(
                    num_features, momentum, epsilon, weight_attr, bias_attr, data_format
                )
                for _ in range(num_splits - 1)
            ]
        )

    def forward(self, input):
        if self.training:  # aux BN only relevant while training
            split_size = input.shape[0] // self.num_splits
            assert (
                input.shape[0] == split_size * self.num_splits
            ), "batch size must be evenly divisible by num_splits"
            split_input = paddle.split(input, self.num_splits, 0)
            x = [super().forward(split_input[0])]
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i + 1]))
            return paddle.concat(x, axis=0)
        else:
            return super().forward(input)


def convert_splitbn_model(module, num_splits=2):
    """
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with `SplitBatchnorm2d`.
    Args:
        module (torch.nn.Module): input module
        num_splits: number of separate batchnorm layers to split input across
    Example::
        >>> # model is an instance of torch.nn.Module
        >>> model = oar.models.convert_splitbn_model(model, num_splits=2)
    """
    mod = module
    if isinstance(module, nn.InstanceNorm2D):
        return module
    if isinstance(module, nn.BatchNorm2D):
        mod = SplitBatchNorm2D(
            module._num_channels,
            module._momentum,
            module._epsilon,
            module._weight_attr,
            module._bias_attr,
            module._data_format,
            num_splits=num_splits,
        )
        mod._mean = module._mean
        mod._variance = module._variance
        if module._weight_attr is not None:
            mod.weight.set_value(module.weight.numpy())
            mod.bias.set_value(module.bias.numpy())
        for aux in mod.aux_bn:
            aux._mean = module._mean.clone()
            aux._variance = module._variance.clone()
            if module._weight_attr is not None:
                aux.weight.set_value(module.weight.numpy())
                aux.bias.set_value(module.bias.numpy())
    for name, child in module.named_sublayers():
        mod.add_sublayer(name, convert_splitbn_model(child, num_splits=num_splits))
    del module
    return mod
