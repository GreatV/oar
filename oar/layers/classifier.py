""" Classifier head and layer factory

Hacked together by / Copyright 2020 Ross Wightman
"""
import paddle
from paddle import nn
from collections import OrderedDict
from functools import partial
from typing import Optional, Union, Callable
from .adaptive_avgmax_pool import SelectAdaptivePool2d
from .create_act import get_act_layer
from .create_norm import get_norm_layer


def _create_pool(
    num_features: int,
    num_classes: int,
    pool_type: str = "avg",
    use_conv: bool = False,
    input_fmt: Optional[str] = None,
):
    flatten_in_pool = not use_conv
    if not pool_type:
        assert (
            num_classes == 0 or use_conv
        ), "Pooling can only be disabled if classifier is also removed or conv classifier is used"
        flatten_in_pool = False
    global_pool = SelectAdaptivePool2d(
        pool_type=pool_type, flatten=flatten_in_pool, input_fmt=input_fmt
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()
    elif use_conv:
        fc = nn.Conv2D(
            in_channels=num_features,
            out_channels=num_classes,
            kernel_size=1,
            bias_attr=True,
        )
    else:
        fc = nn.Linear(
            in_features=num_features, out_features=num_classes, bias_attr=True
        )
    return fc


def create_classifier(
    num_features: int,
    num_classes: int,
    pool_type: str = "avg",
    use_conv: bool = False,
    input_fmt: str = "NCHW",
    drop_rate: Optional[float] = None,
):
    global_pool, num_pooled_features = _create_pool(
        num_features, num_classes, pool_type, use_conv=use_conv, input_fmt=input_fmt
    )
    fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
    if drop_rate is not None:
        dropout = nn.Dropout(p=drop_rate)
        return global_pool, dropout, fc
    return global_pool, fc


class ClassifierHead(nn.Layer):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pool_type: str = "avg",
        drop_rate: float = 0.0,
        use_conv: bool = False,
        input_fmt: str = "NCHW",
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
        """
        super(ClassifierHead, self).__init__()
        self.in_features = in_features
        self.use_conv = use_conv
        self.input_fmt = input_fmt
        global_pool, fc = create_classifier(
            in_features, num_classes, pool_type, use_conv=use_conv, input_fmt=input_fmt
        )
        self.global_pool = global_pool
        self.drop = nn.Dropout(p=drop_rate)
        self.fc = fc
        self.flatten = (
            nn.Flatten(start_axis=1) if use_conv and pool_type else nn.Identity()
        )

    def reset(self, num_classes, pool_type=None):
        if pool_type is not None and pool_type != self.global_pool.pool_type:
            self.global_pool, self.fc = create_classifier(
                self.in_features,
                num_classes,
                pool_type=pool_type,
                use_conv=self.use_conv,
                input_fmt=self.input_fmt,
            )
            self.flatten = (
                nn.Flatten(start_axis=1)
                if self.use_conv and pool_type
                else nn.Identity()
            )
        else:
            num_pooled_features = self.in_features * self.global_pool.feat_mult()
            self.fc = _create_fc(
                num_pooled_features, num_classes, use_conv=self.use_conv
            )

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.drop(x)
        if pre_logits:
            return self.flatten(x)
        x = self.fc(x)
        return self.flatten(x)


class NormMlpClassifierHead(nn.Layer):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_size: Optional[int] = None,
        pool_type: str = "avg",
        drop_rate: float = 0.0,
        norm_layer: Union[str, Callable] = "layernorm2d",
        act_layer: Union[str, Callable] = "tanh",
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            hidden_size: The hidden size of the MLP (pre-logits FC layer) if not None.
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
            norm_layer: Normalization layer type.
            act_layer: MLP activation layer type (only used if hidden_size is not None).
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_features = in_features
        self.use_conv = not pool_type
        norm_layer = get_norm_layer(norm_layer)
        act_layer = get_act_layer(act_layer)
        linear_layer = partial(nn.Conv2D, kernel_size=1) if self.use_conv else nn.Linear
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.norm = norm_layer(in_features)
        self.flatten = nn.Flatten(start_axis=1) if pool_type else nn.Identity()
        if hidden_size:
            self.pre_logits = nn.Sequential(
                *[("fc", linear_layer(in_features, hidden_size)), ("act", act_layer())]
            )
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.Identity()
        self.drop = nn.Dropout(p=drop_rate)
        self.fc = (
            linear_layer(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def reset(self, num_classes, global_pool=None):
        if global_pool is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.flatten = nn.Flatten(start_axis=1) if global_pool else nn.Identity()
        self.use_conv = self.global_pool.is_identity()
        linear_layer = partial(nn.Conv2D, kernel_size=1) if self.use_conv else nn.Linear
        if self.hidden_size:
            if (
                isinstance(self.pre_logits.fc, nn.Conv2D)
                and not self.use_conv
                or isinstance(self.pre_logits.fc, nn.Linear)
                and self.use_conv
            ):
                with paddle.no_grad():
                    new_fc = linear_layer(self.in_features, self.hidden_size)
                    new_fc.weight.copy_(
                        self.pre_logits.fc.weight.reshape(new_fc.weight.shape)
                    )
                    new_fc.bias.copy_(self.pre_logits.fc.bias)
                    self.pre_logits.fc = new_fc
        self.fc = (
            linear_layer(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        x = self.drop(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x
