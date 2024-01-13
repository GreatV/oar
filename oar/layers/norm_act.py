""" Normalization + Activation Layers

Provides Norm+Act fns for standard PyTorch norm layers such as
* BatchNorm
* GroupNorm
* LayerNorm

This allows swapping with alternative layers that are natively both norm + act such as
* EvoNorm (evo_norm.py)
* FilterResponseNorm (filter_response_norm.py)
* InplaceABN (inplace_abn.py)

Hacked together by / Copyright 2022 Ross Wightman
"""
import paddle
from paddle import nn
from paddle.nn import functional as F
from typing import Union, List, Optional, Any
from .create_act import get_act_layer
from .fast_norm import is_fast_norm, fast_group_norm, fast_layer_norm
from .trace_utils import _assert


def _create_act(act_layer, act_kwargs=None, inplace=False, apply_act=True):
    act_layer = get_act_layer(act_layer)
    act_kwargs = act_kwargs or {}
    if act_layer is not None and apply_act:
        if inplace:
            act_kwargs["inplace"] = inplace
        act = act_layer(**act_kwargs)
    else:
        act = nn.Identity()
    return act


class BatchNormAct2d(nn.BatchNorm2D):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """

    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        apply_act=True,
        act_layer=nn.ReLU,
        act_kwargs=None,
        inplace=True,
        drop_layer=None,
        device=None,
        dtype=None,
    ):
        try:
            factory_kwargs = {"device": device, "dtype": dtype}
            super(BatchNormAct2d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
                **factory_kwargs,
            )
        except TypeError:
            super(BatchNormAct2d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )

    def forward(self, x):
        _assert(x.ndim == 4, f"expected 4D input (got {x.ndim}D input)")
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(y=paddle.to_tensor(1))
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        """
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = self.running_mean is None and self.running_var is None
        """
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            x=x,
            running_mean=self.running_mean
            if not self.training or self.track_running_stats
            else None,
            running_var=self.running_var
            if not self.training or self.track_running_stats
            else None,
            weight=self.weight,
            bias=self.bias,
            training=bn_training,
            epsilon=self.eps,
            momentum=1 - exponential_average_factor,
        )
        x = self.drop(x)
        x = self.act(x)
        return x


class SyncBatchNormAct(nn.SyncBatchNorm):
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = super().forward(x)
        if hasattr(self, "drop"):
            x = self.drop(x)
        if hasattr(self, "act"):
            x = self.act(x)
        return x


def convert_sync_batchnorm(module, process_group=None):
    module_output = module
    if isinstance(module, nn.layer.norm._BatchNormBase):
        if isinstance(module, BatchNormAct2d):
            module_output = SyncBatchNormAct(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group=process_group,
            )
            module_output.act = module.act
            module_output.drop = module.drop
        else:
            module_output = nn.SyncBatchNorm(
                num_features=module.num_features,
                epsilon=module.eps,
                weight_attr=module.affine,
                bias_attr=module.affine,
                momentum=1 - module.momentum,
            )
        if module.affine:
            with paddle.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_sublayer(
            name=name, sublayer=convert_sync_batchnorm(child, process_group)
        )
    del module
    return module_output


class FrozenBatchNormAct2d(nn.Layer):
    """
    BatchNormAct2d where the batch statistics and the affine parameters are fixed

    Args:
        num_features (int): Number of features ``C`` from an expected input of size ``(N, C, H, W)``
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        apply_act=True,
        act_layer=nn.ReLU,
        act_kwargs=None,
        inplace=True,
        drop_layer=None,
    ):
        super().__init__()
        self.eps = eps
        self.register_buffer(name="weight", tensor=paddle.ones(shape=num_features))
        self.register_buffer(name="bias", tensor=paddle.zeros(shape=num_features))
        self.register_buffer(
            name="running_mean", tensor=paddle.zeros(shape=num_features)
        )
        self.register_buffer(name="running_var", tensor=paddle.ones(shape=num_features))
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        x = x * scale + bias
        x = self.act(self.drop(x))
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps}, act={self.act})"


def freeze_batch_norm_2d(module):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` or `BatchNormAct2d` and `SyncBatchNormAct2d` layers
    of provided module into `FrozenBatchNorm2d` or `FrozenBatchNormAct2d` respectively.

    Args:
        module (torch.nn.Module): Any PyTorch module.

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    if isinstance(module, (BatchNormAct2d, SyncBatchNormAct)):
        res = FrozenBatchNormAct2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
        res.drop = module.drop
        res.act = module.act
    elif isinstance(module, (nn.BatchNorm2D, nn.SyncBatchNorm)):
        res = paddle.vision.ops.misc.FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for name, child in module.named_children():
            new_child = freeze_batch_norm_2d(child)
            if new_child is not child:
                res.add_sublayer(name=name, sublayer=new_child)
    return res


def unfreeze_batch_norm_2d(module):
    """
    Converts all `FrozenBatchNorm2d` layers of provided module into `BatchNorm2d`. If `module` is itself and instance
    of `FrozenBatchNorm2d`, it is converted into `BatchNorm2d` and returned. Otherwise, the module is walked
    recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    if isinstance(module, FrozenBatchNormAct2d):
        res = BatchNormAct2d(module.num_features)
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
        res.drop = module.drop
        res.act = module.act
    elif isinstance(module, paddle.vision.ops.misc.FrozenBatchNorm2d):
        res = nn.BatchNorm2D(num_features=module.num_features)
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for name, child in module.named_children():
            new_child = unfreeze_batch_norm_2d(child)
            if new_child is not child:
                res.add_sublayer(name=name, sublayer=new_child)
    return res


def _num_groups(num_channels, num_groups, group_size):
    if group_size:
        assert num_channels % group_size == 0
        return num_channels // group_size
    return num_groups


class GroupNormAct(nn.GroupNorm):
    def __init__(
        self,
        num_channels,
        num_groups=32,
        eps=1e-05,
        affine=True,
        group_size=None,
        apply_act=True,
        act_layer=nn.ReLU,
        act_kwargs=None,
        inplace=True,
        drop_layer=None,
    ):
        super(GroupNormAct, self).__init__(
            _num_groups(num_channels, num_groups, group_size),
            num_channels,
            eps=eps,
            affine=affine,
        )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x


class GroupNorm1Act(nn.GroupNorm):
    def __init__(
        self,
        num_channels,
        eps=1e-05,
        affine=True,
        apply_act=True,
        act_layer=nn.ReLU,
        act_kwargs=None,
        inplace=True,
        drop_layer=None,
    ):
        super(GroupNorm1Act, self).__init__(1, num_channels, eps=eps, affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x


class LayerNormAct(nn.LayerNorm):
    def __init__(
        self,
        normalization_shape: Union[int, List[int], list],
        eps=1e-05,
        affine=True,
        apply_act=True,
        act_layer=nn.ReLU,
        act_kwargs=None,
        inplace=True,
        drop_layer=None,
    ):
        super(LayerNormAct, self).__init__(
            normalization_shape, eps=eps, elementwise_affine=affine
        )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        else:
            x = F.layer_norm(
                x=x,
                normalized_shape=self.normalized_shape,
                weight=self.weight,
                bias=self.bias,
                epsilon=self.eps,
            )
        x = self.drop(x)
        x = self.act(x)
        return x


class LayerNormAct2d(nn.LayerNorm):
    def __init__(
        self,
        num_channels,
        eps=1e-05,
        affine=True,
        apply_act=True,
        act_layer=nn.ReLU,
        act_kwargs=None,
        inplace=True,
        drop_layer=None,
    ):
        super(LayerNormAct2d, self).__init__(
            num_channels, eps=eps, elementwise_affine=affine
        )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        x = x.transpose(perm=[0, 2, 3, 1])
        if self._fast_norm:
            x = fast_layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        else:
            x = F.layer_norm(
                x=x,
                normalized_shape=self.normalized_shape,
                weight=self.weight,
                bias=self.bias,
                epsilon=self.eps,
            )
        x = x.transpose(perm=[0, 3, 1, 2])
        x = self.drop(x)
        x = self.act(x)
        return x
