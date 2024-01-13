import paddle
from paddle import nn

try:
    from inplace_abn.functions import inplace_abn, inplace_abn_sync
    has_iabn = True
except ImportError:
    has_iabn = False

    def inplace_abn(
        x,
        weight,
        bias,
        running_mean,
        running_var,
        training=True,
        momentum=0.1,
        eps=1e-05,
        activation="leaky_relu",
        activation_param=0.01,
    ):
        raise ImportError(
            "Please install InplaceABN:'pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.12'"
        )

    def inplace_abn_sync(**kwargs):
        inplace_abn(**kwargs)


class InplaceAbn(nn.Layer):
    """Activated Batch Normalization

    This gathers a BatchNorm and an activation function in a single module

    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    eps : float
        Small constant to prevent numerical issues.
    momentum : float
        Momentum factor applied to compute running statistics.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    act_layer : str or nn.Module type
        Name or type of the activation functions, one of: `leaky_relu`, `elu`
    act_param : float
        Negative slope for the `leaky_relu` activation.
    """

    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        apply_act=True,
        act_layer="leaky_relu",
        act_param=0.01,
        drop_layer=None,
    ):
        super(InplaceAbn, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if apply_act:
            if isinstance(act_layer, str):
                assert act_layer in ("leaky_relu", "elu", "identity", "")
                self.act_name = act_layer if act_layer else "identity"
            elif act_layer == nn.ELU:
                self.act_name = "elu"
            elif act_layer == nn.LeakyReLU:
                self.act_name = "leaky_relu"
            elif act_layer is None or act_layer == nn.Identity:
                self.act_name = "identity"
            else:
                assert False, f"Invalid act layer {act_layer.__name__} for IABN"
        else:
            self.act_name = "identity"
        self.act_param = act_param
        if self.affine:
            out_53 = paddle.create_parameter(
                shape=paddle.ones(shape=num_features).shape,
                dtype=paddle.ones(shape=num_features).numpy().dtype,
                default_initializer=nn.initializer.Assign(
                    paddle.ones(shape=num_features)
                ),
            )
            out_53.stop_gradient = not True
            self.weight = out_53
            out_54 = paddle.create_parameter(
                shape=paddle.zeros(shape=num_features).shape,
                dtype=paddle.zeros(shape=num_features).numpy().dtype,
                default_initializer=nn.initializer.Assign(
                    paddle.zeros(shape=num_features)
                ),
            )
            out_54.stop_gradient = not True
            self.bias = out_54
        else:
            self.add_parameter(name="weight", parameter=None)
            self.add_parameter(name="bias", parameter=None)
        self.register_buffer(
            name="running_mean", tensor=paddle.zeros(shape=num_features)
        )
        self.register_buffer(name="running_var", tensor=paddle.ones(shape=num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init_Constant = nn.initializer.Constant(value=0)
        init_Constant(self.running_mean)
        init_Constant = nn.initializer.Constant(value=1)
        init_Constant(self.running_var)
        if self.affine:
            init_Constant = nn.initializer.Constant(value=1)
            init_Constant(self.weight)
            init_Constant = nn.initializer.Constant(value=0)
            init_Constant(self.bias)

    def forward(self, x):
        output = inplace_abn(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.training,
            self.momentum,
            self.eps,
            self.act_name,
            self.act_param,
        )
        if isinstance(output, tuple):
            output = output[0]
        return output
