import paddle
from typing import Callable, Tuple, Type, Union

LayerType = Union[str, Callable, Type[paddle.nn.Layer]]
PadType = Union[str, int, Tuple[int, int]]
