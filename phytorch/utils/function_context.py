from torch import Tensor
from torch.autograd.function import _ContextMethodMixin


class TorchFunctionContext(_ContextMethodMixin):
    saved_tensors: tuple[Tensor, ...]
    needs_input_grad: tuple[bool, ...]
