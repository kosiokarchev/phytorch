from typing import Optional

from torch import Tensor
from torch.autograd import Function

from ..extensions import special as _special
from ..utils._typing import _TN
from ..utils.complex import with_complex_args
from ..utils.function_context import TorchFunctionContext


# noinspection PyAbstractClass,PyMethodOverriding
class Hyp2f1(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, a: _TN, b: _TN, c: _TN, z: _TN) -> Tensor:
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(a, b, c, z)
        return _special.hyp2f1(a, b, c, z)

    @staticmethod
    def backward(ctx: TorchFunctionContext, grad: Optional[Tensor]) -> tuple[None, None, None, Optional[Tensor]]:
        a, b, c, z = ctx.saved_tensors
        return (
            None, None, None,
            (a*b/c) * hyp2f1(a+1, b+1, c+1, z)
            if grad is not None and ctx.needs_input_grad[0] else None
        )


hyp2f1 = with_complex_args(Hyp2f1.apply)

__all__ = 'hyp2f1',
