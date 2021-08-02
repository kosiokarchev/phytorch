from typing import Optional

from torch import Tensor
from torch.autograd import Function

from ..extensions import special as _special
from ..utils._typing import _TN
from ..utils.complex import as_complex_tensors, with_complex_args
from ..utils.function_context import TorchFunctionContext


# noinspection PyMethodOverriding,PyAbstractClass
class Gamma(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, z: _TN) -> Tensor:
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(z, (res := _special.gamma(z)))
        return res

    @staticmethod
    def backward(ctx: TorchFunctionContext, grad: Optional[Tensor]) -> Optional[Tensor]:
        z, G = ctx.saved_tensors
        return (G * digamma(z)).conj() * grad if grad is not None and ctx.needs_input_grad[0] else None


# noinspection PyMethodOverriding,PyAbstractClass
class Loggamma(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, z: _TN) -> Tensor:
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(z)
        return _special.loggamma(z)

    @staticmethod
    def backward(ctx: TorchFunctionContext, grad: Optional[Tensor]) -> Optional[Tensor]:
        return digamma(ctx.saved_tensors[0]).conj() * grad if grad is not None and ctx.needs_input_grad[0] else None


# noinspection PyMethodOverriding,PyAbstractClass
class Digamma(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, z: _TN) -> Tensor:
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(z)
        return _special.digamma(z)

    @staticmethod
    def backward(ctx: TorchFunctionContext, grad: Optional[Tensor]) -> Optional[Tensor]:
        return polygamma(1, ctx.saved_tensors[0]).conj() * grad if grad is not None and ctx.needs_input_grad[0] else None


# noinspection PyMethodOverriding,PyAbstractClass
class Polygamma(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, n: int, z: _TN) -> Tensor:
        ctx.set_materialize_grads(False)
        ctx.n = n
        ctx.save_for_backward(z)
        return _special.polygamma(n, z)
        # return (-1)**(n+1) * gamma(n+1) * zeta(n+1, z)

    @staticmethod
    def backward(ctx: TorchFunctionContext, grad: Optional[Tensor]) -> tuple[None, Optional[Tensor]]:
        return (
            None,
            polygamma(ctx.n+1, ctx.saved_tensors[0]).conj() * grad
            if grad is not None and ctx.needs_input_grad[0] else None
        )


gamma = with_complex_args(Gamma.apply)
loggamma = with_complex_args(Loggamma.apply)
digamma = with_complex_args(Digamma.apply)


def polygamma(n: int, z: _TN) -> Tensor:
    return Polygamma.apply(n, *as_complex_tensors(z))


__all__ = 'gamma', 'loggamma', 'digamma', 'polygamma'

