from typing import Optional

from more_itertools import circular_shifts
from torch import Tensor
from torch.autograd import Function

from ..extensions import ellipr as _ellipr
from ..utils.function_context import TorchFunctionContext


# noinspection PyMethodOverriding
class Elliprf(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, y, z)
        return _ellipr.elliprf(x, y, z)

    @staticmethod
    def backward(ctx: TorchFunctionContext, grad: Tensor) -> tuple[Optional[Tensor], ...]:
        return tuple(
            (-elliprd(*args[::-1][:-1], args[0]) / 6.).conj() * grad if nig else None
            for nig, args in zip(ctx.needs_input_grad, circular_shifts(ctx.saved_tensors))
        ) if grad is not None else 3*(None,)


# noinspection PyMethodOverriding,PyAbstractClass
class Elliprj(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, x: Tensor, y: Tensor, z: Tensor, p: Tensor) -> Tensor:
        ret = _ellipr.elliprj(x, y, z, p)
        ctx.mark_non_differentiable(ret)
        return ret


# noinspection PyMethodOverriding,PyAbstractClass
class Elliprd(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        ret = _ellipr.elliprd(x, y, z)
        ctx.mark_non_differentiable(ret)
        return ret


# noinspection PyMethodOverriding,PyAbstractClass
class Elliprg(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        ret = _ellipr.elliprg(x, y, z)
        ctx.mark_non_differentiable(ret)
        return ret


elliprf = Elliprf.apply
elliprj = Elliprj.apply
elliprd = Elliprd.apply
elliprg = Elliprg.apply


def elliprc(x: Tensor, y: Tensor) -> Tensor:
    # TODO: Can we do better? There's a kernel, but then still we only get one
    #  automatic gradient. There's an "easy" expression, but it has cases.
    return elliprf(x, y, y)


__all__ = 'elliprf', 'elliprc', 'elliprj', 'elliprd', 'elliprg'
