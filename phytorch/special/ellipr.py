from typing import Optional

from more_itertools import circular_shifts
from torch import Tensor
from torch.autograd import Function

from ..extensions import ellipr as _ellipr
from ..utils._typing import _TN
from ..utils.complex import with_complex_args
from ..utils.function_context import TorchFunctionContext


# noinspection PyMethodOverriding
class Elliprf(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, x: _TN, y: _TN, z: Tensor) -> Tensor:
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
    def forward(ctx: TorchFunctionContext, x: _TN, y: _TN, z: _TN, p: Tensor) -> Tensor:
        ctx.mark_non_differentiable(
            ret := _ellipr.elliprj(x, y, z, p))
        return ret


# noinspection PyMethodOverriding,PyAbstractClass
class Elliprd(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, x: _TN, y: _TN, z: _TN) -> Tensor:
        ctx.mark_non_differentiable(
            ret := _ellipr.elliprd(x, y, z))
        return ret


# noinspection PyMethodOverriding,PyAbstractClass
class Elliprg(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, x: _TN, y: _TN, z: _TN) -> Tensor:
        ctx.mark_non_differentiable(
            ret := _ellipr.elliprg(x, y, z))
        return ret


funcs = {E.__name__.lower(): E for E in (Elliprd, Elliprf, Elliprg, Elliprj)}
globals().update({key: with_complex_args(val.apply) for key, val in funcs.items()})


def elliprc(x: _TN, y: _TN) -> Tensor:
    # TODO: Can we do better? There's a kernel, but then still we only get one
    #  automatic gradient. There's an "easy" expression, but it has cases.
    return elliprf(x, y, y)


__all__ = tuple(funcs.keys()) + ('elliprc',)

# TODO: unsafe flag for elliprj
