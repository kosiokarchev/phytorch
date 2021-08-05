from typing import Optional

from more_itertools import circular_shifts
from torch import Tensor
from torch.autograd import Function

from ..extensions import elliptic as _elliptic
from ..utils._typing import _TN
from ..utils.complex import with_complex_args
from ..utils.function_context import ComplexTorchFunction, TorchFunctionContext


# noinspection PyMethodOverriding
class Elliprf(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, x, y, z):
        return x, y, z

    @staticmethod
    def _forward(ctx, x, y, z, *args):
        return _elliptic.elliprf(x, y, z)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        return tuple(
            (-elliprd(*args[::-1][:-1], args[0]) / 6.).conj() * grad if nig else None
            for nig, args in zip(ctx.needs_input_grad, circular_shifts(ctx.saved_tensors))
        ) if grad is not None else 3*(None,)


# noinspection PyMethodOverriding
class Elliprd(ComplexTorchFunction):
    @staticmethod
    def _forward(ctx, x, y, z, *args):
        return _elliptic.elliprd(x, y, z)


# noinspection PyMethodOverriding
class Elliprg(ComplexTorchFunction):
    @staticmethod
    def _forward(ctx, x, y, z):
        return _elliptic.elliprg(x, y, z)


# noinspection PyMethodOverriding
class Elliprj(ComplexTorchFunction):
    @staticmethod
    def _forward(ctx, x, y, z, p, *args):
        return _elliptic.elliprj(x, y, z, p)


elliprd = with_complex_args(Elliprd.apply)
elliprf = with_complex_args(Elliprf.apply)
elliprg = with_complex_args(Elliprg.apply)
elliprj = with_complex_args(Elliprj.apply)


def elliprc(x: _TN, y: _TN) -> Tensor:
    # TODO: Can we do better? There's a kernel, but then still we only get one
    #  automatic gradient. There's an "easy" expression, but it has cases.
    return elliprf(x, y, y)


__all__ = 'elliprc', 'elliprd', 'elliprf', 'elliprg', 'elliprj'

# TODO: unsafe flag for elliprj
