from torch import Tensor

from ..extensions import special as _special
from ..utils._typing import _TN
from ..utils.complex import as_complex_tensors, with_complex_args
from ..utils.function_context import CargsMixin


# noinspection PyMethodOverriding,PyUnusedLocal
class Gamma(CargsMixin):
    @staticmethod
    def saved_tensors(ctx, z):
        return z,

    @staticmethod
    def _forward(ctx, z, *args):
        return _special.gamma(z)

    @staticmethod
    def grad_z(ctx, z, G):
        return G * digamma(z)

    gradfuncs = grad_z,


# noinspection PyMethodOverriding,PyUnusedLocal
class Loggamma(CargsMixin):
    save_output = False

    @staticmethod
    def saved_tensors(ctx, z):
        return z,

    @staticmethod
    def _forward(ctx, z, *args):
        return _special.loggamma(z)

    @staticmethod
    def grad_z(ctx, z):
        return digamma(z)

    gradfuncs = grad_z,


# noinspection PyUnusedLocal,PyMethodOverriding
class Digamma(CargsMixin):
    save_output = False

    @staticmethod
    def saved_tensors(ctx, z):
        return z,

    @staticmethod
    def _forward(ctx, z, *args):
        return _special.digamma(z)

    @staticmethod
    def grad_z(ctx, z):
        return polygamma(1, z)

    gradfuncs = grad_z,


# noinspection PyUnusedLocal,PyMethodOverriding
class Polygamma(CargsMixin):
    save_output = False

    @staticmethod
    def saved_tensors(ctx, n, z):
        ctx.n = n
        return z,

    @staticmethod
    def _forward(ctx, z, n, *args):
        return _special.polygamma(n, z)
        # return (-1)**(n+1) * gamma(n+1) * zeta(n+1, z)

    @staticmethod
    def grad_z(ctx, z):
        return polygamma(ctx.n+1, z)

    gradfuncs = (None, grad_z)


gamma = with_complex_args(Gamma.apply)
loggamma = with_complex_args(Loggamma.apply)
digamma = psi = with_complex_args(Digamma.apply)


def polygamma(n: int, z: _TN) -> Tensor:
    return Polygamma.apply(n, *as_complex_tensors(z))


__all__ = 'gamma', 'loggamma', 'digamma', 'polygamma'

