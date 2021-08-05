# TODO: annotate Number/Tensor?
from typing import Optional

from torch import Tensor

from ..extensions import elliptic as _elliptic
from ..utils._typing import _TN
from ..utils.complex import with_complex_args
from ..utils.function_context import ComplexTorchFunction


def _get_c(phi: Optional[Tensor], c: Tensor = None):
    try:
        return c if c is not None else _elliptic.csc2(phi)
    except ZeroDivisionError:
        return float('inf')
    except ValueError:
        return float('nan')


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipK(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, m: Tensor):
        return m, 1-m

    @staticmethod
    def _forward(ctx, m, *args):
        return _elliptic.ellipk(m)

    @staticmethod
    def grad_m(ctx, m: Tensor, mm: Tensor, K: Tensor):
        return (ellipe(m) - mm * K) / (2*m*mm)

    gradfuncs = grad_m,


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipE(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, m: Tensor):
        return m,

    @staticmethod
    def _forward(ctx, m, *args):
        return _elliptic.ellipe(m)

    @staticmethod
    def grad_m(ctx, m: Tensor, E: Tensor):
        return (E - ellipk(m)) / (2*m)

    gradfuncs = grad_m,


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipD(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, m):
        return m,

    @staticmethod
    def _forward(ctx, m, *args):
        return _elliptic.ellipd(m)

    @staticmethod
    def grad_m(ctx, m, D):
        return ((m-2)*D + ellipk(m)) / (2*m*(1-m))

    gradfuncs = grad_m,


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipPi(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, n:  Tensor, m: Tensor):
        return n, m

    @staticmethod
    def _forward(ctx, n, m, *args):
        # TODO: edge cases n=1 or m=1
        return _elliptic.ellippi(n, m)

    @staticmethod
    def grad_n(ctx, n, m, Pi):
        mn = m-n
        return (ellipe(m) + mn/n * ellipk(m) + (n - m/n) * Pi) / (2 * mn * (n-1))

    @staticmethod
    def grad_m(ctx, n, m, Pi):
        return (ellipe(m) / (m-1) + Pi) / (2 * (n-m))

    gradfuncs = grad_n, grad_m


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipKinc(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, phi: Optional[_TN], m: _TN, c: _TN = None):
        return _get_c(phi, c), m

    @staticmethod
    def _forward(ctx, c, m, *args) -> Tensor:
        return _elliptic.ellipkinc_(c, m)

    @staticmethod
    def grad_phi(ctx, c, m, F):
        return (1-c/m)**(-0.5)

    @staticmethod
    def grad_m(ctx, c, m, F):
        mm = 1-m
        return ((ellipeinc(None, m, c) - mm*F) / m - ((c-1) / (c-m) / c)**0.5) / (2*mm)

    gradfuncs = grad_phi, grad_m
    ninputs = 3


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipEinc(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, phi: Optional[_TN], m: _TN, c: _TN = None):
        return _get_c(phi, c), m

    @staticmethod
    def _forward(ctx, c, m, *args) -> Tensor:
        return _elliptic.ellipeinc_(c, m)

    @staticmethod
    def grad_phi(ctx, c, m, Einc):
        return (1-m/c)**0.5

    @staticmethod
    def grad_m(ctx, c, m, Einc):
        return (Einc - ellipkinc(None, m, c)) / m / 2

    gradfuncs = grad_phi, grad_m
    ninputs = 3


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipDinc(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, phi: Optional[_TN], m: _TN, c: _TN = None):
        return _get_c(phi, c), m

    @staticmethod
    def _forward(ctx, c, m, *args):
        return _elliptic.ellipdinc_(c, m)

    @staticmethod
    def grad_phi(ctx, c, m, Dinc):
        return (1-m/c)**(-0.5) / c

    @staticmethod
    def grad_m(ctx, c, m, Dinc):
        return (((m-2)*Dinc + ellipkinc(None, m, c)) - ((c-1) / (c-m) / c)**0.5) / (2*m*(1-m))

    gradfuncs = grad_phi, grad_m


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipPiinc(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, n: _TN, phi: Optional[_TN], m: _TN, c: _TN = None):
        return n, _get_c(phi, c), m

    @staticmethod
    def _forward(ctx, n, c, m, *args) -> Tensor:
        return _elliptic.ellippiinc_(n, c, m)

    @staticmethod
    def grad_phi(ctx, n, c, m, Piinc):
        return (1-m/c)**(-0.5) / (1-n/c)

    @staticmethod
    def grad_n(ctx, n, c, m, Piinc):
        mn = m-n
        return (
           ellipeinc(None, m, c) + mn/n * ellipkinc(None, m, c) + (n - m/n) * Piinc
           - n * ((c-1) * (c-m) / (c-n)**2 / c)**0.5
        ) / (2 * mn * (n-1))

    @staticmethod
    def grad_m(ctx, n, c, m, Piinc):
        m1 = m-1
        return (
            ellipeinc(None, m, c) + m1*Piinc
            - m * ((c-1) / (c-m) / c)**0.5
        ) / (2 * (n-m) * m1)

    gradfuncs = grad_n, grad_phi, grad_m
    ninputs = 4


ellipk = with_complex_args(EllipK.apply)
ellipe = with_complex_args(EllipE.apply)
ellipd = with_complex_args(EllipD.apply)
ellippi = with_complex_args(EllipPi.apply)

ellipkinc = ellipf = with_complex_args(EllipKinc.apply)
ellipeinc = with_complex_args(EllipEinc.apply)
ellipdinc = with_complex_args(EllipDinc.apply)
ellippiinc = with_complex_args(EllipPiinc.apply)
