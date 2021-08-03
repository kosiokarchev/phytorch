# TODO: annotate Number/Tensor?
# TODO: move to c++ kernels?
from typing import Optional

from torch import Tensor

from .ellipr import elliprd, elliprf, elliprg, elliprj
from ..math import sin, where
from ..utils._typing import _TN
from ..utils.function_context import ComplexTorchFunction


def _get_c(phi: Optional[Tensor], c: Tensor = None):
    try:
        return c if c is not None else 1 / sin(phi)**2
    except ZeroDivisionError:
        return float('inf')
    except ValueError:
        return float('nan')


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipK(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, m: Tensor):
        return 1-m, m

    @staticmethod
    def _forward(ctx, mm, *args):
        return elliprf(0, mm, 1)

    @staticmethod
    def grad_m(ctx, mm: Tensor, m: Tensor, K: Tensor):
        return (ellipe(m) - mm * K) / (2*m*mm)

    gradfuncs = grad_m,


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipE(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, m: Tensor):
        return m,

    @staticmethod
    def _forward(ctx, m, *args):
        return 2 * elliprg(0, 1-m, 1)

    @staticmethod
    def grad_m(ctx, m: Tensor, E: Tensor):
        return (E - ellipk(m)) / (2*m)

    gradfuncs = grad_m,


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipD(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, m):
        return 1-m, m

    @staticmethod
    def _forward(ctx, mm, *args):
        return elliprd(0, mm, 1) / 3

    @staticmethod
    def grad_m(ctx, mm, m, D):
        return ((m-2)*D + ellipk(m)) / (2*m*mm)

    gradfuncs = grad_m,


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipPi(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, n:  Tensor, m: Tensor):
        return n, m

    @staticmethod
    def _forward(ctx, n, m, *args):
        # TODO: edge cases n=1 or m=1
        return ellipk(m) + n / 3 * elliprj(0, 1 - m, 1, 1 - n)

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
        return m, _get_c(phi, c)

    @staticmethod
    def _forward(ctx, m, c, *args) -> Tensor:
        return elliprf(c-1, c-m, c)

    @staticmethod
    def grad_phi(ctx, m, c, F):
        return (1-c/m)**(-0.5)

    @staticmethod
    def grad_m(ctx, m, c, F):
        mm = 1-m
        return ((ellipeinc(None, m, c) - mm*F) / m - ((c-1) / (c-m) / c)**0.5) / (2*mm)

    gradfuncs = grad_phi, grad_m
    ninputs = 3


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipEinc(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, phi: Optional[_TN], m: _TN, c: _TN = None):
        return m, _get_c(phi, c)

    @staticmethod
    def _forward(ctx, m, c, *args) -> Tensor:
        return where(~((c==1) & (m==1)), elliprf(c-1, c-m, c) - m * ellipdinc(None, m, c), 1)

        # https://dlmf.nist.gov/19.25.E11, but NaNs when c==1
        # return ((c-m) / c / (c-1+eps))**0.5 - (1-m)/3 * elliprd(c-m, c, c-1)

    @staticmethod
    def grad_phi(ctx, m, c, Einc):
        return (1-m/c)**0.5

    @staticmethod
    def grad_m(ctx, m, c, Einc):
        return (Einc - ellipkinc(None, m, c)) / m / 2

    gradfuncs = grad_phi, grad_m
    ninputs = 3


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipDinc(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, phi: Optional[_TN], m: _TN, c: _TN = None):
        return m, _get_c(phi, c)

    @staticmethod
    def _forward(ctx, m, c, *args):
        return elliprd(c-1, c-m, c) / 3

    @staticmethod
    def grad_phi(ctx, m, c, Dinc):
        return (1-m/c)**(-0.5) / c

    @staticmethod
    def grad_m(ctx, m, c, Dinc):
        return (((m-2)*Dinc + ellipkinc(None, m, c)) - ((c-1) / (c-m) / c)**0.5) / (2*m*(1-m))

    gradfuncs = grad_phi, grad_m


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipPiinc(ComplexTorchFunction):
    @staticmethod
    def saved_tensors(ctx, n: _TN, phi: Optional[_TN], m: _TN, c: _TN = None):
        return n, m, _get_c(phi, c)

    @staticmethod
    def _forward(ctx, n, m, c, *args) -> Tensor:
        return ellipkinc(None, m, c) + n / 3 * elliprj(c - 1, c - m, c, c - n)

    @staticmethod
    def grad_phi(ctx, n, m, c, Piinc):
        return (1-m/c)**(-0.5) / (1-n/c)

    @staticmethod
    def grad_n(ctx, n, m, c, Piinc):
        mn = m-n
        return (
           ellipeinc(None, m, c) + mn/n * ellipkinc(None, m, c) + (n - m/n) * Piinc
           - n * ((c-1) * (c-m) / (c-n)**2 / c)**0.5
        ) / (2 * mn * (n-1))

    @staticmethod
    def grad_m(ctx, n, m, c, Piinc):
        m1 = m-1
        return (
            ellipeinc(None, m, c) + m1*Piinc
            - m * ((c-1) / (c-m) / c)**0.5
        ) / (2 * (n-m) * m1)

    gradfuncs = grad_n, grad_phi, grad_m
    ninputs = 4


ellipk = EllipK.apply
ellipe = EllipE.apply
ellipd = EllipD.apply
ellippi = EllipPi.apply

ellipkinc = ellipf = EllipKinc.apply
ellipeinc = EllipEinc.apply
ellippiinc = EllipPiinc.apply
ellipdinc = EllipDinc.apply
