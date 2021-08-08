# TODO: annotate Number/Tensor?
from math import inf, pi
from typing import Callable, Optional

import torch
from torch import Tensor

from ..extensions import elliptic as _elliptic
from ..math import where
from ..utils.complex import with_complex_args
from ..utils.function_context import CargsMixin, CimplMixin


class CosecantSquared(CimplMixin, CargsMixin):
    _impl_func = _elliptic.csc2

    @staticmethod
    def grad_phi(ctx, phi, csc2):
        return - 2 * csc2 / torch.tan(phi)


# noinspection PyUnusedLocal
class EllipK(CimplMixin, CargsMixin):
    _impl_func = _elliptic.ellipk

    @staticmethod
    def grad_m(ctx, m: Tensor, K: Tensor):
        mm = 1-m
        return where(m != 0, (ellipe(m) - mm * K) / (2*m*mm), pi/8)


# noinspection PyUnusedLocal
class EllipE(CimplMixin, CargsMixin):
    _impl_func = _elliptic.ellipe

    @staticmethod
    def grad_m(ctx, m: Tensor, E: Tensor):
        return where(m != 0, (E - ellipk(m)) / (2*m), -pi/8)


# noinspection PyUnusedLocal
class EllipD(CimplMixin, CargsMixin):
    _impl_func = _elliptic.ellipd

    @staticmethod
    def grad_m(ctx, m, D):
        return where(m != 0, ((m-2)*D + ellipk(m)) / (2*m*(1-m)), 3*pi/32)


# noinspection PyUnusedLocal
class EllipPi(CimplMixin, CargsMixin):
    _impl_func = _elliptic.ellippi

    @staticmethod
    def grad_n(ctx, n, m, Pi):
        mn = m-n
        return (ellipe(m) + mn/n * ellipk(m) + (n - m/n) * Pi) / (2 * mn * (n-1))

    @staticmethod
    def grad_m(ctx, n, m, Pi):
        return (ellipe(m) / (m-1) + Pi) / (2 * (n-m))


class BaseEllipinc(CimplMixin, CargsMixin):
    @staticmethod
    def _get_c(phi: Optional[Tensor], c: Tensor = None):
        if phi is not None and c is not None:
            raise ValueError('passing both phi and c is not allowed.')
        return c if c is not None else CosecantSquared.apply(phi)

    _impl_func: Callable[[Tensor, ...], Tensor]

    @staticmethod
    def _grad_phi(ctx, c, m, res):
        raise NotImplementedError

    @classmethod
    def grad_c(cls, *args):
        c = args[-3]  # args = (ctx, (n,) c, m, res)
        return cls._grad_phi(*args) / (-2 * c * (c - 1)**0.5)

    @classmethod
    def _grad_at_phi_zero(cls, c, val):
        return where(c != inf, val, 0)

    @staticmethod
    def grad_m(ctx, *args):
        raise NotImplementedError


# noinspection PyMethodOverriding
class EllipKinc(BaseEllipinc):
    _impl_func = _elliptic.ellipkinc_

    @staticmethod
    def _grad_phi(ctx, c, m, F):
        return (1-m/c)**(-0.5)

    @classmethod
    def grad_m(cls, ctx, c, m, F):
        mm = 1-m
        return cls._grad_at_phi_zero(c, where(
            m != 0,
            ((ellipeinc(None, m, c) - mm*F) / m - ((c-1) / (c-m) / c)**0.5) / (2*mm),
            (torch.asin(1/c**0.5) - (c-1)**0.5 / c) / 4
        ))


# noinspection PyMethodOverriding
class EllipEinc(BaseEllipinc):
    _impl_func = _elliptic.ellipeinc_

    @staticmethod
    def _grad_phi(ctx, c, m, Einc):
        return (1-m/c)**0.5

    @classmethod
    def grad_m(cls, ctx, c, m, Einc):
        return cls._grad_at_phi_zero(c, where(
            m != 0,
            (Einc - ellipkinc(None, m, c)) / m / 2,
            ((c-1)**0.5 / c - torch.asin(1/c**0.5)) / 4
        ))


# noinspection PyMethodOverriding
class EllipDinc(BaseEllipinc):
    _impl_func = _elliptic.ellipdinc_

    @staticmethod
    def _grad_phi(ctx, c, m, Dinc):
        return (1-m/c)**(-0.5) / c

    @classmethod
    def grad_m(cls, ctx, c, m, Dinc):
        return cls._grad_at_phi_zero(c, where(
            m != 0,
            (((m-2)*Dinc + ellipkinc(None, m, c)) - ((c-1) / (c-m) / c)**0.5) / (2*m*(1-m)),
            (3*torch.asin(1/c**0.5) - (2 + 3*c) * (c-1)**0.5 / c**2) / 16
        ))


# noinspection PyMethodOverriding,PyUnusedLocal
class EllipPiinc(BaseEllipinc, CargsMixin):
    _impl_func = _elliptic.ellippiinc_

    @staticmethod
    def _grad_phi(ctx, n, c, m, Piinc):
        return (1-m/c)**(-0.5) / (1-n/c)

    @classmethod
    def grad_n(cls, ctx, n, c, m, Piinc):
        mn = m-n
        return cls._grad_at_phi_zero(c, (
           ellipeinc(None, m, c) + mn/n * ellipkinc(None, m, c) + (n - m/n) * Piinc
           - n * ((c-1) * (c-m) / (c-n)**2 / c)**0.5
        ) / (2 * mn * (n-1)))

    @classmethod
    def grad_m(cls, ctx, n, c, m, Piinc):
        m1 = m-1
        return cls._grad_at_phi_zero(c, (
            ellipeinc(None, m, c) + m1*Piinc
            - m * ((c-1) / (c-m) / c)**0.5
        ) / (2 * (n-m) * m1))

    @classmethod
    def _update_gradfuncs(cls):
        cls.gradfuncs[:0] = (cls.gradfuncs.pop(-1),)


ellipk = EllipK.apply
ellipe = EllipE.apply
ellipd = EllipD.apply
ellippi = EllipPi.apply


@with_complex_args
def ellipkinc(phi, m, c=None):
    return EllipKinc.apply(BaseEllipinc._get_c(phi, c), m)


@with_complex_args
def ellipeinc(phi, m, c=None):
    return EllipEinc.apply(BaseEllipinc._get_c(phi, c), m)


@with_complex_args
def ellipdinc(phi, m, c=None):
    return EllipDinc.apply(BaseEllipinc._get_c(phi, c), m)


@with_complex_args
def ellippiinc(n, phi, m, c=None):
    return EllipPiinc.apply(n, BaseEllipinc._get_c(phi, c), m)


ellipf = ellipkinc
