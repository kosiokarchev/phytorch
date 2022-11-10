from contextlib import nullcontext

import torch
from more_itertools import circular_shifts

from ..extensions import elliptic as _elliptic
from ..math import where
from ..utils.function_context import CargsMixin, CimplMixin
from ..utils.symmetry import product


def feq(a, b, eps=None):
    return abs(a-b) < (eps if eps is not None else 1e-4 if torch.get_default_dtype() is torch.float else 1e-9)


# noinspection PyMethodOverriding,PyUnusedLocal
class Elliprc(CimplMixin, CargsMixin):
    ninputs = 2
    _impl_func = _elliptic.elliprc

    @staticmethod
    def grad_x(ctx, x, y, Rc):
        # see https://dlmf.nist.gov/19.20.iv and d/dx Rf
        general = (Rc - x**(-0.5)) / (2 * (y-x))

        # TODO: gradient cases
        if ctx.devil_may_care:
            return general

        return where(feq(x, y), -1/6 * y**(-1.5), general)

    @staticmethod
    def grad_y(ctx, x, y, Rc):
        # see https://dlmf.nist.gov/19.20.iv and d/dx Rf
        general = (Rc - x**0.5 / y) / (2 * (x-y))

        # TODO: gradient cases
        if ctx.devil_may_care:
            return general

        return where(feq(x, y), -1/3 * y**(-1.5), general)


# noinspection PyUnusedLocal
class Elliprd(CimplMixin, CargsMixin):
    ninputs = 3
    _impl_func = _elliptic.elliprd

    @staticmethod
    def grad_x(ctx, x, y, z, Rd):
        # https://wolfram.com/xid/0bnhv6qov8gan-ggq5f1
        general = (elliprd(y, z, x) - elliprd(x, y, z)) / (2 * (x - z))

        # TODO: gradient cases
        if ctx.devil_may_care:
            return general

        case1 = - 0.3 * x**(-2.5)
        case2 = 3/16 * ((5*x - 2*y) * y**0.5 / x**2 - 3 * elliprc(y, x)) / (x-y)**2
        return where(feq(x, z), where(feq(x, y), case1, case2), general)

    @classmethod
    def grad_y(cls, ctx, x, y, z, Rd):
        return cls.grad_x(ctx, y, x, z, Rd)

    @staticmethod
    def grad_z(ctx, x, y, z, Rd):
        # https://wolfram.com/xid/0bnhv6qov8gan-m3t8ei
        rzx, rzy = rz = tuple(1/(z-_) for _ in (x, y))
        srz, prz = sum(rz), product(rz)
        general = 1.5 * prz * (elliprf(x, y, z) - x**0.5 * y**0.5 * z**(-1.5)) - srz * Rd

        # TODO: gradient cases
        if ctx.devil_may_care:
            return general

        case1 = -0.9 * z**(-2.5)
        case2 = -9/16 * ((2*y - 5*z) * y**0.5 / z**2 + 3 * elliprc(y, z)) * rzy**2
        case3 = -9/16 * ((2*x - 5*z) * x**0.5 / z**2 + 3 * elliprc(x, z)) * rzx**2
        return where(feq(x, z), where(feq(y, z), case1, case2), where(feq(y, z), case3, general))


class SymmetricElliprMixin:
    @staticmethod
    def _grad(x, y, z):
        """Implements d/dz of symmetric function f(x, y, z)"""
        raise NotImplementedError

    @classmethod
    def _backward(cls, ctx):
        with torch.enable_grad() if ctx.requires_grad else nullcontext():
            return (
                cls._grad(*args[::-1][:-1], args[0]) if nig else None
                for nig, args in zip(ctx.needs_input_grad, circular_shifts(ctx.saved_tensors))
            )


class Elliprf(SymmetricElliprMixin, CimplMixin, CargsMixin):
    ninputs = 3
    save_output = False
    _impl_func = _elliptic.elliprf

    @staticmethod
    def _grad(x, y, z):
        # https://dlmf.nist.gov/19.18.E1
        return elliprd(x, y, z) / (-6)


class Elliprg(SymmetricElliprMixin, CimplMixin, CargsMixin):
    ninputs = 3
    save_output = False
    _impl_func = _elliptic.elliprg

    @staticmethod
    def _grad(x, y, z):
        # https://wolfram.com/xid/01ytcsih4q2-nr51hu
        return (3*elliprf(x, y, z) - z * elliprd(x, y, z)) / 12


# noinspection PyUnusedLocal
class Elliprj(CimplMixin, CargsMixin):
    # TODO: unsafe flag for elliprj
    ninputs = 4
    _impl_func = _elliptic.elliprj

    @staticmethod
    def grad_x(ctx, x, y, z, p, Rj):
        # https://wolfram.com/xid/0c0ph263h6mbl-fo4tj
        Rd = elliprd(y, z, x)
        general = (Rj - Rd) / (2 * (p - x))

        # TODO: gradient cases
        if ctx.devil_may_care:
            return general

        x2 = x**2
        _316x2 = 3/16 / x2
        _3x2 = 3*x2
        rxy, rxz = 1/(x-y), 1/(x-z)

        case1 = -0.3 * x**(-2.5)
        case2 = _316x2 * ((5 * x - 2 * z) * z**0.5 - _3x2 * elliprc(z, x)) * rxz**2
        case3 = _316x2 * ((5 * x - 2 * y) * y**0.5 - _3x2 * elliprc(y, x)) * rxy**2
        case4 = (elliprf(x, y, z) + 2 / 3 * (y + z - 2 * x) * Rd - y**0.5 * z**0.5 * x**(-1.5)) * 0.5 * rxy * rxz
        return where(feq(x, p), where(feq(x, y), where(feq(x, z), case1, case2), where(feq(x, z), case3, case4)), general)

    @classmethod
    def grad_y(cls, ctx, x, y, z, p, Rj):
        return cls.grad_x(ctx, y, x, z, p, Rj)

    @classmethod
    def grad_z(cls, ctx, x, y, z, p, Rj):
        return cls.grad_x(ctx, z, y, x, p, Rj)

    @staticmethod
    def grad_p(ctx, x, y, z, p, Rj):
        # https://wolfram.com/xid/0c0ph263h6mbl-dc1v40
        sx, sy, sz, sp = (_**0.5 for _ in (x, y, z, p))
        rpx, rpy, rpz = rp = tuple(1/(p-_) for _ in (x, y, z))
        srp, prp = sum(rp), product(rp)

        general = (3 * prp * (p * elliprf(x, y, z) - 2 * elliprg(x, y, z) + sx*sy*sz/p) - srp * Rj) / 2

        # TODO: gradient cases
        if ctx.devil_may_care:
            return general

        _2p = 2*p
        pm32 = p**(-1.5)
        rp2 = p**(-2)

        case1 = -0.6 * p**(-2.5)
        case2 = -0.375 * ((2*z - 5*p) * sz * rp2 + 3 * elliprc(z, p)) * rpz**2
        case3 = -0.375 * ((2*y - 5*p) * sy * rp2 + 3 * elliprc(y, p)) * rpy**2
        case4 = -0.375 * ((2*x - 5*p) * sx * rp2 + 3 * elliprc(x, p)) * rpx**2
        case5 = (elliprf(y, z, p) + 2/3 * (y+z - _2p) * elliprd(y, z, p) - sy*sz * pm32) * rpy * rpz
        case6 = (elliprf(x, z, p) + 2/3 * (x+z - _2p) * elliprd(x, z, p) - sx*sz * pm32) * rpx * rpz
        case7 = (elliprf(x, y, p) + 2/3 * (x+y - _2p) * elliprd(x, y, p) - sx*sy * pm32) * rpx * rpy

        pex, pey, pez = (feq(p, _) for _ in (x, y, z))
        pen = sum((pex, pey, pez))

        return where(
            pen >= 2,
            where(pen==3, case1, where(pex, where(pey, case2, case3), case4)),
            where(pen==0, general, where(pex, case5, where(pey, case6, case7)))
        )


elliprc = Elliprc.application()
elliprd = Elliprd.application()
elliprf = Elliprf.application()
elliprg = Elliprg.application()
elliprj = Elliprj.application()
