from abc import ABC
from itertools import chain
from operator import neg
from typing import Callable, ClassVar, Iterable

from torch import Tensor

from ._analytic import BaseAnalyticFLRWDriver, BaseAnalyticLambdaCDM, BaseAnalyticLambdaCDMR
from .. import special
from ...special.elliptic import ellipeinc, ellipkinc, ellippiinc, elliprc, elliprf, elliprj
from ...special.elliptic_reduction.symbolic import SymbolicEllipticReduction
from ...utils._typing import _TN


class AnalyticFLRWDriver(BaseAnalyticFLRWDriver, ABC):
    _integral_comoving_distance: ClassVar[Callable[[Iterable[_TN], tuple[_TN, _TN]], Tensor]]
    _integral_lookback_time: ClassVar[Callable[[Iterable[_TN], tuple[_TN, _TN]], Tensor]]
    _integral_absorption_distance: ClassVar[Callable[[Iterable[_TN], tuple[_TN, _TN]], Tensor]]

    # TODO: do we really need __init_subclass__?!
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

        rednn = SymbolicEllipticReduction.get(cls._epoly_degree, cls._epoly_degree)
        redn1n = SymbolicEllipticReduction.get(cls._epoly_degree+1, cls._epoly_degree)

        cls._integral_comoving_distance = staticmethod(rednn.desymbolise(rednn.Ie(0)))
        cls._integral_lookback_time = staticmethod(redn1n.desymbolise(redn1n.Ie(-cls._epoly_degree-1)))
        cls._integral_absorption_distance = staticmethod(redn1n.desymbolise(redn1n.Im(
            cls._epoly_degree*(0,) + (2,)
        )))

    def lookback_time_dimless(self, z: _TN) -> _TN:
        return self._fix_dimless(self._integral_lookback_time(
            chain(map(neg, self._epoly_roots), (1,)),
            (0, z)
        ))

    def age_dimless(self, z: _TN) -> _TN:
        raise NotImplementedError

    def absorption_distance_dimless(self, z: _TN) -> _TN:
        return self._fix_dimless(self._integral_absorption_distance(
            chain(map(neg, self._epoly_roots), (1,)),
            (0, z)
        ))

    def comoving_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
        return self._fix_dimless(self._integral_comoving_distance(
            map(neg, self._epoly_roots),
            (z1, z2)
        ))


class LambdaCDM(AnalyticFLRWDriver, BaseAnalyticLambdaCDM):
    def _get_ellip_params(self, *zs):
        a, b, c = self._epoly_roots
        rba = 1 / (b-a)
        m = (c-a) * rba
        return (a, b, c), m, rba, *((z-a)*rba for z in zs)

    def _ellip_comoving_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
        # https://www.wolframalpha.com/input/?i=integrate+1%2Fsqrt%28%28x-a%29*%28x-b%29*%28x-c%29%29
        # Gradshteyn and Ryzhik, 3.131:8 with a, b, c = c, a, b
        _, m, rba, c1, c2 = self._get_ellip_params(z1, z2)
        return -2 * self._fix_dimless(rba**0.5 * (
            ellipkinc(None, m, c2) - ellipkinc(None, m, c1)
        ))

    def _ellip_lookback_time_dimless(self, z: _TN) -> _TN:
        # https://www.wolframalpha.com/input/?i=integrate+1%2Fsqrt%28%28x-a%29*%28x-b%29*%28x-c%29%29+%2F+%28x%2B1%29
        # Gradeshteyn and Ryzhik, 3.137:8 with a, b, c = c, a, b
        (a, b, c), m, rba, c1, c2 = self._get_ellip_params(0, z)
        a1 = a+1
        n = - a1 * rba

        f = lambda x, cx: (
            -n/3 * elliprj(cx-1, cx-m, cx, cx-n)
            # re-written using https://dlmf.nist.gov/19.25.E14 from
            # ellipkinc(None, m, cx) - ellippiinc(n, None, m, cx)
        )

        return (-2) * self._fix_dimless(rba**0.5 / a1 * (f(z, c2) - f(0, c1)))

    def _ellip_absorption_distance_dimless(self, z: _TN) -> _TN:
        # https://www.wolframalpha.com/input/?i=integrate+1%2Fsqrt%28%28x-b%29*%28x-a%29*%28x-c%29%29+*+%28x%2B1%29%5E2
        # note: a <-> b !
        (a, b, c), m, rba, c1, c2 = self._get_ellip_params(0, z)

        coeffk = (2*b**2 + b*(a+c+6) - a*c + 3) / (a-b)**0.5
        coeffe = 2 * (a-b)**0.5 * (b+a+c+3)

        f = lambda x, cx: (
            self.efunc(x) / self._epoly_leading**0.5 * (2 * (b+a+c+3) / (x-a) + 1)
            + 1j * (coeffk * ellipkinc(None, m, cx)
                    + coeffe * ellipeinc(None, m, cx))
        )

        return 2/3 * self._fix_dimless(f(z, c2) - f(0, c1))

    def _get_ellipr_params(self, z1, z2):
        rts = tuple(self._epoly_roots)
        d11, d12, d13, d21, d22, d23 = (
            (z - r)**0.5
            for z in (z1, z2) for r in rts
        )
        return rts, (d11, d12, d13, d21, d22, d23), (
            (d11*d22*d23 + d21*d12*d13)**2,
            (d11*d22*d13 + d21*d12*d23)**2,
            (d11*d12*d23 + d21*d22*d13)**2)

    def _ellipr_comoving_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
        return 2 * (z2-z1) * self._fix_dimless(elliprf(*self._get_ellipr_params(z1, z2)[-1]))

    def _ellipr_lookback_time_dimless(self, z: _TN) -> _TN:
        (r1, r2, r3), (d11, d12, d13, d21, d22, d23), (u1, u2, u3) = self._get_ellipr_params(0, z)
        p = (r1+1) * z**2 + u1

        return 2 * z * self._fix_dimless(
            elliprc((d11*d12*d13*(z+1) + d21*d22*d23)**2, (z+1)*p)
            + z**2 / 3 * elliprj(u1, u2, u3, p)
        )

    comoving_distance_dimless_z1z2 = _ellipr_comoving_distance_dimless_z1z2
    lookback_time_dimless = _ellipr_lookback_time_dimless


class LambdaCDMR(AnalyticFLRWDriver, BaseAnalyticLambdaCDMR):
    def _get_elliptic_params(self, *zs):
        a, b, c, d = self._epoly_roots
        racbd = 1 / (a-c) / (b-d)
        m = (a-d) * (b-c) * racbd
        adbd = (a-d)/(b-d)
        return (a, b, c, d), m, racbd, *((z-b)/(z-a) * adbd for z in zs)

    def _ellip_comoving_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
        # Gradshteyn and Ryzhik, 3.147:8, TODO: but with a minus
        _, m, racbd, c1, c2 = self._get_elliptic_params(z1, z2)
        return -2 * self._fix_dimless(racbd**0.5 * (
            ellipkinc(None, m, c2) - ellipkinc(None, m, c1)
        ))

    def _ellip_lookback_time_dimless(self, z: _TN) -> _TN:
        # Gradshteyn and Ryzhik, 3.151:8, TODO: but with a minus
        (a, b, c, d), m, racbd, c1, c2 = self._get_elliptic_params(0, z)

        f = lambda cx: (
            (a-b) * ellippiinc(((a-d)*(b+1)) / ((b-d)*(a+1)), None, m, cx)
            - (a+1) * ellipkinc(None, m, cx)
        )
        return 2 * self._fix_dimless(racbd**0.5 / (a+1)/(b+1) * (f(c2) - f(c1)))

    def _get_ellipr_params(self, z1, z2):
        rts = tuple(self._epoly_roots)
        d11, d12, d13, d14, d21, d22, d23, d24 = (
            (z - r)**0.5
            for z in (z1, z2) for r in self._epoly_roots
        )
        return rts, (d11, d12, d13, d14, d21, d22, d23, d24), (
            (d11*d12*d23*d24 + d21*d22*d13*d14)**2,
            (d11*d22*d13*d24 + d21*d12*d23*d14)**2,
            (d11*d22*d23*d14 + d21*d12*d13*d24)**2)

    def _ellipr_comoving_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
        return 2 * (z2-z1) * self._fix_dimless(elliprf(*self._get_ellipr_params(z1, z2)[-1]))

    def _ellipr_lookback_time_dimless(self, z: _TN) -> _TN:
        (r1, r2, r3, r4), (d11, d12, d13, d14, d21, d22, d23, d24), (u1, u2, u3) = self._get_ellipr_params(0, z)
        p = u1 - (r3-r1)*(r4-r1)*(r2+1)/(r1+1) * z**2

        return 2 * z * self._fix_dimless((
            elliprf(u1, u2, u3)
            + z**2 / 3 * (r2-r1) * (r3-r1) * (r4-r1) / (r1+1) * elliprj(u1, u2, u3, p)
            - elliprc((d12 * d13 * d14 / d11 * (z + 1) + d22 * d23 * d24 / d21)**2, (z + 1) / d11**2 / d21**2 * p)
        ) / (r1+1))

    comoving_distance_dimless_z1z2 = _ellipr_comoving_distance_dimless_z1z2
    lookback_time_dimless = _ellipr_lookback_time_dimless


class FlatLambdaCDM(special.FlatLambdaCDM, LambdaCDM):
    pass
    # TODO: hyp2f1: doesn't seem necessary
    # def comoving_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
    #     # hyttps://www.wolframalpha.com/input/?i=integrate+1+%2F+sqrt%28a*x%5E3+%2B+%281-a%29%29
    #     r = self.Om0 / (self.Om0-1)
    #     zp11, zp12 = 1+z1, 1+z2
    #     return ((1-self.Om0)**(-0.5) * (
    #         zp12 * hyp2f1(1/3, 1/2, 4/3, zp12**3 * r)
    #         - zp11 * hyp2f1(1/3, 1/2, 4/3, zp11**3 * r)
    #     )).real


class FlatLambdaCDMR(special.FlatLambdaCDMR, LambdaCDMR):
    pass
