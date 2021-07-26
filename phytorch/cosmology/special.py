from __future__ import annotations

from abc import ABC
from functools import partial, partialmethod
from math import pi

from .core import FLRW
from ..utils._typing import _TN


class FlatFLRW(FLRW, ABC):
    Ok0 = 0

    def transform_curved(self, distance_dimless: _TN) -> _TN:
        return distance_dimless

    def comoving_volume_dimless(self, z: _TN, eps=1e-8) -> _TN:
        return 4*pi/3 * self.comoving_transverse_distance_dimless(z)**3


class BaseLambdaCDM(FLRW, ABC):
    Om0: _TN
    Ode0: _TN
    Ob0: _TN = 0.

    @property
    def Odm0(self) -> _TN:
        return self.Om0 - self.Ob0

    @property
    def Ok0(self) -> _TN:
        return 1. - (self.Om0 + self.Ode0)

    def e2func(self, z: _TN) -> _TN:
        zp1 = z+1
        return zp1**2 * (self.Om0 * zp1 + self.Ok0) + self.Ode0

    def _redshift_power(self, z: _TN, O0_name: str, power: float) -> _TN:
        return getattr(self, O0_name) * (z+1)**power / self.e2func(z)

    _redshift_matter = partial(_redshift_power, power=3)
    _redshift_curvature = partial(_redshift_power, power=2)

    def _redshift_constant(self, z: _TN, O0_name: str) -> _TN:
        return getattr(self, O0_name) / self.e2func(z)

    class _redshift_method:
        def __init__(self, _redshift_method):
            self._redshift_method = _redshift_method

        def __set_name__(self, owner: LambdaCDMR, name):
            setattr(owner, name, partialmethod(self._redshift_method, O0_name=f'{name}0'))

    Om = _redshift_method(_redshift_matter)
    Ob = _redshift_method(_redshift_matter)
    Odm = _redshift_method(_redshift_matter)
    Ok = _redshift_method(_redshift_curvature)
    Ode = _redshift_method(_redshift_constant)


class LambdaCDM(BaseLambdaCDM, ABC):
    pass


class LambdaCDMR(LambdaCDM, ABC):
    Or0: _TN = 0.

    @property
    def Ok0(self) -> _TN:
        return super().Ok0 - self.Or0

    def e2func(self, z: _TN) -> _TN:
        zp1 = z+1
        return zp1**2 * ((self.Or0 * zp1 + self.Om0) * zp1 + self.Ok0) + self.Ode0

    _redshift_radiation = partial(LambdaCDM._redshift_power, power=4)
    Or = LambdaCDM._redshift_method(_redshift_radiation)


class FlatLambdaCDMR(FlatFLRW, LambdaCDMR, ABC):
    # TODO: optimised for ellipr
    # def comoving_distance_dimless(self, z: _t) -> _t: ...
    # def lookback_time_dimless(self, z: _t) -> _t: ...
    # def age_dimless(self, z: _t) -> _t: ...
    pass
