from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial, partialmethod
from math import pi
from typing import ClassVar, TYPE_CHECKING

from .core import _GQuantity, FLRW
from ..constants import c as speed_of_light, sigma as sigma_sb
from ..math import asinh
from ..units.unit import Unit
from ..utils._typing import _TN


class FlatFLRW(FLRW, ABC):
    Ok0 = 0

    def transform_curved(self, distance_dimless: _TN) -> _TN:
        return distance_dimless

    def comoving_volume_dimless(self, z: _TN, eps=1e-8) -> _TN:
        return 4*pi/3 * self.comoving_transverse_distance_dimless(z)**3


class BaseLambdaCDM(FLRW, ABC):
    @property
    @abstractmethod
    def Ok0(self) -> _TN: ...

    Om0: _TN
    Ode0: _TN
    Ob0: _TN = 0.

    @property
    def Odm0(self) -> _TN:
        return self.Om0 - self.Ob0

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

        if TYPE_CHECKING:
            def __call__(self, z: _TN) -> _TN: ...

    Om = _redshift_method(_redshift_matter)
    Ob = _redshift_method(_redshift_matter)
    Odm = _redshift_method(_redshift_matter)
    Ok = _redshift_method(_redshift_curvature)
    Ode = _redshift_method(_redshift_constant)


class BaseFlatLambdaCDM(FlatFLRW, BaseLambdaCDM, ABC):
    @property
    @abstractmethod
    def Ode0(self): ...

    @Ode0.setter
    def Ode0(self, value): raise NotImplementedError


class LambdaCDM(BaseLambdaCDM, ABC):
    @property
    def Ok0(self) -> _TN:
        return 1. - (self.Om0 + self.Ode0)

    def e2func(self, z: _TN) -> _TN:
        zp1 = z+1
        return zp1**2 * (self.Om0 * zp1 + self.Ok0) + self.Ode0


class LambdaCDMR(LambdaCDM, ABC):
    Or0: _TN = 0.
    Neff: _TN = 0.

    _radiation_density_constant: ClassVar = 4 * sigma_sb / speed_of_light**3
    _neutrino_energy_scale: ClassVar = 7/8 * (4/11)**(4/3)

    @property
    def Tcmb0(self):
        return (self.Or0 / (1 + self.Neff * self._neutrino_energy_scale) * self.critical_density0 / self._radiation_density_constant)**0.25

    def Tcmb(self, z: _TN) -> _GQuantity:
        return self.Tcmb0 * (1+z)

    @Tcmb0.setter
    def Tcmb0(self, value):
        self.Or0 = (1 + self.Neff * self._neutrino_energy_scale) * (value**4 * self._radiation_density_constant / self.critical_density0).to(Unit()).value

    _redshift_radiation = partial(LambdaCDM._redshift_power, power=4)
    Or = LambdaCDM._redshift_method(_redshift_radiation)

    @property
    def Ok0(self) -> _TN:
        return 1 - (self.Om0 + self.Ode0 + self.Or0)

    def e2func(self, z: _TN) -> _TN:
        zp1 = z+1
        return zp1**2 * ((self.Or0 * zp1 + self.Om0) * zp1 + self.Ok0) + self.Ode0


class FlatLambdaCDM(LambdaCDM, BaseFlatLambdaCDM, ABC):
    @property
    def Ode0(self):
        return 1 - self.Om0

    @Ode0.setter
    def Ode0(self, value):
        self.Om0 = 1 - value

    def age_dimless(self, z: _TN) -> _TN:
        return (2/3) / (1-self.Om0)**0.5 * asinh(((1/self.Om0 - 1) / (1+z)**3)**0.5)


class FlatLambdaCDMR(LambdaCDMR, BaseFlatLambdaCDM, ABC):
    @property
    def Ode0(self):
        return 1 - (self.Om0 + self.Or0)

    @Ode0.setter
    def Ode0(self, value):
        self.Om0 = 1 - (value + self.Or0)

    # TODO: optimise
    # def comoving_distance_dimless(self, z: _t) -> _t: ...
    # def lookback_time_dimless(self, z: _t) -> _t: ...
    # def age_dimless(self, z: _t) -> _t: ...


class AbstractBaseLambdaCDM(BaseLambdaCDM):
    def lookback_time_dimless(self, z: _TN) -> _TN: ...
    def age_dimless(self, z: _TN) -> _TN: ...
    def absorption_distance_dimless(self, z: _TN) -> _TN: ...


class AbstractLambdaCDM(AbstractBaseLambdaCDM, LambdaCDMR):
    pass


class AbstractFlatLambdaCDM(FlatLambdaCDM, AbstractLambdaCDM):
    pass


class AbstractLambdaCDMR(AbstractBaseLambdaCDM, LambdaCDMR):
    pass


class AbstractFlatLambdaCDMR(FlatLambdaCDMR, AbstractLambdaCDMR):
    pass
