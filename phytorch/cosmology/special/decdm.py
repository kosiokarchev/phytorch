from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, TypeVar

from ..core import FLRW, _acCosmologyT
from ..utils import _GQuantity, IndirectPropertyParameter, Parameter, PropertyParameter
from ...constants import c as speed_of_light, sigma as sigma_sb
from ...units.unit import Unit
from ...utils._typing import _TN


_BaseDECDMT = TypeVar('_BaseDECDMT', bound='BaseDECDM')


class BaseDECDM(FLRW[_BaseDECDMT, _acCosmologyT], ABC):
    _Ode0: _TN = object()

    # @property
    @IndirectPropertyParameter
    # TODO: @abstractmethodgroup
    def Ode0(self) -> _TN:
        return (self._Ode0 if self._Ode0 is not type(self)._Ode0
                else 1. - self.Ok0 - self._sum0_noDE)

    @Ode0.setter
    def Ode0(self, value: _TN):
        self._Ode0 = value

    @abstractmethod
    def _de_density_scale(self, zp1: _TN): ...

    def de_density_scale(self, z: _TN, zp1: _TN = None) -> _TN:
        return self._de_density_scale(1+z if zp1 is None else zp1)

    def _Ode(self, zp1: _TN):
        return self.Ode0 * self._de_density_scale(zp1) / self._e2func(zp1)

    def Ode(self, z: _TN, zp1: _TN = None) -> _TN:
        return self._Ode(1+z if zp1 is None else zp1)

    Om0: _TN = Parameter()
    Om = FLRW._redshift_method(FLRW._redshift_matter)

    Ob0: _TN = Parameter(0.)
    Ob = FLRW._redshift_method(FLRW._redshift_matter)

    @property
    def Odm0(self) -> _TN:
        return self.Om0 - self.Ob0

    Odm = FLRW._redshift_method(FLRW._redshift_matter)

    @property
    def _sum0_noDE(self) -> _TN:
        return self.Om0

    @property
    # TODO: @abstractmethodgroup
    def Ok0(self) -> _TN:
        return 1. - self._sum0_noDE - self.Ode0

    def _e2func(self, zp1: _TN) -> _TN:
        return zp1**2 * (self.Om0 * zp1 + self.Ok0) + self.Ode0 * self._de_density_scale(zp1)


class RadiationFLRWMixin(BaseDECDM, ABC):
    Or0: _TN = Parameter(0.)
    Or = FLRW._redshift_method(FLRW._redshift_radiation)

    Neff: _TN = Parameter(0.)

    _radiation_density_constant: ClassVar[Unit] = 4 * sigma_sb / speed_of_light**3
    _neutrino_energy_scale: ClassVar[float] = 7/8 * (4/11)**(4/3)

    @PropertyParameter
    def Tcmb0(self) -> _GQuantity:
        return (self.Or0 / (1 + self.Neff * self._neutrino_energy_scale) * self.critical_density0 / self._radiation_density_constant)**0.25

    @Tcmb0.setter
    def Tcmb0(self, value: _GQuantity):
        self.Or0 = (1 + self.Neff * self._neutrino_energy_scale) * (value**4 * self._radiation_density_constant / self.critical_density0).to(Unit()).value

    def Tcmb(self, z: _TN) -> _GQuantity:
        return self.Tcmb0 * (1+z)

    @property
    def _sum0_noDE(self) -> _TN:
        return super()._sum0_noDE + self.Or0
