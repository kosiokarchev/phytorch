from __future__ import annotations

from abc import ABC, abstractmethod
from math import inf, pi
from typing import ClassVar, Union

from ..constants import c as speed_of_light, G as Newton_G
from ..math import complexify, csinc, log10, realise, sinc
from ..quantities.quantity import GenericQuantity
from ..units.angular import steradian
from ..units.astro import Mpc, pc
from ..units.si import km, s
from ..units.Unit import Unit
from ..utils._typing import _TN, ValueProtocol


_GQuantity = Union[GenericQuantity, Unit, ValueProtocol]


H100 = 100 * km/s/Mpc


class Cosmology(ABC):
    @staticmethod
    def scale_factor(z):
        return 1 / (z+1)

    @staticmethod
    def inv_scale_factor(z):
        return z+1


class FLRWDriver(Cosmology, ABC):
    Ok0: _TN

    @property
    def sqrtOk0(self) -> _TN:
        return complexify(self.Ok0)**0.5

    @property
    def isqrtOk0_pi(self) -> _TN:
        return 1j * self.sqrtOk0 / pi

    def transform_curved(self, distance_dimless: _TN) -> _TN:
        return distance_dimless * realise(sinc(self.isqrtOk0_pi * distance_dimless))

    @abstractmethod
    def e2func(self, z: _TN) -> _TN: ...

    def efunc(self, z: _TN) -> _TN:
        return self.e2func(z)**0.5

    def inv_efunc(self, z: _TN) -> _TN:
        return self.e2func(z)**(-0.5)

    def lookback_time_integrand(self, z: _TN) -> _TN:
        return self.inv_efunc(z) / (z+1)

    def abs_distance_integrand(self, z: _TN) -> _TN:
        return (z+1)**2 * self.inv_efunc(z)

    # TODO: @abstractmethodgroup
    def lookback_time_dimless(self, z: _TN) -> _TN:
        return self.age_dimless(0) - self.age_dimless(z)

    # TODO: @abstractmethodgroup
    def age_dimless(self, z: _TN) -> _TN:
        return self.lookback_time_dimless(inf)

    @abstractmethod
    def absorption_distance_dimless(self, z: _TN) -> _TN: ...

    # TODO: @abstractmethodgroup
    def comoving_distance_dimless(self, z: _TN) -> _TN:
        return self.comoving_distance_dimless_z1z2(0, z)

    # TODO: @abstractmethodgroup
    def comoving_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
        return self.comoving_distance_dimless(z2) - self.comoving_distance_dimless(z1)

    def comoving_transverse_distance_dimless(self, z: _TN) -> _TN:
        return self.transform_curved(self.comoving_distance_dimless(z))

    def comoving_transverse_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
        return self.transform_curved(self.comoving_distance_dimless_z1z2(z1, z2))

    def angular_diameter_distance_dimless(self, z: _TN) -> _TN:
        return self.comoving_transverse_distance_dimless(z) / (z+1)

    def angular_diameter_distance_dimless_z1z2(self, z1: _TN, z2: _TN):
        return self.comoving_transverse_distance_dimless_z1z2(z1, z2) / (z2+1)

    def luminosity_distance_dimless(self, z: _TN) -> _TN:
        return self.comoving_transverse_distance_dimless(z) * (z+1)

    def differential_comoving_volume_dimless(self, z):
        return self.comoving_transverse_distance_dimless(z)**2 * self.inv_efunc(z)

    def comoving_volume_dimless(self, z: _TN, eps=1e-8) -> _TN:
        dc = self.comoving_distance_dimless(z)
        return 8*pi*dc**3 * realise(csinc(2 * self.isqrtOk0_pi * dc, eps))


class FLRW(FLRWDriver, ABC):
    _critical_density_constant: ClassVar = 3 / (8 * pi * Newton_G)

    H0: _GQuantity = H100

    def H(self, z: _TN) -> _GQuantity:
        return self.H0 * self.efunc(z)

    @property
    def hubble_time(self) -> _GQuantity:
        return 1 / self.H0

    @property
    def hubble_distance(self) -> _GQuantity:
        return speed_of_light / self.H0

    @property
    def hubble_distance_in_10pc(self) -> _TN:
        return self.hubble_distance.to(10*pc).value

    @property
    def hubble_volume(self) -> _GQuantity:
        return self.hubble_distance**3

    @property
    def critical_density0(self) -> _GQuantity:
        return self._critical_density_constant * self.H0**2

    def critical_density(self, z: _TN) -> _GQuantity:
        return self.critical_density0 * self.e2func(z)

    def lookback_time(self, z: _TN) -> _GQuantity:
        return self.hubble_time * self.lookback_time_dimless(z)

    def lookback_distance(self, z: _TN) -> _GQuantity:
        return self.hubble_distance * self.lookback_time_dimless(z)

    def age(self, z: _TN) -> _GQuantity:
        return self.hubble_time * self.age_dimless(z)

    # For compatibility with astropy, the absorption_distance is dimensionless
    def absorption_distance(self, z: _TN) -> _TN:
        return self.absorption_distance_dimless(z)

    def comoving_distance(self, z: _TN) -> _GQuantity:
        return self.hubble_distance * self.comoving_distance_dimless(z)

    def comoving_distance_z1z2(self, z1: _TN, z2: _TN) -> _GQuantity:
        return self.hubble_distance * self.comoving_distance_dimless_z1z2(z1, z2)

    def comoving_transverse_distance(self, z: _TN) -> _GQuantity:
        return self.hubble_distance * self.comoving_transverse_distance_dimless(z)

    def comoving_transverse_distance_z1z2(self, z1: _TN, z2: _TN) -> _GQuantity:
        return self.hubble_distance * self.comoving_transverse_distance_dimless_z1z2(z1, z2)

    def angular_diameter_distance(self, z: _TN) -> _GQuantity:
        return self.hubble_distance * self.angular_diameter_distance_dimless(z)

    def angular_diameter_distance_z1z2(self, z1: _TN, z2: _TN) -> _GQuantity:
        return self.hubble_distance * self.angular_diameter_distance_dimless_z1z2(z1, z2)

    def luminosity_distance(self, z: _TN) -> _GQuantity:
        return self.hubble_distance * self.luminosity_distance_dimless(z)

    def differential_comoving_volume(self, z):
        return self.hubble_volume / steradian * self.differential_comoving_volume_dimless(z)

    def comoving_volume(self, z: _TN) -> _GQuantity:
        return self.hubble_volume * self.comoving_volume_dimless(z)

    def distmod(self, z: _TN):
        # TODO: return as magnitude quantity?
        return 2.5 * log10((self.luminosity_distance_dimless(z) * self.hubble_distance_in_10pc)**2)
