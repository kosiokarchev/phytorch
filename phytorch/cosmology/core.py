from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial, partialmethod
from math import inf, pi
from typing import Any, ClassVar, Generic, get_type_hints, Mapping, TYPE_CHECKING, TypeVar

import forge
import torch

from .utils import _GQuantity, _no_value, AbstractParameter, Parameter, PropertyParameter
from ..constants import c as speed_of_light, G as Newton_G
from ..math import complexify, csinc, log10, realise, sinc
from ..units.angular import steradian
from ..units.astro import Mpc, pc
from ..units.si import km, s
from ..units.unit import Unit
from ..utils._typing import _TN
from ..utils.interop import _astropy, AstropyConvertible, BaseToAstropy


H100 = 100 * km/s/Mpc


_CosmologyT = TypeVar('_CosmologyT', bound='Cosmology')
_FLRWDriverT = TypeVar('_FLRWDriverT', bound='FLRWDriver')
_FLRWT = TypeVar('_FLRWT', bound='FLRW')
_acCosmologyT = TypeVar('_acCosmologyT', bound=_astropy.cosmology.Cosmology)


class Cosmology(AstropyConvertible[_CosmologyT, _acCosmologyT], Generic[_CosmologyT, _acCosmologyT], ABC):
    _parameter = object()
    _parameters: Mapping[str, tuple[Any, AbstractParameter]]

    def _set_params(self, **kwargs):
        for key, val in kwargs.items():
            if val is not _no_value:
                setattr(self, key, val)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        anns = get_type_hints(cls)

        cls._parameters = {
            key: (anns.get(key, Any), obj)
            for key in dir(cls) for obj in [getattr(cls, key, forge.empty)]
            if isinstance(obj, AbstractParameter)
        }

        cls.__init__ = forge.sign(forge.self, *(
            forge.kwarg(name=name, type=ann, default=obj.default)
            for name, (ann, obj) in cls._parameters.items()
        ))(cls.__init__)

    def __init__(self, **kwargs):
        super().__init__()
        for key, (ann, obj) in self._parameters.items():
            if isinstance(obj, Parameter) and obj.default is not Parameter.default:
                setattr(self, key, obj.default)
        self._set_params(**kwargs)


    @property
    def parameters(self) -> Mapping[str, Any]:
        return {key: getattr(self, key, None) for key, val in self._parameters.items()
                if not isinstance(val, PropertyParameter)}

    @staticmethod
    def scale_factor(z):
        return 1 / (z+1)

    @staticmethod
    def inv_scale_factor(z):
        return z+1


    class _toAstropy(BaseToAstropy[_CosmologyT, _acCosmologyT]):
        _cls = _astropy.cosmology.Cosmology

        def __call__(self):
            return self._cls.from_format({
                key: val.toAstropy() if isinstance(val, AstropyConvertible)
                else val.numpy() if torch.is_tensor(val) else val
                for key, val in self._.parameters.items()
            }, cosmology=self._cls, move_to_meta=True, format='mapping')


class FLRWDriver(Cosmology[_FLRWDriverT, _acCosmologyT], ABC):
    def _redshift_power(self, z: _TN, O0_name: str, power: float) -> _TN:
        return getattr(self, O0_name) * (z+1)**power / self.e2func(z)

    _redshift_radiation = partial(_redshift_power, power=4)
    _redshift_matter = partial(_redshift_power, power=3)
    _redshift_curvature = partial(_redshift_power, power=2)

    def _redshift_constant(self, z: _TN, O0_name: str) -> _TN:
        return getattr(self, O0_name) / self.e2func(z)

    class _redshift_method:
        def __init__(self, _redshift_method):
            self._redshift_method = _redshift_method

        def __set_name__(self, owner: FLRWDriver, name):
            setattr(owner, name, partialmethod(self._redshift_method, O0_name=f'{name}0'))

        if TYPE_CHECKING:
            def __call__(self, z: _TN) -> _TN: ...

    Ok0: _TN = Parameter()
    Ok = _redshift_method(_redshift_curvature)

    @property
    def sqrtOk0(self) -> _TN:
        return complexify(self.Ok0)**0.5

    @property
    def isqrtOk0_pi(self) -> _TN:
        return 1j * self.sqrtOk0 / pi

    def transform_curved(self, distance_dimless: _TN) -> _TN:
        return distance_dimless * realise(sinc(self.isqrtOk0_pi * distance_dimless))

    @abstractmethod
    def _e2func(self, zp1: _TN) -> _TN: ...

    def e2func(self, z: _TN, zp1: _TN = None) -> _TN:
        return self._e2func(1+z if zp1 is None else zp1)

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


class FLRW(FLRWDriver[_FLRWT, _acCosmologyT], ABC):
    _critical_density_constant: ClassVar[Unit] = 3 / (8 * pi * Newton_G)

    H0: _GQuantity = Parameter(H100)

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
