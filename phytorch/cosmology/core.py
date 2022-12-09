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
_acCosmologyT = TypeVar('_acCosmologyT', bound=_astropy.Cosmology)


class Cosmology(AstropyConvertible[_CosmologyT, _acCosmologyT], Generic[_CosmologyT, _acCosmologyT], ABC):
    """The abstract base class for all cosmologies."""

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
    r"""An abstract class that defines basic *dimensionless* routines.

    Subclasses must implement concrete calculation routines:

    * The dimensionless Hubble parameter :math:`E^2(z) \equiv (H(z) / H_0)` in the
      `_e2func` method.
    * The (dimensionless) comoving distance

      .. math::
         \frac{\chi(z_1, z_2)}{cH_0^{-1}} = \int_{z_1}^{z_2} \frac{\mathrm{d}z}{E(z)}

      either in the `comoving_distance_dimless_z1z2` or `comoving_distance_dimless` method.
    * The (dimensionless) lookback time

      .. math::
         \frac{t(z_1, z_2)}{H_0^{-1}} = \int_{z_1}^{z_2} \frac{\mathrm{d}z}{(z+1) E(z)}

      either in the `lookback_time_dimless` or `age_dimless` method.
    * The (dimensionless) absorption distance

      .. math::
         \frac{d_{\mathrm{abs}}(z_1, z_2)}{cH_0^{-1}} = \int_{z_1}^{z_2} \frac{(z+1)^2}{E(z)} \mathrm{d}z

      in the `absorption_distance_dimless` method.
    """

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
    """Relative effective curvature density at the present time.
    
    See Also
    --------
    :py:attr:`astropy.cosmology.FLRW.Ok0`"""

    Ok = _redshift_method(_redshift_curvature)
    """Relative effective curvature density at the given redshift.
    
    See Also
    --------
    :py:meth:`astropy.cosmology.FLRW.Ok`"""

    @property
    def sqrtOk0(self) -> _TN:
        r""":math:`\sqrt{\Omega_{k0}} \in \mathbb{C}`"""
        return complexify(self.Ok0)**0.5

    @property
    def isqrtOk0_pi(self) -> _TN:
        r""":math:`\mathrm{i} \sqrt{\Omega_{k0}} / \pi \in \mathbb{C}`"""
        return 1j * self.sqrtOk0 / pi

    def transform_curved(self, distance_dimless: _TN) -> _TN:
        r"""Transform (dimensionless) radial to transverse distance using curvature:

        .. math::
           \frac{D_M}{cH_0^{-1}} = \frac{\chi}{cH_0^{-1}} \operatorname{sinc}\left(\mathrm{i} \sqrt{\Omega_{k0}} \frac{\chi}{cH_0^{-1}}\right).

        The result is always well-defined and real since

        .. math::
           \operatorname{sinc}\left(\mathrm{i} \sqrt{\Omega_{k0}} \frac{\chi}{cH_0^{-1}}\right) =
           \begin{cases}
              \operatorname{sinc}\left(\sqrt{\left|\Omega_{k0}\right|} \frac{\chi}{cH_0^{-1}}\right) & \text{for } \Omega_{k0} < 0,
              \\ 1 & \text{for } \Omega_{k0} = 0,
              \\ \operatorname{sinhc}\left(\sqrt{\left|\Omega_{k0}\right|} \frac{\chi}{cH_0^{-1}}\right) & \text{for } \Omega_{k0} > 0,
           \end{cases}

        where :math:`\operatorname{sinhc}(x) \equiv \sinh(x) / x`.

        Notes
        -----
        Note that the |pytorch| definition of `~torch.sinc` is slightly modified: :math:`\operatorname{sinc}(x) \equiv \sin(\pi x) / (\pi x)`.
        """
        return distance_dimless * realise(sinc(self.isqrtOk0_pi * distance_dimless))

    @abstractmethod
    def _e2func(self, zp1: _TN) -> _TN:
        """Function that calculates the dimensionless Hubble parameter for the concrete cosmological model.

        Must be overridden in subclasses. Note that the argument is :math:`z+1`.
        """

    def e2func(self, z: _TN, zp1: _TN = None) -> _TN:
        r"""Dimensionless Hubble parameter squared at the given redshift (:math:`E^2(z) \equiv (H(z) / H_0)^2`)."""
        return self._e2func(1+z if zp1 is None else zp1)

    def efunc(self, z: _TN) -> _TN:
        """Dimensionless Hubble parameter at the given redshift.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.efunc`"""
        return self.e2func(z)**0.5

    def inv_efunc(self, z: _TN) -> _TN:
        """Inverse of the dimensionless Hubble parameter at the given redshift.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.inv_efunc`"""
        return self.e2func(z)**(-0.5)

    def lookback_time_integrand(self, z: _TN) -> _TN:
        r"""Integrand of the (dimensionless) lookback time:

        .. math::
            H_0 \frac{\mathrm{d}t}{\mathrm{d}z} = \frac{1}{(z+1) E(z)}.


        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.lookback_time_integrand`"""
        return self.inv_efunc(z) / (z+1)

    def abs_distance_integrand(self, z: _TN) -> _TN:
        r"""Integrand of the (dimensionless) absorption distance:

        .. math::
            \frac{H_0}{c} \frac{\mathrm{d}d_{\mathrm{abs}}}{\mathrm{d}z} = \frac{(z+1)^2}{E(z)}.


        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.abs_distance_integrand`
        """
        return (z+1)**2 * self.inv_efunc(z)

    # TODO: @abstractmethodgroup
    def lookback_time_dimless(self, z: _TN) -> _TN:
        """Dimensionless `lookback_time` (in units of the `hubble_time`)."""
        return self.age_dimless(0) - self.age_dimless(z)

    # TODO: @abstractmethodgroup
    def age_dimless(self, z: _TN) -> _TN:
        """Dimensionless `age` of the universe (in units of the `hubble_time`)."""
        return self.lookback_time_dimless(inf)

    @abstractmethod
    def absorption_distance_dimless(self, z: _TN) -> _TN:
        """Dimensionless `absorption_distance` (in units of the `hubble_distance`)."""

    # TODO: @abstractmethodgroup
    def comoving_distance_dimless(self, z: _TN) -> _TN:
        """Dimensionless `comoving_distance` (in units of the `hubble_distance`)."""
        return self.comoving_distance_dimless_z1z2(0, z)

    # TODO: @abstractmethodgroup
    def comoving_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
        """Dimensionless `comoving_distance_z1z2` (in units of the `hubble_distance`)."""
        return self.comoving_distance_dimless(z2) - self.comoving_distance_dimless(z1)

    def comoving_transverse_distance_dimless(self, z: _TN) -> _TN:
        """Dimensionless `comoving_transverse_distance` (in units of the `hubble_distance`)."""
        return self.transform_curved(self.comoving_distance_dimless(z))

    def comoving_transverse_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
        """Dimensionless `comoving_transverse_distance_z1z2` (in units of the `hubble_distance`)."""
        return self.transform_curved(self.comoving_distance_dimless_z1z2(z1, z2))

    def angular_diameter_distance_dimless(self, z: _TN) -> _TN:
        """Dimensionless `angular_diameter_distance` (in units of the `hubble_distance`)."""
        return self.comoving_transverse_distance_dimless(z) / (z+1)

    def angular_diameter_distance_dimless_z1z2(self, z1: _TN, z2: _TN):
        """Dimensionless `angular_diameter_distance_z1z2` (in units of the `hubble_distance`)."""
        return self.comoving_transverse_distance_dimless_z1z2(z1, z2) / (z2+1)

    def luminosity_distance_dimless(self, z: _TN) -> _TN:
        """Dimensionless `luminosity_distance` (in units of the `hubble_distance`)."""
        return self.comoving_transverse_distance_dimless(z) * (z+1)

    def differential_comoving_volume_dimless(self, z: _TN) -> _TN:
        """Dimensionless `differential_comoving_volume` (in units of the `hubble_volume` per `steradian`)."""
        return self.comoving_transverse_distance_dimless(z)**2 * self.inv_efunc(z)

    def comoving_volume_dimless(self, z: _TN, eps=1e-8) -> _TN:
        """Dimensionless `comoving_volume` (in units of the `hubble_volume`)."""
        dc = self.comoving_distance_dimless(z)
        return 8*pi*dc**3 * realise(csinc(2 * self.isqrtOk0_pi * dc, eps))


class FLRW(FLRWDriver[_FLRWT, _acCosmologyT], ABC):
    """The abstract base class for FLRW cosmologies.

    Implements *dimensionfull* calculations on top of the *dimensionless*
    routines from `FLRWDriver`.
    """

    _critical_density_constant: ClassVar[Unit] = 3 / (8 * pi * Newton_G)

    H0: _GQuantity = Parameter(H100)
    r"""The Hubble constant :math:`H_0 \equiv H(z=0)`.
    
    See Also
    --------
    :py:attr:`astropy.cosmology.FLRW.H0`."""

    def H(self, z: _TN) -> _GQuantity:
        """Value of the Hubble parameter :math:`H(z)` at the given redshift.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.H`"""
        return self.H0 * self.efunc(z)

    @property
    def hubble_time(self) -> _GQuantity:
        r"""Hubble time ("age of the universe"):
        :math:`H_0^{-1}`.

        See Also
        --------
        :py:attr:`astropy.cosmology.FLRW.hubble_time`"""
        return 1 / self.H0

    @property
    def hubble_distance(self) -> _GQuantity:
        r"""Hubble distance ("size of the universe"):
        :math:`cH_0^{-1}`.

        See Also
        --------
        :py:attr:`astropy.cosmology.FLRW.hubble_distance`"""
        return speed_of_light / self.H0

    @property
    def hubble_distance_in_10pc(self) -> _TN:
        """Hubble distance in units of 10pc (unitless)."""
        return self.hubble_distance.to(10*pc).value

    @property
    def hubble_volume(self) -> _GQuantity:
        """Hubble distance cubed."""
        return self.hubble_distance**3

    @property
    def critical_density0(self) -> _GQuantity:
        r"""Critical density at the present time:

        .. math::
           \rho_{\mathrm{cr}}(z=0) = \frac{3}{8\pi} \frac{H_0^2}{G}.

        See Also
        --------
        :py:attr:`astropy.cosmology.FLRW.critical_density0`"""
        return self._critical_density_constant * self.H0**2

    def critical_density(self, z: _TN) -> _GQuantity:
        r"""Critical density at the given redshift:

        .. math::
           \rho_{\mathrm{cr}}(z) = \frac{3}{8\pi} \frac{H(z)^2}{G}.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.critical_density`"""
        return self.critical_density0 * self.e2func(z)

    def lookback_time(self, z: _TN) -> _GQuantity:
        r"""Lookback time to the given redshift:

        .. math::
           t(z) = \frac{1}{H_0} \int_{0}^{z} \frac{\mathrm{d}z}{(z+1)E(z)}.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.lookback_time`
        """
        return self.hubble_time * self.lookback_time_dimless(z)

    def lookback_distance(self, z: _TN) -> _GQuantity:
        """Lookback distance (length that light traverses during `lookback_time`) to the given redshift.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.lookback_distance`"""
        return self.hubble_distance * self.lookback_time_dimless(z)

    def age(self, z: _TN) -> _GQuantity:
        r"""Age of the universe at the given redshift:

        .. math::
           t(\infty) - t(z) = \frac{1}{H_0} \int_{z}^{\infty} \frac{\mathrm{d}z}{(z+1)E(z)}.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.age`"""
        return self.hubble_time * self.age_dimless(z)

    # For compatibility with astropy, the absorption_distance is dimensionless
    def absorption_distance(self, z: _TN) -> _TN:
        r"""Absorption distance to the given redshift:

        .. math::
           d_{\mathrm{abs}}(z) = \frac{c}{H_0} \int_{0}^{z} \frac{(z+1)^2}{E(z)} \mathrm{d}z.


        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.absorption_distance`

        Notes
        -----
        For compatibility with |astropy|, the `absorption_distance` is returned
        as dimensionless instead of a quantity and is thus equivalent to
        `absorption_distance_dimless`."""
        return self.absorption_distance_dimless(z)

    def comoving_distance(self, z: _TN) -> _GQuantity:
        r"""Comoving distance to the given redshift:

        .. math::
           \chi(z) = \frac{c}{H_0} \int_{0}^{z} \frac{\mathrm{d}z}{E(z)}.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.comoving_distance`"""
        return self.hubble_distance * self.comoving_distance_dimless(z)

    def comoving_distance_z1z2(self, z1: _TN, z2: _TN) -> _GQuantity:
        r"""Comoving distance between two given redshifts:

        .. math::
           \chi(z_1, z_2) = \frac{c}{H_0} \int_{z_1}^{z_2} \frac{\mathrm{d}z}{E(z)}.
        """
        return self.hubble_distance * self.comoving_distance_dimless_z1z2(z1, z2)

    def comoving_transverse_distance(self, z: _TN) -> _GQuantity:
        """Comoving transverse distance to the given redshift.

        Refer to `transform_curved`.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.comoving_transverse_distance`"""
        return self.hubble_distance * self.comoving_transverse_distance_dimless(z)

    def comoving_transverse_distance_z1z2(self, z1: _TN, z2: _TN) -> _GQuantity:
        """Comoving transverse distance between two given redshifts.

        Refer to `transform_curved`.
        """
        return self.hubble_distance * self.comoving_transverse_distance_dimless_z1z2(z1, z2)

    def angular_diameter_distance(self, z: _TN) -> _GQuantity:
        r"""Angular diameter / size distance to the given redshift:

        .. math::
           D_A(z) = \frac{D_M(z)}{z+1}.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.angular_diameter_distance`"""
        return self.hubble_distance * self.angular_diameter_distance_dimless(z)

    def angular_diameter_distance_z1z2(self, z1: _TN, z2: _TN) -> _GQuantity:
        r"""Angular diameter / size distance between two given redshifts:

        .. math::
           D_A(z_1, z_2) = \frac{D_M(z_1, z_2)}{z_2+1}.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.angular_diameter_distance_z1z2`"""
        return self.hubble_distance * self.angular_diameter_distance_dimless_z1z2(z1, z2)

    def luminosity_distance(self, z: _TN) -> _GQuantity:
        r"""Luminosity distance to the given redshift:

        .. math::
           D_L(z) = (z+1)^2 D_A(z) = (z+1) D_M(z).

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.luminosity_distance`"""
        return self.hubble_distance * self.luminosity_distance_dimless(z)

    def differential_comoving_volume(self, z: _TN) -> _GQuantity:
        r"""Differential comoving volume at the given redshift:

        .. math::
           \frac{\mathrm{d}V(z)}{\mathrm{d}\Omega \mathrm{d}z} = \frac{cH_0^{-1}}{E(z)} D_M^2(z),

        where :math:`\mathrm{d}\Omega` is an infinitesimal solid angle.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.differential_comoving_volume`"""
        return self.hubble_volume / steradian * self.differential_comoving_volume_dimless(z)

    def comoving_volume(self, z: _TN) -> _GQuantity:
        r"""Comoving spherical volume to the given redshift:

        .. math::
           V(z) = 8\pi \chi^3 \operatorname{csinc}\left(2 \mathrm{i} \sqrt{\Omega_{k0}} \frac{\chi}{cH_0^{-1}}\right),

        where :math:`\operatorname{csinc(x)} \equiv (1 - \operatorname{sinc}(x)) / x^2`
        (with the limit :math:`\operatorname{csinc(x)} \rightarrow 1/6` as :math:`x \rightarrow 0`),
        resulting from the integration of `differential_comoving_volume`.

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.comoving_volume`"""
        return self.hubble_volume * self.comoving_volume_dimless(z)

    def distmod(self, z: _TN) -> _TN:
        r"""Distance modulus in magnitudes (unitless):

        .. math::
           \mu(z) = 2.5 \log_{10}\left[\left(D_L / 10\,\mathrm{pc}\right)^2\right].

        Note that this formulation allows negative luminosity distances 🙃

        See Also
        --------
        :py:meth:`astropy.cosmology.FLRW.distmod`"""
        # TODO: return as magnitude quantity?
        # square inside logarithm allows negative distances (:
        return 2.5 * log10((self.luminosity_distance_dimless(z) * self.hubble_distance_in_10pc)**2)
