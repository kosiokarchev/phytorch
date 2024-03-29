from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from .decdm import BaseDECDM, RadiationFLRWMixin
from ..core import _acCosmologyT, FLRW
from ..utils import Parameter, PropertyParameter
from ...math import exp
from ...utils._typing import _TN
from ...utils.interop import _astropy


__all__ = (
    'wCDM', 'LambdaCDM', 'w0waCDM', 'wpwaCDM', 'w0wzCDM',
    'wCDMR', 'LambdaCDMR', 'w0waCDMR', 'wpwaCDMR', 'w0wzCDMR'
)


_BasewCDMT = TypeVar('_BasewCDMT', bound='BasewCDM')

_LambdaCDMT = TypeVar('_LambdaCDMT', bound='LambdaCDM')
_wCDMT = TypeVar('_wCDMT', bound='wCDM')
_w0waCDMT = TypeVar('_w0waCDMT', bound='w0waCDM')
_wpwaCDMT = TypeVar('_wpwaCDMT', bound='wpwaCDM')
_w0wzCDMT = TypeVar('_w0wzCDMT', bound='w0wzCDM')

_acLambdaCDMT = TypeVar('_acLambdaCDMT', bound=_astropy.LambdaCDM)
_acwCDMT = TypeVar('_acwCDMT', bound=_astropy.wCDM)
_acw0waCDMT = TypeVar('_acw0waCDMT', bound=_astropy.w0waCDM)
_acwpwaCDMT = TypeVar('_acwpwaCDMT', bound=_astropy.wpwaCDM)
_acw0wzCDMT = TypeVar('_acw0wzCDMT', bound=_astropy.w0wzCDM)


class BasewCDM(BaseDECDM[_BasewCDMT, _acCosmologyT], ABC):
    @abstractmethod
    def _w(self, zp1: _TN) -> _TN: ...

    def w(self, z: _TN, zp1: _TN = None) -> _TN:
        return self._w(1.+z if zp1 is None else zp1)


class wCDM(BasewCDM[_wCDMT, _acwCDMT], ABC):
    r"""Cosmology with a dark energy with constant equation of state:

    .. math::
       w(z) = w_0 = \mathrm{const}.

    See Also
    --------
    :py:class:`astropy.cosmology.wCDM`"""

    w0: _TN = Parameter(-1.)

    def _w(self, zp1: _TN) -> _TN:
        return self.w0

    def _de_density_scale(self, zp1: _TN):
        return zp1 ** (3. * (1. + self.w0))


    class _toAstropy(BaseDECDM._toAstropy[_wCDMT, _acwCDMT]):
        _cls = _astropy.cosmology.wCDM


class wCDMR(RadiationFLRWMixin, wCDM, ABC):
    """`wCDM` with radiation."""


class LambdaCDM(wCDM[_LambdaCDMT, _astropy.cosmology.LambdaCDM], ABC):
    r"""Dark energy from a cosmological constant :math:`\Lambda`, implying an equation of state

    .. math::
       w(z) = w_0 = -1.

    See Also
    --------
    :py:class:`astropy.cosmology.LambdaCDM`"""
    w0 = -1

    Ode = FLRW._redshift_method(BaseDECDM._redshift_constant)

    def _de_density_scale(self, zp1: _TN):
        return 1.

    def _e2func(self, zp1: _TN) -> _TN:
        return zp1**2 * (self.Om0 * zp1 + self.Ok0) + self.Ode0

    class _toAstropy(FLRW._toAstropy[_LambdaCDMT, _astropy.cosmology.LambdaCDM]):
        _cls = _astropy.cosmology.LambdaCDM


class LambdaCDMR(RadiationFLRWMixin, LambdaCDM, ABC):
    """`LambdaCDM` with radiation."""


class w0waCDM(wCDM[_w0waCDMT, _acw0waCDMT], ABC):
    r"""Cosmology with a dark energy evolving linearly with scale factor:

    .. math::
       w(z) = w_0 + w_a (1-a) = w_0 + w_a \left(1 - \frac{1}{z+1}\right).

    See Also
    --------
    :py:class:`astropy.cosmology.w0waCDM`"""
    wa: _TN = Parameter(0.)

    def _w(self, zp1: _TN) -> _TN:
        return self.w0 + self.wa * (1. - 1./zp1)

    def _de_density_scale(self, zp1: _TN):
        return zp1 ** (3. * (1. + self.w0 + self.wa)) * exp(-3. * self.wa * (1. - 1./zp1))


    class _toAstropy(wCDM._toAstropy[_w0waCDMT, _acw0waCDMT]):
        _cls = _astropy.cosmology.w0waCDM


class w0waCDMR(RadiationFLRWMixin, w0waCDM, ABC):
    """`w0waCDM` with radiation."""


class wpwaCDM(w0waCDM[_wpwaCDMT, _acwpwaCDMT], ABC):
    r"""Cosmology with a pivoting dark energy equation of state:

    .. math::
       w(z) = w_p + w_a (a_p-a) = w_p + w_a \left(\frac{1}{z_p+1} - \frac{1}{z+1}\right).

    See Also
    --------
    :py:class:`astropy.cosmology.wpwaCDM`"""
    @PropertyParameter
    def wp(self):
        return self.w0

    @wp.setter
    def wp(self, value):
        self.w0 = value

    zp: _TN = Parameter(0.)

    @property
    def ap(self):
        return 1. / (1. + self.zp)

    def _w(self, zp1: _TN) -> _TN:
        return self.w0 + self.wa * (self.ap - 1./zp1)

    def _de_density_scale(self, zp1: _TN):
        return zp1 ** (3. * (1. + self.w0 + self.ap * self.wa)) * exp(-3. * self.wa * (1. - 1./zp1))


    class _toAstropy(w0waCDM._toAstropy[_w0waCDMT, _acw0waCDMT]):
        _cls = _astropy.cosmology.wpwaCDM


class wpwaCDMR(RadiationFLRWMixin, wpwaCDM, ABC):
    """`wpwaCDM` with radiation."""


class w0wzCDM(wCDM[_w0wzCDMT, _acw0wzCDMT], ABC):
    r"""Cosmology with a dark energy evolving linearly with redshift:

    .. math::
       w(z) = w_0 + w_z z.

    See Also
    --------
    :py:class:`astropy.cosmology.w0wzCDM`"""
    wz: _TN = Parameter(0.)

    def _w(self, zp1: _TN) -> _TN:
        return self.w0 - self.wz + self.wz * zp1

    def _de_density_scale(self, zp1: _TN):
        return zp1 ** (3. * (1. + self.w0 - self.wz)) * exp(-3. * self.wz * zp1 - 3.*self.wz)


    class _toAstropy(wCDM._toAstropy[_w0wzCDMT, _acw0wzCDMT]):
        _cls = _astropy.cosmology.w0wzCDM


class w0wzCDMR(RadiationFLRWMixin, w0wzCDM, ABC):
    """`w0wzCDM` with radiation."""
