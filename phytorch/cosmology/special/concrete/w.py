from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, TypeVar

from ..decdm import BaseDECDM, RadiationFLRWMixin
from ...core import _acCosmologyT
from ...utils import Parameter, PropertyParameter
from ....math import exp
from ....utils._typing import _TN


__all__ = (
    'wCDM', 'w0waCDM', 'wpwaCDM', 'w0wzCDM',
    'wCDMR', 'w0waCDMR', 'wpwaCDMR', 'w0wzCDMR'
)

from ....utils.interop import _astropy, BaseToAstropy


_BasewCDMT = TypeVar('_BasewCDMT', bound='BasewCDM')
_wCDMT = TypeVar('_wCDMT', bound='wCDM')
_w0waCDMT = TypeVar('_w0waCDMT', bound='w0waCDM')
_wpwaCDMT = TypeVar('_wpwaCDMT', bound='wpwaCDM')
_w0wzCDMT = TypeVar('_w0wzCDMT', bound='w0wzCDM')

_acwCDMT = TypeVar('_acwCDMT', bound=_astropy.cosmology.wCDM)
_acw0waCDMT = TypeVar('_acw0waCDMT', bound=_astropy.cosmology.w0waCDM)
_acwpwaCDMT = TypeVar('_acwpwaCDMT', bound=_astropy.cosmology.wpwaCDM)
_acw0wzCDMT = TypeVar('_acw0wzCDMT', bound=_astropy.cosmology.w0wzCDM)


class BasewCDM(BaseDECDM[_BasewCDMT, _acCosmologyT], ABC):
    @abstractmethod
    def _w(self, zp1: _TN) -> _TN: ...

    def w(self, z: _TN, zp1: _TN = None) -> _TN:
        return self._w(1.+z if zp1 is None else zp1)


_T = TypeVar('_T')
_aT = TypeVar('_aT')


class wCDM(BasewCDM[_wCDMT, _acwCDMT], ABC):
    w0: _TN = Parameter(-1.)

    def _w(self, zp1: _TN) -> _TN:
        return 1. + self.w0

    def _de_density_scale(self, zp1: _TN):
        return zp1 ** (3. * (1. + self.w0))


    class _toAstropy(BaseDECDM._toAstropy[_wCDMT, _acwCDMT]):
        _cls = _astropy.cosmology.wCDM


class wCDMR(RadiationFLRWMixin, wCDM, ABC):
    pass


class w0waCDM(wCDM[_w0waCDMT, _acw0waCDMT], ABC):
    wa: _TN = Parameter(0.)

    def _w(self, zp1: _TN) -> _TN:
        return self.w0 + self.wa * (1. - 1./zp1)

    def _de_density_scale(self, zp1: _TN):
        return zp1 ** (3. * (1. + self.w0 + self.wa)) * exp(-3. * self.wa * (1. - 1./zp1))


    class _toAstropy(wCDM._toAstropy[_w0waCDMT, _acw0waCDMT]):
        _cls = _astropy.cosmology.w0waCDM


class w0waCDMR(RadiationFLRWMixin, w0waCDM, ABC):
    pass


class wpwaCDM(w0waCDM[_wpwaCDMT, _acwpwaCDMT], ABC):
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
    pass


class w0wzCDM(wCDM[_w0wzCDMT, _acw0wzCDMT], ABC):
    wz: _TN = Parameter(0.)

    def _w(self, zp1: _TN) -> _TN:
        return self.w0 - self.wz + self.wz * zp1

    def _de_density_scale(self, zp1: _TN):
        return zp1 ** (3. * (1. + self.w0 - self.wz)) * exp(-3. * self.wz * zp1 - 3.*self.wz)


    class _toAstropy(wCDM._toAstropy[_w0wzCDMT, _acw0wzCDMT]):
        _cls = _astropy.cosmology.w0wzCDM


class w0wzCDMR(RadiationFLRWMixin, w0wzCDM, ABC):
    pass
