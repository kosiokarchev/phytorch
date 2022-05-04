from __future__ import annotations

from abc import ABC
from typing import TypeVar

from ..decdm import BaseDECDM, RadiationFLRWMixin
from ...core import FLRW
from ....utils._typing import _TN
from ....utils.interop import _astropy


__all__ = 'LambdaCDM', 'LambdaCDMR'



_LambdaCDMT = TypeVar('_LambdaCDMT', bound='LambdaCDM')
_LambdaCDMRT = TypeVar('_LambdaCDMRT', bound='LambdaCDMR')


class LambdaCDM(BaseDECDM[_LambdaCDMT, _astropy.cosmology.LambdaCDM], ABC):
    Ode = FLRW._redshift_method(BaseDECDM._redshift_constant)

    def _de_density_scale(self, zp1: _TN):
        return 1.

    def _e2func(self, zp1: _TN) -> _TN:
        return zp1**2 * (self.Om0 * zp1 + self.Ok0) + self.Ode0

    class _toAstropy(FLRW._toAstropy[_LambdaCDMT, _astropy.cosmology.LambdaCDM]):
        _cls = _astropy.cosmology.LambdaCDM


class LambdaCDMR(RadiationFLRWMixin, LambdaCDM[_LambdaCDMRT], ABC):
    def _e2func(self, zp1: _TN) -> _TN:
        return zp1**2 * ((self.Or0 * zp1 + self.Om0) * zp1 + self.Ok0) + self.Ode0
