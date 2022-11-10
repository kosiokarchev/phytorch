from __future__ import annotations

from abc import ABC
from math import pi
from typing import MutableMapping, Type

from . import concrete
from ..core import FLRW
from ..utils import PropertyParameter
from ...math import asinh
from ...utils._typing import _TN


class FlatFLRWMixin(FLRW, ABC):
    Ok0 = 0.

    def transform_curved(self, distance_dimless: _TN) -> _TN:
        return distance_dimless

    def comoving_volume_dimless(self, z: _TN, eps=1e-8) -> _TN:
        return 4*pi/3 * self.comoving_transverse_distance_dimless(z)**3


_clsdicts: MutableMapping[Type[FLRW], dict] = {}


def _overrides(base: Type[FLRW]):
    def f(cls: type):
        _clsdicts[base] = dict(cls.__dict__)
        return cls
    return f


@_overrides(concrete.LambdaCDM)
class _FlatLambdaCDM(FlatFLRWMixin, concrete.LambdaCDM, ABC):
    Ode0 = PropertyParameter(concrete.LambdaCDM.Ode0.fget)

    @Ode0.setter
    def Ode0(self, value: _TN):
        self.Om0 = 1. - value

    def age_dimless(self, z: _TN) -> _TN:
        return (2/3) / (1-self.Om0)**0.5 * asinh(((1/self.Om0 - 1) / (1+z)**3)**0.5)


@_overrides(concrete.LambdaCDMR)
class _FlatLambdaCDMR(FlatFLRWMixin, concrete.LambdaCDMR, ABC):
    Ode0 = PropertyParameter(concrete.LambdaCDMR.Ode0.fget)

    @Ode0.setter
    def Ode0(self, value: _TN):
        self.Om0 = 1. - self.Or0 - value

    # TODO: optimise
    # def comoving_distance_dimless(self, z: _t) -> _t: ...
    # def lookback_time_dimless(self, z: _t) -> _t: ...
    # def age_dimless(self, z: _t) -> _t: ...


# TODO: lazify flat classes?
globals().update(_clss := {
    name: type(name, (FlatFLRWMixin, obj, ABC), _clsdicts.get(obj, {}))
    for key in dir(concrete) for obj in [getattr(concrete, key)]
    if isinstance(obj, type) and issubclass(obj, FLRW)
    for name in ['Flat'+key]
})
__all__ = *_clss.keys(),
