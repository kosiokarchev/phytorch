from abc import ABC, abstractmethod
from typing import Collection, Iterable, overload, Union

from .core import FLRW
from ..math import exp
from ..utils._typing import _TN


class FLRWComponent(ABC):
    O0: _TN

    @abstractmethod
    def redshift(self, z: _TN) -> _TN: ...


class PowerlawComponent(FLRWComponent, ABC):
    _power: _TN

    def redshift(self, z: _TN) -> _TN:
        return (z+1)**self._power


class RadiationComponent(PowerlawComponent):
    _power = 4


class MatterComponent(PowerlawComponent):
    _power = 3


class w0Component(PowerlawComponent):
    w0: _TN

    @property
    def _power(self):
        return 3 * (1 + self.w0)


class w0waComponent(w0Component):
    wa: _TN

    @property
    def _power(self):
        return super()._power + 3*self.wa

    def redshift(self, z: _TN) -> _TN:
        return super().redshift(z) * exp(-3 * self.wa * z/(z+1))


class LambdaComponent(w0Component):
    w0 = -1
    _power = 0

    def redshift(self, z: _TN) -> _TN:
        return 1


class ComponentwiseFLRW(FLRW):
    components: Collection[FLRWComponent]

    def e2func(self, z: _TN) -> _TN:
        return sum(c.O0*c.redshift(z) for c in self.components)

    @overload
    def Omega(self, z: _TN, comp: FLRWComponent) -> _TN: ...

    @overload
    def Omega(self, z: _TN, *comps: FLRWComponent) -> Iterable[_TN]: ...

    def Omega(self, z: _TN, *comps):
        ie2 = 1 / self.e2func(z)
        ret = (c.redshift(z) * ie2 for c in comps)
        return next(ret) if len(comps) > 1 else ret
