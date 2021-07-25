from abc import ABC
from itertools import chain
from operator import neg

from ._analytic import BaseAnalyticFLRWDriver, BaseAnalyticLambdaCDMR
from ...special.elliptic_reduction.functional import elliptic_integral
from ...utils._typing import _TN


class AnalyticFLRWDriver(BaseAnalyticFLRWDriver, ABC):
    def lookback_time_dimless(self, z: _TN) -> _TN:
        return self._fix_dimless(elliptic_integral(
            0, z, *chain(map(neg, self._epoly_roots), (1,)),
            h=self._epoly_degree, m=self._epoly_degree * (0,) + (-1,)
        ))

    def age_dimless(self, z: _TN) -> _TN:
        raise NotImplementedError

    def comoving_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
        return self._fix_dimless(elliptic_integral(
            z1, z2, *map(neg, self._epoly_roots),
            h=self._epoly_degree, m=self._epoly_degree * (0,)
        ))


# noinspection PyAbstractClass
class LambdaCDMR(AnalyticFLRWDriver, BaseAnalyticLambdaCDMR):
    pass
