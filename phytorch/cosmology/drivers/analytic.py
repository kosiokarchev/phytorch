from abc import ABC
from itertools import chain
from operator import neg
from typing import Callable, ClassVar, Iterable

from torch import Tensor

from ._analytic import BaseAnalyticFLRWDriver, BaseAnalyticLambdaCDM, BaseAnalyticLambdaCDMR
from .. import special
from ...special.elliptic_reduction.symbolic import SymbolicEllipticReduction
from ...utils._typing import _TN


class AnalyticFLRWDriver(BaseAnalyticFLRWDriver, ABC):
    _integral_comoving_distance: ClassVar[Callable[[Iterable[_TN], Iterable[_TN], tuple[_TN, _TN]], Tensor]]
    _integral_lookback_time: ClassVar[Callable[[Iterable[_TN], Iterable[_TN], tuple[_TN, _TN]], Tensor]]
    _integral_absorption_distance: ClassVar[Callable[[Iterable[_TN], Iterable[_TN], tuple[_TN, _TN]], Tensor]]

    # TODO: do we really need __init_subclass__?!
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        rednn = SymbolicEllipticReduction.get(cls._epoly_degree, cls._epoly_degree)
        cls._integral_comoving_distance = staticmethod(rednn.desymbolise(rednn.Ie(0)))
        redn1n = SymbolicEllipticReduction.get(cls._epoly_degree+1, cls._epoly_degree)
        # TODO: unhack h=3
        #    would have redn1n.Ie(-cls._epoly_degree-1) instead of h
        cls._integral_lookback_time = staticmethod(redn1n.desymbolise(redn1n.Ie(-redn1n.h-1)))
        if cls._epoly_degree != 3:
            cls._integral_absorption_distance = staticmethod(redn1n.desymbolise(redn1n.Im(
                cls._epoly_degree*(0,) + (2,)
            )))

    def lookback_time_dimless(self, z: _TN) -> _TN:
        return self._fix_dimless(self._integral_lookback_time(
            chain(map(neg, self._epoly_roots), (1,)),
            (self._epoly_degree+1)*(1,),
            (0, z)
        ))

    def age_dimless(self, z: _TN) -> _TN:
        raise NotImplementedError

    def absorption_distance_dimless(self, z: _TN) -> _TN:
        return self._fix_dimless(self._integral_absorption_distance(
            chain(map(neg, self._epoly_roots), (1,)),
            (self._epoly_degree+1)*(1,),
            (0, z)
        ))

    def comoving_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
        return self._fix_dimless(self._integral_comoving_distance(
            map(neg, self._epoly_roots),
            self._epoly_degree*(1,),
            (z1, z2)
        ))


class LambdaCDM(AnalyticFLRWDriver, BaseAnalyticLambdaCDM):
    # TODO: unhack h=3
    def absorption_distance_dimless(self, z: _TN) -> _TN:
        raise NotImplementedError


class LambdaCDMR(AnalyticFLRWDriver, BaseAnalyticLambdaCDMR):
    pass


class FlatLambdaCDM(LambdaCDM, special.FlatLambdaCDM):
    pass


class FlatLambdaCDMR(LambdaCDMR, special.FlatLambdaCDMR):
    pass
