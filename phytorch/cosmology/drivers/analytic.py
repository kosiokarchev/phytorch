from abc import ABC, abstractmethod
from itertools import chain
from operator import neg
from typing import Callable, ClassVar, Iterable

from torch import Tensor

from .. import special
from ..core import FLRWDriver
from ...roots import roots
from ...special.elliptic_reduction.symbolic import SymbolicEllipticReduction
from ...utils._typing import _TN


class AnalyticFLRWDriver(FLRWDriver, ABC):
    """Degree of the $E(z) = P(z+1)$ polynomial."""
    _epoly_degree: ClassVar[int]
    _integral_comoving_distance: ClassVar[Callable[[Iterable[_TN], Iterable[_TN], tuple[_TN, _TN]], Tensor]]
    _integral_lookback_time: ClassVar[Callable[[Iterable[_TN], Iterable[_TN], tuple[_TN, _TN]], Tensor]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        rednn = SymbolicEllipticReduction.get(cls._epoly_degree, cls._epoly_degree)
        cls._integral_comoving_distance = staticmethod(rednn.desymbolise(rednn.Ie(0)))
        redn1n = SymbolicEllipticReduction.get(cls._epoly_degree+1, cls._epoly_degree)
        cls._integral_lookback_time = staticmethod(redn1n.desymbolise(redn1n.Ie(-cls._epoly_degree-1)))

    @property
    @abstractmethod
    def _epoly_leading(self) -> _TN:
        """Leading coefficient of the $E(z) = P(z+1)$ polynomial."""

    @property
    @abstractmethod
    def _epoly_coeffs_(self) -> Iterable[_TN]:
        """Unnormalised coefficients of the $E = P(z+1)$ polynomial."""

    @property
    def _epoly_coeffs(self) -> Iterable[_TN]:
        """Normalised coefficients of $P(z+1)$."""
        return (c/self._epoly_leading for c in self._epoly_coeffs_)

    @property
    def _epoly_roots(self) -> Iterable[_TN]:
        """Roots of $E(z) = P(z)$. Note: not of $P(z+1)$!"""
        return (r-1 for r in roots(*self._epoly_coeffs))

    def _fix_dimless(self, val: _TN) -> _TN:
        return val.real / self._epoly_leading**0.5

    def lookback_time_dimless(self, z: _TN) -> _TN:
        return self._fix_dimless(self._integral_lookback_time(
            chain(map(neg, self._epoly_roots), (1,)),
            (self._epoly_degree+1)*(1,),
            (0, z)
        ))

    def age_dimless(self, z: _TN) -> _TN:
        raise NotImplementedError

    def comoving_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN:
        return self._fix_dimless(self._integral_comoving_distance(
            map(neg, self._epoly_roots),
            self._epoly_degree*(1,),
            (z1, z2)
        ))


# noinspection PyAbstractClass
class LambdaCDMR(AnalyticFLRWDriver, special.LambdaCDMR):
    _epoly_degree = 4

    @property
    def _epoly_leading(self) -> _TN:
        return self.Or0

    @property
    def _epoly_coeffs_(self) -> Iterable[_TN]:
        return self.Om0, self.Ok0, 0, self.Ode0
