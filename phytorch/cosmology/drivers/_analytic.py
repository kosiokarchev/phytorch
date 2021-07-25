from abc import ABC, abstractmethod
from typing import ClassVar, Iterable

from .. import special
from ..core import FLRWDriver
from ...roots import roots
from ...utils._typing import _TN


class BaseAnalyticFLRWDriver(FLRWDriver, ABC):
    """Degree of the $E(z) = P(z+1)$ polynomial."""
    _epoly_degree: ClassVar[int]

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

    _roots_force_numeric = False

    @property
    def _epoly_roots(self) -> Iterable[_TN]:
        """Roots of $E(z) = P(z)$. Note: not of $P(z+1)$!"""
        return (r-1 for r in roots(*self._epoly_coeffs, force_numeric=self._roots_force_numeric))

    def _fix_dimless(self, val: _TN) -> _TN:
        return val.real / self._epoly_leading**0.5


class BaseAnalyticLambdaCDMR(BaseAnalyticFLRWDriver, special.LambdaCDMR, ABC):
    _epoly_degree = 4

    @property
    def _epoly_leading(self) -> _TN:
        return self.Or0

    @property
    def _epoly_coeffs_(self) -> Iterable[_TN]:
        return self.Om0, self.Ok0, 0, self.Ode0
