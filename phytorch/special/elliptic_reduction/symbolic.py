from __future__ import annotations

from functools import cache
from typing import Callable, Sequence

import sympy as sym
import torch
from more_itertools import take
from sympy.utilities.lambdify import implemented_function

from .core import EllipticReduction
from ...utils._typing import _T


class SymbolicEllipticReduction(EllipticReduction):
    elliprc = implemented_function('R_C', EllipticReduction.elliprc)
    elliprd = implemented_function('R_D', EllipticReduction.elliprd)
    elliprf = implemented_function('R_F', EllipticReduction.elliprf)
    elliprj = implemented_function('R_J', EllipticReduction.elliprj)

    @classmethod
    @cache
    def get(cls, n, h):
        return cls(n, h)

    def __init__(self, n=4, h=4):
        super().__init__(
            *sym.symbols('x, y'),
            *(tuple(take(n, sym.numbered_symbols(l, start=1))) for l in 'ab'),
            h
        )

    @cache
    def Ie(self, i: int) -> Callable[[Sequence[_T], Sequence[_T], tuple[_T, _T]], _T]:
        return sym.lambdify([self.a, self.b, (self.y, self.x)], super().Ie(i),
                            modules=[{'sqrt': lambda x: x**0.5}, torch])
