from __future__ import annotations

from functools import cache
from typing import Callable, Iterable

import sympy as sym
import torch
from more_itertools import take
from sympy.utilities.lambdify import implemented_function

from .core import EllipticReduction
from ...utils._typing import _T


class SymbolicEllipticReduction(EllipticReduction):
    elliprc = staticmethod(implemented_function('R_C', EllipticReduction.elliprc))
    elliprd = staticmethod(implemented_function('R_D', EllipticReduction.elliprd))
    elliprf = staticmethod(implemented_function('R_F', EllipticReduction.elliprf))
    elliprj = staticmethod(implemented_function('R_J', EllipticReduction.elliprj))

    @classmethod
    @cache
    def get(cls, n: int, h: int) -> SymbolicEllipticReduction:
        return cls(n=n, h=h)

    def __init__(self, n=4, h=4):
        super().__init__(
            *sym.symbols('x, y'),
            *(tuple(take(n, sym.numbered_symbols(l, start=1))) for l in 'ab'),
            h
        )

    @cache
    def desymbolise(self, expr: sym.Expr) -> Callable[[Iterable[_T], Iterable[_T], tuple[_T, _T]], _T]:
        # TODO: unhack h=3
        a, b = (tuple(self.a)[1:], tuple(self.b)[1:]) if not isinstance(self.a[1], sym.Symbol) else (self.a, self.b)
        return sym.lambdify([a, b, (self.y, self.x)], expr,
                            modules=[{'sqrt': lambda x: x**0.5}, torch])
