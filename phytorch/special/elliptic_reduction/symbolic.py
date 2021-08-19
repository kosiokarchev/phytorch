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

    def __init__(self, n=4, h=4, canonic=True):
        if canonic:
            x, y = sym.symbols('x, y')
            a = tuple(take(n, sym.numbered_symbols('a', start=1)))
        else:
            x, y = sym.symbols('z_2, z_1', real=True)
            a = tuple(-r for r in take(n, sym.numbered_symbols('r', start=1, real=True)))
        super().__init__(x, y, a, h)

    @cache
    def desymbolise(self, expr: sym.Expr) -> Callable[[Iterable[_T], Iterable[_T], tuple[_T, _T]], _T]:
        return sym.lambdify([self.a, (self.y, self.x)], expr,
                            modules=[{'sqrt': lambda x: x**0.5}, torch])
