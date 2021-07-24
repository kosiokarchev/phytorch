from functools import partial, reduce
from itertools import combinations
from operator import mul
from typing import ItemsView, Iterable

import sympy.utilities.iterables

from ._typing import _T


product = partial(reduce, mul)


def sign(a):
    return a // abs(a)


def elementary_symmetric(p: int, a: Iterable[_T]) -> _T:
    return sum(map(product, combinations(a, p)))


def partitions(s: int) -> Iterable[ItemsView[int, int]]:
    return map(dict.items, sympy.utilities.iterables.partitions(s))
