from decimal import Decimal
from numbers import Number, Rational
from typing import Any, Callable, TypeVar, Union

from torch import Tensor


_TN = TypeVar('_TN', Tensor, Number)
_t = Union[Tensor, Number]
_T = TypeVar('_T')
_bop = Callable[[_T, _T], _T]
_mop = Callable[..., _T]
_fractionable = Union[Rational, int, float, Decimal, str]


class ValueProtocol:
    value: Union[Number, Any]
