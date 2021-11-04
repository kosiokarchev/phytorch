from __future__ import annotations

import copy
from decimal import Decimal
from numbers import Number, Rational
from typing import Any, Callable, Protocol, runtime_checkable, Type, TypeVar, Union

from torch import Tensor
from typing_extensions import TypeAlias


_TN = TypeVar('_TN', Tensor, Number)
_t: TypeAlias = Union[Tensor, Number]
_T = TypeVar('_T')
_bop: TypeAlias = Callable[[_T, _T], _T]
_mop: TypeAlias = Callable[..., _T]
_fractionable: TypeAlias = Union[Rational, int, float, Decimal, str]


@runtime_checkable
class ValueProtocol(Protocol):
    value: Union[Number, Any]


def upcast(obj: _T, typ: Type[_T]):
    ret = copy.copy(obj)
    ret.__class__ = typ
    ret.__dict__ = obj.__dict__
    return ret
