from __future__ import annotations

from abc import ABC
from typing import Generic, Type, TypeVar


_T = TypeVar('_T')
_aT = TypeVar('_aT')


class BaseInterop(Generic[_T, _aT], ABC):
    def __init__(self, _: _T):
        self._ = _

    _cls: Type[_aT]
    def __call__(self) -> _aT: ...


class BaseToAstropy(BaseInterop[_T, _aT]):
    pass


class AstropyConvertible(Generic[_T, _aT], ABC):
    _toAstropy: Type[BaseToAstropy[_T, _aT]]

    @property
    def toAstropy(self) -> BaseInterop[_T, _aT]:
        return self._toAstropy(self)
