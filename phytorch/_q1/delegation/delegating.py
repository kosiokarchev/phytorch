from __future__ import annotations

from abc import ABC
from contextlib import nullcontext
from typing import Generic, get_args, TypeVar


_t = TypeVar('_t')


class Delegating(Generic[_t], ABC):
    _generic_class_index = 0

    @classmethod
    @property
    def _T(cls: Delegating[_t]) -> _t:
        return get_args(cls.__orig_bases__[cls._generic_class_index])[0]

    _is_abstract_delegating = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._is_abstract_delegating = isinstance(cls._T, TypeVar)

    @property
    def delegator_context(self):
        return nullcontext()
