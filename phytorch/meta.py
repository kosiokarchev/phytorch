from __future__ import annotations

from abc import ABC
from typing import Annotated, get_args, get_origin, get_type_hints, TypeVar

from more_itertools import consume


_T = TypeVar('_T')


class Meta(ABC):
    _meta_annotation = object()

    @classmethod
    def meta_attribute(cls, obj: _T) -> _T:
        return Annotated[obj, cls._meta_annotation]

    _meta_attributes: set[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._meta_attributes = {
            key for key, val in get_type_hints(cls, include_extras=True).items()
            if get_origin(val) is Annotated and get_args(val)[1] is cls._meta_annotation}

    def __new__(cls, *args, **kwargs):
        meta_kwargs = {key: kwargs.pop(key) for key in cls._meta_attributes if key in kwargs}
        self = super().__new__(cls, *args, **kwargs)
        return self._meta_update(self, **meta_kwargs)

    @staticmethod
    def _meta_update(other: Meta, /, **kwargs):
        consume(setattr(other, key, kwargs[key]) for key in other._meta_attributes if key in kwargs)
        return other
