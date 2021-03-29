import inspect
from _warnings import warn
from collections import UserDict
from contextlib import AbstractContextManager
from functools import partial
from itertools import chain, starmap
from typing import Any, Iterable, Mapping, MutableMapping, Tuple, TypeVar, Union

from more_itertools import collapse, consume, first, unique_everseen

from .Unit import Unit


_T = TypeVar('_T')
_informat = Union[str, Tuple[Union[Iterable[str], str], Union[Iterable[str], str]]]
_outformat = Tuple[Tuple[str, ...], Tuple[str, ...]]


def names_and_abbrevs(item: _informat) -> _outformat:
    if isinstance(item, str):
        item = (item, item[0])
    elif len(item) == 1:
        item = 2*item
    item = *map(collapse, item),
    return (*map(str.lower, item[0]),), tuple(item[1])


class OverwrittenWarning(Warning):
    pass


class AbstractScope(UserDict):
    def __init__(self, context: MutableMapping[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.context = context if context is not None else inspect.stack()[1].frame.f_locals

class GlobalScope(AbstractScope, AbstractContextManager):
    active = True

    def register(self, name, val, ignore_if_exists=False):
        if name in self.data:
            if ignore_if_exists:
                return
            warn(f'"{name}" already defined in "{self.context}"', OverwrittenWarning)
        self.data[name] = val
        return self

    def register_many(self, ignore_if_exists=False, **kwargs):
        consume(starmap(partial(self.register, ignore_if_exists=ignore_if_exists), kwargs.items()))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()
        self.active = False

    def _del(self):
        self.context.clear()

    def __del__(self):
        if self.active:
            self._del()
            self.context.update(self)


class InitKeysScope(AbstractScope):
    def __init__(self, context=None, **kwargs):
        super().__init__(context if context is not None else inspect.stack()[1].frame.f_locals, **kwargs)
        self.init_keys = tuple(self.context.keys())


class AutoCleanupGlobalScope(InitKeysScope, GlobalScope):
    def _del(self):
        for key in self.init_keys:
            self.context.pop(key, None)


class AllScope(InitKeysScope):
    @property
    def __all__(self):
        return tuple(filter(lambda key: key not in self.init_keys, self.context.keys()))


def unpack_and_name(raw_unit_map: Mapping[_informat, Unit]) -> Mapping[_outformat, Unit]:
    return {
        (names, abbrevs): val.set_name(first(abbrevs))
        for key, val in raw_unit_map.items()
        for (names, abbrevs) in [names_and_abbrevs(key)]
    }


def collapse_names(unit_map: Mapping[_outformat, _T]) -> Mapping[str, _T]:
    return {
        name: val for (names, abbrevs), val in unit_map.items()
        for name in unique_everseen(chain(names, abbrevs))
    }


def register_unit_map(unit_map: Mapping[_outformat, Unit],
                      globals_: MutableMapping[str, Any] = None, ignore_if_exists=False):
    if globals_ is None:
        globals_ = inspect.stack()[1].frame.f_locals
    return GlobalScope(globals_).register_many(**collapse_names(unit_map), ignore_if_exists=ignore_if_exists)
