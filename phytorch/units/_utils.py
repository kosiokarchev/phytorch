import inspect
from itertools import chain
from typing import Any, Iterable, Mapping, MutableMapping, Tuple, TypeVar, Union

from more_itertools import collapse, first, unique_everseen

from .unit import Unit
from ..utils.scoping import GlobalScope


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
