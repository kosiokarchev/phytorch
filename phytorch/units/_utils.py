import inspect
from dataclasses import dataclass
from decimal import Decimal, Context as dContext
from fractions import Fraction
from itertools import chain
from numbers import Real
from typing import Any, Iterable, Mapping, MutableMapping, Tuple, TypeVar, Union

from more_itertools import collapse, first, unique_everseen

from .unit import Unit, Dimension
from ..utils.scoping import AutoCleanupGlobalScope
from ..utils.symmetry import product

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
    return AutoCleanupGlobalScope(globals_).register_many(**collapse_names(unit_map), ignore_if_exists=ignore_if_exists)


@dataclass
class Latexifier:
    base_units: Mapping[Dimension, Unit]

    power_as_frac: bool = True
    keep_unit_power: bool = False

    def base_from(self, u: Unit) -> Unit:
        return product(
            (self.base_units[dim]**p for dim, p in u.items()),
            Unit()
        )

    def to_base(self, u: Unit) -> float:
        return u.to(self.base_from(u))

    @staticmethod
    def format_value(value: Real, prec: int = 16) -> str:
        dv = Decimal(float(value))
        sign, digits, exponent = dv.as_tuple()
        exponent = len(digits) + exponent - 1

        return r' \times '.join(filter(bool, (
            str(dv.scaleb(-exponent).normalize(dContext(prec=prec))),
            f'10^{{{exponent}}}' if exponent else ''
        )))

    def format_power(self, p: Fraction) -> str:
        return (
            str(p.numerator) if p.denominator == 1 else
            ('-' if p < 0 else '') + rf'\frac{{{abs(p.numerator)}}}{{{p.denominator}}}'
            if self.power_as_frac else
            f'{p.numerator} / {p.denominator}'
        )

    def format_units(self, u: Unit) -> Mapping[Dimension, str]:
        return {
            dim: rf'\mathrm{{{self.base_units[dim].name}}}' + (
                '' if p == 1 and not self.keep_unit_power else
                rf'^{{{self.format_power(p)}}}'
            )
            for dim, p in u.items()
        }

    def __call__(self, u: Unit, prec: int = 16):
        formatted_units = self.format_units(u)
        return r'\ '.join(filter(bool, (
            self.format_value(self.to_base(u)),
            r'\,'.join(
                formatted_units[dim]
                for dim in self.base_units.keys() if dim in formatted_units
            )
        )))
