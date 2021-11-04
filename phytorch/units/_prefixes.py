import operator
from fractions import Fraction
from itertools import chain, product, starmap
from typing import Container, Mapping

from more_itertools import first

from ._utils import _outformat, names_and_abbrevs
from .unit import Unit


_prefix_map = {
    names_and_abbrevs(prefix): expon
    for eseq in [(*range(1, 4), *range(6, 25, 3))]
    for prefix, expon in zip(
        ('yocto', 'zepto', 'atto', 'femto', 'pico', 'nano',
         ('micro', ('u', 'μ')), 'milli', 'centi', 'deci',
         ('deca', 'da'), 'hecto', 'kilo', 'Mega', 'Giga',
         'Tera', 'Peta', 'Exa', 'Zetta', 'Yotta'),
        chain(map(operator.neg, reversed(eseq)), eseq)
    )
}


def _generic_prefix(name, expon, unit):
    return (Fraction(10)**expon * unit).set_name(name + unit.name)


def _prefix_many_to_many(unit_map: Mapping[_outformat, Unit], prefix_map=_prefix_map,
                         except_: Container[str] = ()):
    return {
        name: punit
        for (names, abbrevs), unit in unit_map.items()
        if unit.name not in except_
        for (prefixes, prefix_abbrevs), expon in prefix_map.items()
        for punit in [(Fraction(10)**expon * unit).set_name(first(prefix_abbrevs) + unit.name)]
        for name in map(''.join, chain(*starmap(product, zip((prefixes, prefix_abbrevs), (names, abbrevs)))))
    }
