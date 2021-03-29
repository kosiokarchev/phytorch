from more_itertools import one

from .._utils import names_and_abbrevs
from ..Unit import CURRENT, Dimension, LENGTH, MASS, TEMPERATURE, TIME, Unit


base_unit_map = {
    (names, abbrevs): Unit(dim, name=one(abbrevs))
    for (names, abbrevs), dim in {
        names_and_abbrevs(item): {val: 1} if isinstance(val, Dimension) else val
        for item, val in {
            (('meter', 'metre'), 'm'): LENGTH, 'second': TIME,
            ('kilogram', 'kg'): MASS, 'Ampere': CURRENT, 'Kelvin': TEMPERATURE,
        }.items()
    }.items()
}
