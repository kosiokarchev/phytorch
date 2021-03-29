from fractions import Fraction

from ._prefixes import _prefix_many_to_many
from ._utils import register_unit_map, unpack_and_name
from .angular import arcsec
from .si.additional import AU, day
from .si.base import m
from .si.coherent import Hz, Pa, W
from ..constants import atm, c, h, Ryd


unit_map = unpack_and_name({
    ('bar',): 10**5 * Pa, ('barn',): Fraction(10)**(-28) * m**2,
    ('year', 'yr'): (yr := Fraction(365.25) * day),
    ('Jansky', 'Jy'): Fraction(10)**(-26) * W / Hz / m**2,
    ('lightyear', 'lyr'): c*yr,
    ('parsec', 'pc'): AU / arcsec,
    ('Rydberg', 'Ry'): Ryd * h * c,
    ('Torr',): atm / 760,
})

register_unit_map(unit_map).register_many(**_prefix_many_to_many(unit_map, except_=('Ry',)))
