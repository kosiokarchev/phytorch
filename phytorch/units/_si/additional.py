from collections import ChainMap
from fractions import Fraction

from .base import kg, m, s
from .coherent import V
from .prefixed import dm, hm
from .._prefixes import _prefix_many_to_many
from .._utils import names_and_abbrevs, register_unit_map, unpack_and_name
from ...constants import e


eVdef = {names_and_abbrevs(('electronvolt', 'eV')): e * V}

register_unit_map(unpack_and_name(ChainMap({
    ('minute', 'min'): 60 * s, 'hour': 3600 * s, 'day': 86400 * s,
    ('astronomical_unit', ('au', 'AU')): 149_597_870_700 * m,
    ('hectare', 'ha'): hm**2, (('litre', 'liter'), ('L', 'l')): dm**3,
    (('ton', 'tonne'), 't'): 10**3 * kg,
    # my additions:
    (('angstrom', 'ångström'), ('A', 'Å')): Fraction('1e-10') * m
}, eVdef))).register_many(**_prefix_many_to_many(eVdef))
