from more_itertools import consume

from .constant import Constant
from ..units._si.base import kg, m
from ..units._si.coherent import W
from ..utils.scoping import AllScope


_all_scope = AllScope()

consume({
    globals().__setitem__(name, val)
    for names, (val, description) in {
        ('M_⊕', 'earthMass', 'M_earth'): (5.97217e24 * kg, 'Earth mass'),
        ('R_⊕', 'earthRadius', 'earthRad', 'R_earth'): (6378136.6 * m, 'Earth equatorial radius'),
        ('M_♃', 'jupiterMass', 'M_jupiter'): (1.898125e27 * kg, 'Jupiter mass'),
        ('R_♃', 'jupiterRadius', 'jupiterRad', 'R_jupiter'): (71492000 * m, 'Jupiter equatorial radius'),
        ('M_☉', 'solarMass', 'solMass', 'M_sun'): (1.98847e30 * kg, 'solar mass'),
        ('R_☉', 'solarRadius', 'solRad', 'R_sun'): (6.957e8 * m, 'solar radius'),
        ('L_☉', 'solarLuminosity', 'solLum', 'L_sun'): (3.828e26 * W, 'solar luminosity')
    }.items()
    for val in [Constant(names[0], val, description)]
    for name in names
})

__all__ = tuple(filter(lambda name: not name.startswith('_'), _all_scope.__all__))
