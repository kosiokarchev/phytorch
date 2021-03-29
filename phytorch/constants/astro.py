from more_itertools import last

from ..units._si.base import kg, m
from ..units._si.coherent import W
from ..units._utils import GlobalScope


GlobalScope().register_many(**{name: val for names, (val, description) in {
    ('earthMass', 'M_⊕', 'M_earth'): (5.9721679e24 * kg, 'Earth mass'),
    ('earthRadius', 'earthRad', 'R_⊕', 'R_earth'): (6378100 * m, 'Earth equatorial radius'),
    ('jupiterMass', 'M_♃', 'M_jupiter'): (1.8981246e27 * kg, 'Jupiter mass'),
    ('jupiterRadius', 'jupiterRad', 'R_♃', 'R_jupiter'): (71492000 * m, 'Jupiter radius'),
    ('solarMass', 'solMass', 'M_☉', 'M_sun'): (1.9884099e30 * kg, 'solar mass'),
    ('solarRadius', 'solRad', 'R_☉', 'R_sun'): (6.957e8 * m, 'solar radius'),
    ('solarLuminosity', 'solLum', 'L_☉', 'L_sun'): (3.828e26 * W, 'solar luminosity')
}.items() for val in [val.set_name(last(names))] for name in names})
