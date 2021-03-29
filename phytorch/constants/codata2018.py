from math import pi

from .Constant import Constant
from ..units._si.base import kg, m, s, K
from ..units._si.coherent import N, J, Pa, C


# Defined
c = Constant('c', 299_792_458 * m / s, 'speed of light in vacuum')
h = Constant('h', 6.626_070_15e-34 * J * s, 'Planck constant')
k = Constant('k', 1.380_649e-23 * J / K, 'Boltzman constant')
N_A = Constant('N_A', 6.022_140_76e23, 'Avogadro constant')
e = Constant('e', 1.602_176_634e-19 * C, 'elementary charge')

atm = Constant('atm', 101_325 * Pa, 'standard atmosphere')
g = Constant('g', 9.806_65 * m / s**2, 'standard acceleration of gravity')
# -----

# Measured
alpha = α = Constant('alpha', 7.297_352_5693e-3, 'fine-structure constant')
G = Constant('G', 6.674_30e-11 * N * m**2 / kg**2, 'Newtonian constant of gravitation')
m_p = Constant('m_p', 1.672_621_923_69e-27 * kg, 'proton mass')
m_n = Constant('m_n', 1.674_927_498_04e-27 * kg, 'neutron mass')
m_e = Constant('m_e', 9.109_383_7015e-31 * kg, 'electron mass')
u = Constant('u', 1.660_539_066_60e-27 * kg, 'atomic mass constant')
# --------

# Derived
hbar = Constant('hbar', h / (2 * pi), 'reduced Planck constant')
mu_0 = µ_0 = Constant('mu_0', 4 * pi * alpha * hbar / e**2 / c, 'vacuum magnetic permeability')
eps_0 = ε_0 = Constant('eps_0', 1 / mu_0 / c**2, 'vacuum electric permittivity')

mu_B = µ_B = Constant('mu_B', e * hbar / (2 * m_e), 'Bohr magneton')
mu_N = µ_N = Constant('mu_N', e * hbar / (2 * m_p), 'nuclear magneton')
Ryd = Constant('Ryd', alpha**2 * m_e * c / (2 * h), 'Rydberg constant')
a_0 = Constant('a_0', hbar / (alpha * m_e * c), 'Bohr radius')
sigma_e = σ_e = Constant('sigma_e', (8 * pi / 3) * alpha**4 * a_0**2, 'Thomson cross section')
sigma = σ = Constant('sigma', pi**2 / 60 * k**4 / (hbar**3 * c**2), 'Stefan-Boltzmann constant')
b = Constant('b', h * c / k / 4.965114231744276303, 'Wien wavelength displacement law constant')
b_prime = Constant('b`', 2.821439372122078893 * k / h, 'Wien frequency displacement law constant')
# -------

# TODO: more elegant solution
del pi, Constant, C, J, K, kg, m, N, Pa, s
