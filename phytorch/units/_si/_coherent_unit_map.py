from copy import copy

from .base import A, K, kg, m, s
from .._utils import unpack_and_name


coherent_unit_map = unpack_and_name({
    ('Hertz', 'Hz'): 1/s,
    'Newton': (N := kg * m / s**2), ('Pascal', 'Pa'): N / m**2,
    'Joule': (J := N * m), 'Watt': (W := J / s),
    'Coulomb': (C := A * s), 'Volt': (V := W / A), 'Farad': C / V,
    ('Ohm', 'Ω'): (Ω := V / A), ('Siemens', ('S', '℧')): 1 / Ω,
    ('Weber', 'Wb'): (Wb := V * s), 'Tesla': Wb / m**2, 'Henry': Wb / A,
    (('Celsius', 'degree_Celsius'), ('deg_C', '°C')): copy(K),
    ('Bequerel', 'Bq'): 1/s,
    ('Gray', 'Gy'): J / kg, ('Sievert', 'Sv'): J / kg
})
