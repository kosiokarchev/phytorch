# noinspection PyUnresolvedReferences
from ..si import s, second, g, gram, cm, centimeter, centimetre


gal = Gal = (cm / s**2).set_name('Gal')
dyne = dyn = (g * Gal).set_name('dyn')
erg = (dyn * cm).set_name('erg')
barye = Ba = (dyn / cm**2).set_name('Ba')
poise = P = (Ba * s).set_name('P')
stokes = St = (cm**2 / s).set_name('St')
kayser = ~cm
