``phytorch.constants``
======================

.. py:currentmodule:: phytorch.constants

`Constant`\ s are just regular `Unit`\ s with a memorable `~Unit.name` and a `~Constant.description`.

The primary set of constants comes from the |CODATAweb|_ recommended values. Since they change every few years,
|phytorch| organises the values into submodules: `codata2014`, `codata2018`, and the special `default`,
which points to the latest release. Constants can be accessed either from any of the `CODATA` modules, or from
`phytorch.constants` directly (which will fetch them from `~phytorch.constants.default`):

.. ipython::

    >>> from phytorch import constants as const
    >>> const.default is const.codata2018
    >>> const.c is const.default.c is const.codata2018.c
    >>> const.codata2014.G, const.codata2018.G

CODATA constants
----------------

`CODATA` constants fall in several categories (here we list values from `default`):

- fundamental defined constants with values set by defining the system of units itself:

    .. list-table::
       :header-rows: 1
       :align: left
       :width: 100%

       * - name(s)
         - description
         - symbol
         - definition

       * - .. ptconst:: c
         - speed of light in vacuum
         - :math:`c`
         - :math:`299\,792\,458\ \mathrm{m} / \mathrm{s}`
       * - .. ptconst:: h
         - Planck constant
         - :math:`h`
         - :math:`6.626\,070\,15 \times 10^{-34}\ \mathrm{J}\,\mathrm{s}`
       * - .. ptconst:: k
         - Boltzmann constant
         - :math:`k`
         - :math:`1.380\,649 \times 10^{-23}\ \mathrm{J} / \mathrm{K}`
       * - .. ptconst:: e
         - elementary charge
         - :math:`e`
         - :math:`1.602\,176\,634 \times 10^{-19}\ \mathrm{A}\,\mathrm{s}`
       * - .. ptconst:: N_A
         - Avogadro constant
         - :math:`N_A`
         - :math:`6.022\,140\,76 \times 10^{23}`

- conventional Earth-related "constants" (these are also "exact"):

    .. list-table::
       :header-rows: 1
       :align: left
       :width: 100%

       * - name(s)
         - description
         - symbol
         - value

       * - .. ptconst:: atm
         - standard atmosphere
         - :math:`\mathrm{atm}`
         - :math:`101\,325\ \mathrm{Pa}`
       * - .. ptconst:: g
         - standard acceleration of gravity
         - :math:`g`
         - :math:`9.80665\ \mathrm{m} / \mathrm{s}^{2}`

- other fundamental constants, whose values are derived from measurements:

    .. list-table::
       :header-rows: 1
       :align: left
       :width: 100%

       * - name(s)
         - description
         - symbol
         - measured value

       * - .. ptconst:: α
                        alpha
         - fine-structure constant
         - :math:`\alpha`
         - :math:`7.297\,352\,5693(11) \times 10^{-3}`
       * - .. ptconst:: G
         - Newtonian constant of gravitation
         - :math:`G`
         - :math:`6.674\,30(15) \times 10^{-11}\ \mathrm{kg}^{-1}\,\mathrm{m}^{3}\,\mathrm{s}^{-2}`
       * - .. ptconst:: m_p
         - proton mass
         - :math:`m_p`
         - :math:`1.672\,621\,923\,69(51) \times 10^{-27}\ \mathrm{kg}`
       * - .. ptconst:: m_n
         - neutron mass
         - :math:`m_n`
         - :math:`1.674\,927\,498\,04(95) \times 10^{-27}\ \mathrm{kg}`
       * - .. ptconst:: m_e
         - electron mass
         - :math:`m_e`
         - :math:`9.109\,383\,7015(28) \times 10^{-31}\ \mathrm{kg}`
       * - .. ptconst:: u
         - atomic mass constant
         - :math:`m_u`
         - :math:`1.660\,539\,066\,60(50) \times 10^{-27}\ \mathrm{kg}`

- derived constants, which are just convenient shorthand combinations of other constants (these are calculated
  dynamically and so are consistent within the `CODATA` set):

    .. list-table::
       :header-rows: 1
       :align: left
       :width: 100%

       * - name(s)
         - description
         - symbol
         - definition

       * - .. ptconst:: ħ
                        hbar
         - reduced Planck constant
         - :math:`\hbar`
         - :math:`h / (2\pi)`
       * - .. ptconst:: μ_0
                        mu_0
         - vacuum magnetic permeability
         - :math:`\mu_0`
         - :math:`4\pi \, \alpha \hbar / (c e^2)`
       * - .. ptconst:: ε_0
                        eps_0
         - vacuum electric permittivity
         - :math:`\varepsilon_0`
         - :math:`1 / (\mu_0 c^2)`
       * - .. ptconst:: μ_B
                        mu_B
         - Bohr magneton
         - :math:`\mu_B`
         - :math:`e \hbar / (2 m_e)`
       * - .. ptconst:: μ_N
                        mu_N
         - nuclear magneton
         - :math:`\mu_N`
         - :math:`e \hbar / (2 m_p)`
       * - .. ptconst:: Ryd
         - Rydberg constant
         - :math:`R_{\infty}`
         - :math:`\alpha^2 m_e c / (2 h)`
       * - .. ptconst:: a_0
         - Bohr radius
         - :math:`a_0`
         - :math:`\hbar / (\alpha m_e c)`
       * - .. ptconst:: σ_e
                        sigma_e
         - Thomson cross section
         - :math:`\sigma_e`
         - :math:`(8\pi/3) \, \alpha^4 a_0^2`
       * - .. ptconst:: σ
                        sigma
         - Stefan-Boltzmann constant
         - :math:`\sigma`
         - :math:`(\pi^2/60) \, k^4 / (\hbar^3 c^2)`
       * - .. ptconst:: b
         - Wien wavelength displacement law constant
         - :math:`b`
         - :math:`(h c / k) \,/\, 4.965\!\ldots`
       * - .. ptconst:: b_prime
         - Wien frequency displacement law constant
         - :math:`b^\prime`
         - :math:`2.821\!\ldots \, (k / h)`

Astronomical constants
----------------------

- Additionally, `phytorch.constants`\ [*]_ defines a limited set of "astronomical" constants:\ [*]_

    .. list-table::
       :header-rows: 1
       :align: left
       :width: 100%

       * - name(s)
         - description
         - symbol
         - value\ :raw-html:`<sup><a href="https://ssd.jpl.nasa.gov/planets/phys_par.html">a</a>, <a href="https://web.archive.org/web/20131110215339/http://asa.usno.navy.mil/static/files/2014/Astronomical_Constants_2014.pdf">b</a></sup>`

       * - .. ptconst:: M_earth
                        earthMass
         - Earth mass
         - :math:`M_⊕`
         - :math:`5.972 \times 10^{24}\ \mathrm{kg}`
       * - .. ptconst:: R_earth
                        earthRad
                        earthRadius
         - Earth equatorial radius
         - :math:`R_⊕`
         - :math:`6.378 \times 10^{6}\ \mathrm{m}`

       * - .. ptconst:: M_jupiter
                        jupiterMass
         - Jupiter mass
         - :math:`M_♃`
         - :math:`1.898 \times 10^{27}\ \mathrm{kg}`
       * - .. ptconst:: R_jupiter
                        jupiterRad
                        jupiterRadius
         - Jupiter equatorial radius
         - :math:`R_♃`
         - :math:`7.149 \times 10^{7}\ \mathrm{m}`

       * - .. ptconst:: M_sun
                        solMass
                        solarMass
         - solar mass
         - :math:`M_☉`
         - :math:`1.988 \times 10^{30}\ \mathrm{kg}`
       * - .. ptconst:: R_sun
                        solRad
                        solarRadius
         - solar radius
         - :math:`R_☉`
         - :math:`6.957 \times 10^{8}\ \mathrm{m}`
       * - .. ptconst:: L_sun
                        solLum
                        solarLuminosity
         - solar luminosity
         - :math:`L_☉`
         - :math:`3.828 \times 10^{26}\ \mathrm{W}`

.. rubric:: Footnotes

.. [*] In fact, these are defined in `phytorch.constants.astro` but exported in the same way as constants from `default`.

.. [*] Note that constants are *quantities* in the sense that they refer to concrete properties of the real Universe.
       See also `phytorch.units.astro` for astro(nomical|physical) *units* that are *used* for measuring the Universe.

API
---

Classes
.......

.. autosummary::

   phytorch.constants.constant.Constant

.. autoclass:: phytorch.constants.constant.Constant
   :members:

-----

Modules
.......

.. autosummary::

   phytorch.constants.astro
   phytorch.constants.default
   phytorch.constants.codata2014
   phytorch.constants.codata2018

.. automodule:: phytorch.constants
   :members: default, codata2014, codata2018

   .. py:data:: phytorch.constants.astro
      :type: module

      A module that defines some astronomical constants, which are exported to `phytorch.constants` in the same way as
      those from `default`.

-----

Helper classes
..............

.. autosummary::

   phytorch.constants.codata.CODATA
   phytorch.constants.codata.CODATA_vals

.. automodule:: phytorch.constants.codata
   :members:
