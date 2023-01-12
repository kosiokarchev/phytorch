Units and Quantities
====================

    **⚠️ Warning: under construction!**

This document briefly demonstrates |phytorch|'s functionalities related to
physical units and (unitful) quantities.

Units
-----

Creating and manipulating units
...............................

The easiest way to get started is to import some pre-defined units:

.. ipython::

    >>> from phytorch.units import si
    >>> si.kg, si.nanohertz

You can then combine them (multiply and divide by units and numbers and raise to a real power) to form new ones!

.. ipython::

    >>> si.kg * si.m / si.s**2

.. note::
    Note how the string representation of the unit encodes its "creation history". This is not a feature and will
    likely change in the future.

Every unit has a `~Unit.value` and a `~Unit.dimension`:

.. ipython::

    >>> si.eV.value, si.eV.dimension

This tells us that an electronvolt is :math:`\approx 1.6 \times 10^{-19}` of the base units for
:math:`\text{mass} \times \text{length}^2 / \text{time}^2`, which happens to be energy, for which the base unit is the joule.
Let's check:

.. ipython::

    >>> si.eV.dimension == si.joule.dimension

Converting units
................

Units of the same dimension can be converted to each other. For example, we can calculate the age of the Universe in...
days?... from the Hubble constant (don't trust all those digits, though):

.. ipython::

    >>> from phytorch.units import astro
    >>> (H0 := 100 * si.km / si.s / astro.Mpc)
    >>> age = 1 / H0
    >>> age.to(si.day)

Note that the result of the conversion is a pure number! This is equivalent to

.. ipython::

    >>> (age / si.day).value

Note that :python:`age / si.day` is still a unit, albeit a dimensionless one:

.. ipython::

    >>> (age / si.day).dimension

Trying to convert incompatible units (with different `~Unit.dimension`\ s) will result in an error:

.. ipython::

    >>> from phytorch.units.astro import lightyear

    @verbatim
    >>> age.to(lightyear)
    TypeError: Cannot convert 1 ((100 km s^(-1) Mpc^(-1))^(-1)),
               aka [T^(1)], to lyr, aka [L^(1)]

Converting units to |astropy|
.............................

`AstroPy's units module <https://docs.astropy.org/en/stable/units/index.html>`_ is the inspiration behind
`phytorch.units`, and so a |phytorch| `Unit` can be easily converted to an `astropy.units.Unit`:

.. ipython::

    >>> age_ap = age.toAstropy('age')
    >>> (type(age_ap), age_ap, age_ap.represents, age_ap.to('day'))

.. todo::
    Convert `astropy.units.Unit`\ s to |phytorch| `Unit`\ s.

Constants
.........

`phytorch.constants` provides universal constants as defined by `CODATA <https://physics.nist.gov/cuu/Constants/index.html>`_.
Additionally, `phytorch.constants.astro` defines some astronomical... ahem... constants. See `the documentation
</api/constants>` for a full list.

`Constant`\ s are nothing more than `Unit`\ s with a few presentational bells and whistles:

.. ipython::

    >>> from phytorch.constants import c, m_e
    >>> c, m_e
    >>> (E_e := m_e * c**2)
    >>> E_e.to(si.keV)

Last one: size of the Universe in Plank lengths:

.. ipython::

    >>> from phytorch.constants import G, ħ
    >>> (age * c).to((ħ * G / c**3)**0.5)

Quantities
----------

*Quantities*---a combination of a `~torch.Tensor` and a `Unit`---allow unit information to be propagated through
numerical calculations, automatically deriving the units of resulting quantities while also acting as a safeguard
against nonsensical operations, like taking the logarithm of unitful quantities.

Creating quantities
...................

A `Quantity` can be created by multiplying (or dividing) a `~torch.Tensor` and a `Unit`:

.. ipython::

    >>> import torch
    >>> (q := torch.rand((2, 3)) * si.hour)

.. todo:: Presentation details are still to be ironed out.

.. warning:: Creating a `Quantity` backed by a non-float-typed `~torch.Tensor` is largely undefined and may not behave as one'd expect.

Similarly to a `Unit`, the `~TensorQuantity.value` and `~TensorQuantity.unit` of a `Quantity` can be accessed as attributes:

.. ipython::

    >>> q.value
    >>> q.unit

Note that `Quantity.value <~TensorQuantity.value>` is a *view* of the underlying data, so changes to it will be reflected
back on the `Quantity`:

.. ipython::

    >>> q.value[0, :] = 1.
    >>> q

The same is true of the `~torch.Tensor` used to create the `Quantity`:

.. ipython::

    >>> (q := (t := torch.rand(4)) * si.hour)
    >>> t[2] = 3.14
    >>> q

But note that

.. ipython::

    >>> q.value is not t

Converting quantities
.....................

`Quantities <Quantity>` can be converted to different units of the same dimension using `~GenericQuantity.to`:

.. ipython::

    >>> q.to(si.minute)

Note that this creates an entirely different `Quantity` backed by a separate `~torch.Tensor`, so the original is not
modified. Unless you convert to a unit that is *equal* to the current one: in that case some computation is spared,
and the original quantity is returned:

.. ipython::

    >>> q.to(60*si.minute) is q

Note further that in this case the `~TensorQuantity.unit` remains *the same object*, so you cannot expect that
:python:`quantity.to(unit).unit is unit` but only that :python:`quantity.to(unit).unit == unit`.

Of course, wrong conversions raise exceptions:

.. ipython::

    @verbatim
    >>> q.to(si.kilometer)
    TypeError: Cannot convert h, aka [T^(1)], to km, aka [L^(1)]

Manipulating quantities
.......................

|phytorch|'s `quantities <Quantity>` support the full set of mathematical operations defined by |pytorch|\ [*]_ in a
way that makes sense for the unit information that they carry:

.. ipython::
    :suppress:

    >>> from phytorch.units import Unit
    >>> from phytorch.units.si import h, min, km
    >>> t = torch.tensor

- For starters, you can only add / subtract quantites with compatible units:\ [*]_

  .. ipython::

      >>> t(1.) * h + t(15.) * min

      @verbatim
      >>> t(1.) * h + t(15.) * km
      UnitError: expected [L^(-1) T^(1)] but got dimensionless.

  Numbers and pure `~torch.Tensor`\ s are considered dimensionless quantities:

  .. ipython::

      @verbatim
      >>> t(1.) * h + 5.
      UnitError: expected [T^(1)] but got dimensionless.

- Comparisons also only work with compatible quantities:

  .. ipython::

      >>> t(1.) * h <= t([30., 60., 90.]) * min

      @verbatim
      >>> t(1.) * Unit(apple=1) > t(1.) * Unit(orange=1)
      UnitError: expected [apple^(1)] but got [orange^(1)].

- On the other hand, you can multiply / divide any quantities:

  .. ipython::

      >>> (t(120.) * km) / (t(72.) * min)

  Note that the numeric `~GenericQuantity.value` is now just the result of the operation on the
  `~GenericQuantity.value`\ s of the original `Quantities <Quantity>`, unlike in addition when the second `Quantity`
  was converted.

- And raise a `Quantity` to a *scalar* power

  .. ipython::

      >>> (t(3.14) * km)**2
      >>> (t(3.14) * km)**torch.tensor(2.)
      >>> (t(3.14) * km).sqrt()

  Raising each element to a different power results in different units, so it is forbidden:

  .. ipython::

      @verbatim
      >>> (t(3.14) * km) ** torch.tensor([2., 3., 4.])
      ValueError: only one element tensors can be converted to Python scalars

  Since dimensionless units are still units, this extends also to the case of a dimensionless `Quantity` as base of
  exponentation. Finally, raising to the power of a `Quantity` is allowed only if it's dimensionless and has a single
  `~torch.Tensor.item`:

  .. ipython::

      >>> (t(2.) * km) ** ((t(1.) * h) / (t(30.) * min))

- Most mathematical single-argument functions are only allowed for dimensionless quantities:

  .. ipython::

      @verbatim
      >>> torch.exp(t(1.) * km)
      UnitError: expected [] but got [L^(1)].

      >>> torch.sin((t(1.) * h) / (t(15.) * min))

  Note that the result is an ordinary `~torch.Tensor`! And that the `radian` is defined as a dimensionless unit with
  unit scale:

  .. ipython::

      >>> from phytorch.units import angular
      >>> angular.radian.value, angular.radian.dimension
      >>> torch.sin(t(4.) * angular.rad)

- All in all, |pytorch| countains countless\ |citation needed| operations. The user is encouraged to try them out and
  see for themselves if the results are sensible.

.. rubric:: Footnotes

.. [*] Ha! Did you really believe me?!
.. [*] Fun fact about the error in this example: ``aTensor + bTensor`` dispatches to `torch.add`, which has an
       additional argument, ``alpha``, meant to rescale ``bTensor`` before adding it to ``aTensor``. Given the units
       ``aUnit`` and ``bUnit``, the operation is only permitted if ``alphaUnit`` is compatible with the ratio
       ``aUnit / bUnit``, but since the default value is ``1`` (dimensionless), |phytorch| complains.
