|phytorch|.cosmology
====================

The `phytorch.cosmology` module is modelled after |astropy|'s `sub-package of
the same name <astropy-cosmology>` and provides routines for calculating
cosmographic distances and related quantities in a FLRW cosmology.

The structure of the codebase is somewhat involved so as to be maximally
composable and extensible. In a gist, classes can be abstract, concrete, or
drivers. Hierarchies of *abstract* classes define the overall interface and
implement shared functionalities. *Concrete* classes implement the details of
concrete cosmological models, e.g. ΛCDM (`with <~concrete.LambdaCDMR>`
or `without <~concrete.LambdaCDM>` radiation, and/or
`flat <~cosmology.special.flat>`), or with evolving dark energy.
*Drivers* fill in the routines for calculating distances using, e.g.
`analytical formulae <drivers.analytic>` or `numerical integration
<drivers.odeint>`. Finally, user-facing classes must combine a *concrete* model
with a *driver* to unlock the full functionality.

.. rubric:: Concrete models

Concrete classes are located in the `phytorch.cosmology.special.concrete`
sub-module. Currently, thee are five dark energy models implemented, each of
which has four corresponding classes with/without radiation and with/without
curvature:

* ``(Flat)LambdaCDM(R)``: cosmological constant :math:`\Lambda \implies w = -1`;

* ``(Flat)wCDM(R)``: constant equation of state :math:`w = w_0`;

* ``(Flat)w0waCDM(R)``: linear evolution with scale factor:

  .. math::
     w(z) = w_0 + w_a (1-a) = w_0 + w_a \left(1 - \frac{1}{z+1}\right);

* ``(Flat)wpwaCDM(R)``: pivoting equation of state:

  .. math::
     w(z) = w_p + w_a (a_p-a) = w_p + w_a \left(\frac{1}{z_p+1} - \frac{1}{z+1}\right);

* ``(Flat)w0wzCDM(R)``: linear evolution with redshift: :math:`w(z) = w_0 + w_z z`.

.. rubric:: Cosmography drivers

The driver sub-modules located under `phytorch.cosmology.drivers` provide
implementations of cosmographic distance calculations for the models they support:

* `~drivers.abstract`: allows one to instantiate a concrete model without
  necessarily implementing distances (e.g. in order to calculate density
  evolutions). Supports all 20 concrete models (trivially).

* `~drivers.odeint`: calculates the distances using numerical integration with
  the |torchdiffeq| package. Supports all 20 concrete models but currently has
  trouble with batching.

* `~drivers.analytic`: implements analytic expressions based on Carlson's
  elliptic integrals. Fully differentiable, batched, and awesome; however,
  only supports ``(Flat)LambdaCDM(R)`` models and requires compiling the
  :doc:`/guide/extensions`.

  .. todo::
     Reference paper!


Converting to |astropy|
-----------------------

.. todo::
   Add ability to convert *from* |astropy|.
