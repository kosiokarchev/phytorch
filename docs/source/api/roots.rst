``phytorch.roots``
==================

.. |roots| replace:: `~phytorch.roots.roots`

.. automodule:: phytorch.roots

   Polynomial-root finding is a bijective [*]_ mapping from tuples (ordered sets)
   of *coefficients* to equally sized (unordered) sets of *roots*. Both coefficients
   and roots can in general be complex. If the roots are real, the coefficients
   are too, necessarily, but the converse is not true, and polynomials with real
   coefficients can have non-real roots.

   Two "algorithms" for finding roots are available:

   - For degrees up to and including 4, analytic solutions [2]_ [3]_ [4]_ have
     been known for centuries. These are implemented in
     `phytorch.extensions.roots` and are used by default. To enforce
     usage of the numeric algorithm even in these cases, supply a true
     :arg:`force_kwargs` to |roots|.

   - Polynomial roots can be calculated as the eigenvalues of the polynomial's
     `companion matrix <https://en.wikipedia.org/wiki/Companion_matrix>`_. [5]_
     Even though maybe slower\ |citation needed| and more memory consuming, this
     method applies to polynomials of any degree and is usually more numerically
     stable\ |citation needed|. It can be enforced also for :math:`N \le 4`
     via :arg:`force_numeric`.

   The gradient of |roots| is implemented as detailed in Appendix A of [ptcosmo]_.

   .. rubric:: Footnotes

   .. [*] One might object that polynomials have one more coefficients than
          roots and that the mapping is not bijective. Two birds with one stone:
          we always define the leading coefficient to be unity.

   References
   ----------

   .. [2] `Quadratic Equation <https://mathworld.wolfram.com/QuadraticEquation.html>`_ (Wolfram MathWorld)
   .. [3] `Cubic Formula  <https://mathworld.wolfram.com/CubicFormula.html>`_ (Wolfram MathWorld)
   .. [4] `Quartic Equation <https://mathworld.wolfram.com/QuarticEquation.html>`_ (Wolfram MathWorld)
   .. [5] Section 9.5.4 of `Numerical Recipes <http://numerical.recipes/book/book.html>`_ (Press et al.)
   .. [Vieta] `Vieta's formulas <https://mathworld.wolfram.com/VietasFormulas.html>`_ (Wolfram MathWorld)
   .. [ptcosmo] `Analytic auto-differentiable ΛCDM cosmography <https://arxiv.org/abs/2212.01937>`_ (Karchev 2022)

API
---

.. autosummary::

   phytorch.roots.roots
   phytorch.roots.sroots
   phytorch.roots.vieta
   phytorch.roots.companion_matrix

-----

.. autofunction:: roots
.. autofunction:: sroots

-----

.. autofunction:: vieta

-----

.. autofunction:: companion_matrix
