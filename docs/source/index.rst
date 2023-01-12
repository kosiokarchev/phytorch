.. phytorch documentation master file, created by
   sphinx-quickstart on Wed Nov  9 10:14:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to |phytorch|'s documentation!
======================================

``phytorch`` (stylised as |phytorch|) is a |Python| package that provides
physics-related utilities to the |pytorch| ecosystem.

Installation instructions
-------------------------

.. highlight:: shell

|phytorch| consists of |Python| modules and a |pytorch| extension written in
C++/CUDA. Therefore, installation has two steps, but should otherwise be fully
automatic.

#. First, clone (or download) the source code, then :shell:`cd` into the root
   directory and run

   .. code-block::

      pip install -e .

   This will automatically install the few dependencies and the pure-|Python|
   code (in developer mode, so that it runs out of the current directory
   instead of ``site-packages``). If you are not interested in special
   functions and differentiable cosmographic distances, you're good to go!

   .. note::

      In that case, you can also now install the pure-Python components from
      `PyPI <https://pypi.org/project/phytorch/>`_ using

      .. code-block::

         pip install phytorch

#. Then, to compile the extensions,

   .. code-block::

      cd phytorch-extensions
      python setup.py build_ext -b ../phytorch/extensions

   .. todo::

      Automate compiling the extensions.

Contents
--------

.. toctree::
   :maxdepth: 2

   guide/index
   examples
   api/index

Indices and tables
..................

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
