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

      pip install .

   This will automatically install the few dependencies and the pure-|Python| code.
   If you are not interested in special functions and differentiable
   cosmographic distances, you're good to go!

   .. todo::

      Release the package to PyPI!
#. Then, to compile the extensions,

   .. code-block::

      cd phytorch-extensions
      python setup.py build_ext

   .. warning::

      Building the extensions currently **requires** you to have a CUDA compiler,
      which, I realise, sucks. I'm working on making a cpu-only version possible.

   Finally, the extensions, which have now been built in a folder like
   ``phytorch-extensions/build/lib*``, need to be linked to
   ``phytorch/extension``::

      cd ../phytorch/extensions
      ln -s ../../phytorch-extensions/build/lib*/* .


   .. todo::

      Automate installation of the extensions.

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
