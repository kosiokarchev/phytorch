# φtorch: spark up your physics
#### *utilities for computational physics based on PyTorch*

---

[![pypi](https://img.shields.io/pypi/v/phytorch?logo=pypi)](https://pypi.org/project/phytorch/)
[![docs](https://readthedocs.org/projects/phytorch/badge/?version=latest)](https://phytorch.readthedocs.io/en/latest/?badge=latest)
![license](https://img.shields.io/pypi/l/phytorch)

---

`phytorch` (φtorch) is a package for doing physics using
[PyTorch](https://pytorch.org/) as the backend for computations, which allows
parallelisation on GPUs and automatic differentiation.
[Check out the docs](https://phytorch.readthedocs.org) to see what's on offer!

## Installation

φtorch consists of Python modules and a PyTorch extension written in C++/CUDA.
Therefore, installation has two steps, but should otherwise be fully automatic.

1. First, clone (or download) the source code, then `cd` into the root directory
    and run
    ```shell
    pip install -e .
    ```
    This will automatically install the few dependencies and the pure-Python 
    code (in developer mode, which means that it won't be installed in the
    `site-packages` but will run out of the current directory). If you are not
    interested in special functions and differentiable cosmographic distances,
    you’re good to go!
   > **Note**
   > In that case, you can also now install the pure-Python components from
   > PyPI using
   > ```shell
   > pip install phytorch
   > ```

2. Then, to compile the extensions,
    ```shell
    cd phytorch-extensions
    python setup.py build_ext
    ```
   > **Warning**
   > *Building the extensions currently requires you to have a CUDA compiler,
   > which, I realise, sucks. I’m working on making a cpu-only version possible.*

    Finally, the extensions, which have now been built in a folder like
    `phytorch-extensions/build/lib*`, need to be linked to `phytorch/extensions`:
    ```shell
    cd ../phytorch/extensions
    ln -s ../../phytorch-extensions/build/lib*/* .
    ```

---
*This program is free software and comes with no warranty except that which can
be reasonably expected of computer logic. It is, however, licensed under the
GNU General Public Licence v3.0, which disallows closed source re-distribution
and any such foul nonsense.*
