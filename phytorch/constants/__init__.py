import sys
from functools import partial
from importlib import import_module
from itertools import chain
from typing import Annotated, Callable, cast, MutableMapping, Type, TYPE_CHECKING

from .codata import CODATA, CODATA_vals

if TYPE_CHECKING:
    from ._codata import *
    from .astro import *


_this_module = sys.modules[__name__]
_factories: MutableMapping[str, Callable] = {
    'default': lambda: getattr(_this_module, 'codata2018'),
    'astro': lambda: import_module('.astro', __package__)
}
_default_keys = 'default', 'astro'


def _codata_mod(name: str, vals: CODATA_vals, doc: str = None):
    return cast(Type[CODATA], Annotated[CODATA, _factories.setdefault(name, partial(CODATA, name, doc, vals))])


codata2014: _codata_mod('codata2014', CODATA_vals(
    c=299_792_458,
    h=6.626_070_040e-34,
    k=1.380_648_52e-23,
    N_A=6.022_140_857e23,
    e=1.602_176_620_8e-19,
    atm=101_325,
    g=9.806_65,
    α=7.297_352_566_4e-3,
    G=6.674_08e-11,
    m_p=1.672_621_898e-27,
    m_n=1.674_927_471e-27,
    m_e=9.109_383_56e-31,
    u=1.660_539_040e-27
))
r"""CODATA 2014 release:
`<https://pml.nist.gov/cuu/pdf/wall_2014.pdf>`_."""

codata2018: _codata_mod('codata2018', CODATA_vals(
    c=299_792_458,
    h=6.626_070_15e-34,
    k=1.380_649e-23,
    N_A=6.022_140_76e23,
    e=1.602_176_634e-19,
    atm=101_325,
    g=9.806_65,
    α=7.297_352_5693e-3,
    G=6.674_30e-11,
    m_p=1.672_621_923_69e-27,
    m_n=1.674_927_498_04e-27,
    m_e=9.109_383_7015e-31,
    u=1.660_539_066_60e-27,
))
r"""CODATA 2018 release:
`<https://pml.nist.gov/cuu/pdf/wall_2018.pdf>`_."""


default: CODATA
"""The default set of CODATA constants. Currently points to `codata2018`.

Attempting to access a variable on the top-level `~phytorch.constants` module
will attempt to look the variable up on `~phytorch.constants.default`, which
allows accessing the default constants directly from `phytorch.constants`."""


def _module_all(mod):
    return mod.__all__ if hasattr(mod, '__all__') else dir(mod)


def __getattr__(name):
    if name in _factories:
        ret = _factories[name]()
    else:
        for key in _default_keys:
            if name in _module_all(mod := _factories[key]()):
                ret = getattr(mod, name)
                break
        else:
            raise AttributeError(f'module \'{__name__}\' has no attribute \'{name}\'.')

    globals()[name] = ret
    return ret


def __dir__():
    return set(chain(globals().keys(), _factories.keys(), *(
        _module_all(_factories[key]()) for key in _default_keys
    )))
