import sys
from functools import partial
from itertools import chain
from typing import Annotated, Callable, cast, MutableMapping, Type, TYPE_CHECKING

from .astro import *
from .codata import CODATA, CODATA_vals


if TYPE_CHECKING:
    from ._codata import *


_this_module = sys.modules[__name__]
_factories: MutableMapping[str, Callable] = {
    'default': lambda: getattr(_this_module, 'codata2018')
}
_default = lambda: getattr(_this_module, 'default')
default: CODATA


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


def __getattr__(name):
    if name in _factories:
        ret = _factories[name]()
    elif name in (d := _default()).__all__:
        ret = getattr(d, name)
    else:
        raise AttributeError(f'module \'{__name__}\' has no attribute \'{name}\'.')

    globals()[name] = ret
    return ret


def __dir__():
    return chain.from_iterable(map(set, (globals().keys(), _factories.keys(), _default().__all__)))
