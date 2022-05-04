from abc import ABC

from . import concrete
from ..core import FLRW, H100
from ..utils import _GQuantity, _no_value
from ...utils._typing import _TN


class FlatFLRWMixin(FLRW, ABC):
    def transform_curved(self, distance_dimless: _TN) -> _TN: ...
    def comoving_volume_dimless(self, z: _TN, eps: float = ...) -> _TN: ...

class FlatLambdaCDM(FlatFLRWMixin, concrete.LambdaCDM, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN = _no_value, Ob0: _TN = 0.): ...

class FlatLambdaCDMR(FlatFLRWMixin, concrete.LambdaCDMR, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN = _no_value, Ob0: _TN = 0.,
                 Or0: _TN = 0., Tcmb0: _GQuantity = _no_value): ...

class FlatwCDM(FlatFLRWMixin, concrete.wCDM, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN = _no_value, Ob0: _TN = 0.,
                 w0: _TN = -1.): ...

class FlatwCDMR(FlatFLRWMixin, concrete.wCDMR, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN = _no_value, Ob0: _TN = 0.,
                 w0: _TN = -1.,
                 Or0: _TN = 0., Tcmb0: _GQuantity = _no_value): ...

class Flatw0waCDM(FlatFLRWMixin, concrete.w0waCDM, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN = _no_value, Ob0: _TN = 0.,
                 w0: _TN = -1., wa: _TN = 0.): ...

class Flatw0waCDMR(FlatFLRWMixin, concrete.w0waCDMR, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN +_no_value, Ob0: _TN = 0.,
                 w0: _TN = -1., wa: _TN = 0.,
                 Or0: _TN = 0., Tcmb0: _GQuantity = _no_value): ...

class FlatwpwaCDM(FlatFLRWMixin, concrete.wpwaCDM, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN =_no_value, Ob0: _TN = 0.,
                 w0: _TN = -1., wa: _TN = 0., zp: _TN = 0., wp: _TN = _no_value): ...

class FlatwpwaCDMR(FlatFLRWMixin, concrete.wpwaCDMR, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN = _no_value, Ob0: _TN = 0.,
                 w0: _TN = -1., wa: _TN = 0., zp: _TN = 0., wp: _TN = _no_value,
                 Or0: _TN = 0., Tcmb0: _GQuantity = _no_value): ...

class Flatw0wzCDM(FlatFLRWMixin, concrete.w0wzCDM, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN = _no_value, Ob0: _TN = 0.,
                 w0: _TN = -1., wz: _TN = 0.): ...

class Flatw0wzCDMR(FlatFLRWMixin, concrete.w0wzCDMR, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN = _no_value, Ob0: _TN = 0.,
                 w0: _TN = -1., wz: _TN = 0.,
                 Or0: _TN = 0., Tcmb0: _GQuantity = _no_value): ...
