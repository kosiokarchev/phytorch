import abc
from abc import ABC, abstractmethod

from .decdm import BaseDECDM, RadiationFLRWMixin
from ..core import H100
from ..utils import _GQuantity, _no_value
from ...utils._typing import _TN


class BasewCDM(BaseDECDM, ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def _w(self, zp1: _TN) -> _TN: ...
    def w(self, z: _TN, zp1: _TN = ...) -> _TN: ...


class wCDM(BasewCDM, ABC):
    w0: _TN

    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN, Ob0: _TN = 0.,
                 w0: _TN = -1.): ...

class wCDMR(RadiationFLRWMixin, wCDM, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN, Ob0: _TN = 0.,
                 w0: _TN = -1.,
                 Or0: _TN = 0., Tcmb0: _GQuantity = _no_value): ...


class LambdaCDM(wCDM, ABC):
    w0 = -1

    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN, Ob0: _TN = 0.): ...

class LambdaCDMR(RadiationFLRWMixin, LambdaCDM, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN, Ob0: _TN = 0.,
                 Or0: _TN = 0., Tcmb0: _GQuantity = _no_value): ...


class w0waCDM(wCDM, ABC):
    wa: _TN

    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN, Ob0: _TN = 0.,
                 w0: _TN = -1., wa: _TN = 0.): ...

class w0waCDMR(RadiationFLRWMixin, w0waCDM, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN, Ob0: _TN = 0.,
                 w0: _TN = -1., wa: _TN = 0.,
                 Or0: _TN = 0., Tcmb0: _GQuantity = _no_value): ...


class wpwaCDM(w0waCDM, ABC):
    zp: _TN

    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN, Ob0: _TN = 0.,
                 w0: _TN = -1., wa: _TN = 0., zp: _TN = 0., wp: _TN = _no_value): ...

    @property
    def ap(self): ...

class wpwaCDMR(RadiationFLRWMixin, wpwaCDM, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN, Ob0: _TN = 0.,
                 w0: _TN = -1., wa: _TN = 0., zp: _TN = 0., wp: _TN = _no_value,
                 Or0: _TN = 0., Tcmb0: _GQuantity = _no_value): ...


class w0wzCDM(wCDM, ABC):
    wz: _TN

    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN, Ob0: _TN = 0.,
                 w0: _TN = -1., wz: _TN = 0.): ...

class w0wzCDMR(RadiationFLRWMixin, w0wzCDM, ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN, Ob0: _TN = 0.,
                 w0: _TN = -1., wz: _TN = 0.,
                 Or0: _TN = 0., Tcmb0: _GQuantity = _no_value): ...
