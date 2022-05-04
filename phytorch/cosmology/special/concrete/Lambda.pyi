from abc import ABC
from typing import TypeVar

from ..decdm import BaseDECDM, RadiationFLRWMixin
from ...core import H100
from ...utils import _GQuantity, _no_value
from ....utils._typing import _TN
from ....utils.interop import _astropy


_LambdaCDMT = TypeVar('_LambdaCDMT', bound='LambdaCDM')
_LambdaCDMRT = TypeVar('_LambdaCDMRT', bound='LambdaCDMR')

class LambdaCDM(BaseDECDM[_LambdaCDMT, _astropy.cosmology.LambdaCDM], ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN, Ob0: _TN = 0.): ...

class LambdaCDMR(RadiationFLRWMixin, LambdaCDM[_LambdaCDMRT], ABC):
    def __init__(self, *, H0=H100, Om0: _TN, Ode0: _TN, Ob0: _TN = 0.,
                 Or0: _TN = 0., Tcmb0: _GQuantity = _no_value): ...
