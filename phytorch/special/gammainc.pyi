from torch import Tensor
from ..utils._typing import _TN

def Tgamma(m: int, a: _TN, z: _TN) -> Tensor: ...
def gammainc(a: _TN, x: _TN) -> Tensor: ...
def gammaincc(a: _TN, x: _TN) -> Tensor: ...
def gammaincinv(a: _TN, p: _TN) -> Tensor: ...
def gammainccinv(a: _TN, p: _TN) -> Tensor: ...