from torch import Tensor
from ..utils._typing import _TN


def gamma(z: _TN) -> Tensor: ...
def loggamma(z: _TN) -> Tensor: ...
def digamma(z: _TN) -> Tensor: ...
def psi(z: _TN) -> Tensor: ...
def polygamma(n: int, z: _TN) -> Tensor: ...
