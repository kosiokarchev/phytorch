from torch import Tensor

def gamma(z: Tensor) -> Tensor: ...
def loggamma(z: Tensor) -> Tensor: ...
def digamma(z: Tensor) -> Tensor: ...
def polygamma(n: int, z: Tensor) -> Tensor: ...

def hyp2f1(a: Tensor, b: Tensor, c: Tensor, z: Tensor) -> Tensor: ...

def deta1(z: Tensor) -> Tensor: ...
def zeta(z: Tensor) -> Tensor: ...