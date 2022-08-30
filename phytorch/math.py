import cmath
from cmath import pi

import torch

from .utils.complex import to_complex


def realise(x):
    return x if torch.is_tensor(x) and not torch.is_complex(x) else x.real


def conjugate(x):
    return (x.conj() if torch.is_complex(x) else x) if torch.is_tensor(x) else x.conjugate()


def complexify(x):
    return to_complex(x) if torch.is_tensor(x) else complex(x)


def where(cond, x, y):
    y = torch.as_tensor(y, dtype=x.dtype, device=x.device) if torch.is_tensor(x) else y
    return (x.where(cond, torch.as_tensor(y, dtype=x.dtype, device=x.device))
            if torch.is_tensor(cond) else (x if cond else y))


def sinc(x):
    if torch.is_tensor(x):
        return torch.sinc(x)
    x = x * pi
    return cmath.sin(x) / x if x != 0 else 1


def csinc(x, eps=1e-8):
    return where(abs(x) > eps, (1 - sinc(x)) / (x*pi)**2, 1/6)


def _overload(torchfunc, cmathfunc):
    def f(x):
        return (torchfunc if torch.is_tensor(x) else cmathfunc)(x)
    return f


exp = _overload(torch.exp, cmath.exp)
log = _overload(torch.log, cmath.log)
log10 =_overload(torch.log10, cmath.log10)
sin = _overload(torch.sin, cmath.sin)
cos = _overload(torch.cos, cmath.cos)
asin = _overload(torch.asin, cmath.asin)
acos = _overload(torch.acos, cmath.acos)
sinh = _overload(torch.sinh, cmath.sinh)
cosh = _overload(torch.cosh, cmath.cosh)
asinh = _overload(torch.asinh, cmath.asinh)
acosh = _overload(torch.acosh, cmath.acosh)
