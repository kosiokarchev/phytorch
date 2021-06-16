import cmath
from cmath import pi

import torch


def realise(x):
    return x if torch.is_tensor(x) and not torch.is_complex(x) else x.real


def where(cond, x, y):
    return (x.where(cond, torch.as_tensor(y, dtype=x.dtype, device=x.device))
            if torch.is_tensor(cond) else (x if cond else y))


def sinc(x):
    if torch.is_tensor(x):
        return torch.sinc(x)
    x = x * pi
    return cmath.sin(x) / x if x != 0 else 0


def csinc(x, eps=1e-8):
    return where(x < eps, (1 - sinc(x)) / x**2, 1/6)


def _overload(torchfunc, cmathfunc):
    def f(x):
        return (torchfunc if torch.is_tensor(x) else cmathfunc)(x)
    return f


# noinspection PyRedeclaration
exp = _overload(torch.exp, cmath.exp)
# noinspection PyRedeclaration
log10 =_overload(torch.log10, cmath.log10)
