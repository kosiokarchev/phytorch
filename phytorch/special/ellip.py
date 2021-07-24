from functools import update_wrapper
from math import inf
from typing import Optional

from torch import Tensor

from .ellipr import elliprd, elliprf, elliprg, elliprj
from ..math import sin, where


# TODO: annotate Number/Tensor?
# TODO: differentiability (see https://dlmf.nist.gov/19.25.E6, https://dlmf.nist.gov/19.25.E12)
# TODO: move to c++ kernels?


def _get_c(phi: Optional[Tensor], c: Tensor = None):
    try:
        return c if c is not None else 1 / sin(phi)**2
    except ZeroDivisionError:
        return float('inf')
    except ValueError:
        return float('nan')


def ellipk(m: Tensor) -> Tensor:
    # https://dlmf.nist.gov/19.25.E1, line 1
    return elliprf(0, 1-m, 1)


def ellipe(m: Tensor) -> Tensor:
    # https://dlmf.nist.gov/19.25.E1, line 2
    # TODO: handle infinities
    return 2 * elliprg(0, 1-m, 1)


def ellipd(m: Tensor) -> Tensor:
    # https://dlmf.nist.gov/19.25.E1, line 4
    return elliprd(0, 1-m, 1) / 3


def ellippi(n: Tensor, m: Tensor) -> Tensor:
    # https://dlmf.nist.gov/19.25.E2
    # TODO: edge cases n=1 or m=1
    return ellipk(m) + n / 3 * elliprj(0, 1-m, 1, 1-n)


def ellipkinc(phi: Optional[Tensor], m: Tensor, c: Tensor = None) -> Tensor:
    # https://dlmf.nist.gov/19.25.E5
    return elliprf((c := _get_c(phi, c))-1, c-m, c)


def ellipeinc(phi: Optional[Tensor], m: Tensor, c: Tensor = None) -> Tensor:
    # https://dlmf.nist.gov/19.25.E9
    return where(~(((c := _get_c(phi, c))==1) & (m==1)), elliprf(c-1, c-m, c) - m * ellipdinc(phi, m, c), 1)

    # https://dlmf.nist.gov/19.25.E11, but NaNs when c==1
    # return ((c-m) / c / (c-1+eps))**0.5 - (1-m)/3 * elliprd(c-m, c, c-1)


def ellipdinc(phi: Optional[Tensor], m: Tensor, c: Tensor = None) -> Tensor:
    # https://dlmf.nist.gov/19.25.E13
    return elliprd((c := _get_c(phi, c))-1, c-m, c) / 3


def ellippiinc(n: Tensor, phi: Optional[Tensor], m: Tensor, c: Tensor = None) -> Tensor:
    # https://dlmf.nist.gov/19.25.E14
    return ellipkinc(phi, m, (c := _get_c(phi, c))) + n / 3 * elliprj(c-1, c-m, c, c-n)


ellipf = ellipkinc
