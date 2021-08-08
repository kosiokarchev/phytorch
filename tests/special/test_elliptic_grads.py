from itertools import combinations
from typing import Callable, Iterable, Union

from hypothesis import assume, given, strategies as st
from more_itertools import always_iterable
from phytorch.extensions.elliptic import elliprj
from pytest import mark
from torch import Tensor, tensor
from torch.autograd import gradcheck, gradgradcheck

from phytorch.special.elliptic import *
from phytorch.utils.complex import get_default_complex_dtype
from tests.common import _cut_plane_, _positive_real_complex_, with_default_double
from tests.special.test_ellipr import ELLIPR_FUNCMAP


def cgtensors(args, **kwargs):
    return tuple(
        tensor(arg, dtype=get_default_complex_dtype(), requires_grad=True, **kwargs)
        if arg is not None else None
        for arg in always_iterable(args)
    )


@with_default_double
@mark.parametrize('func, args', (
    (ellipk, 0.4), (ellipe, 0.4), (ellipd, 0.4), (ellippi, (0.7, 0.4)),
    (ellipkinc, (1.1, 0.4)), (ellipkinc, (None, 0.4, 2.5)),
    (ellipeinc, (1.1, 0.4)), (ellipeinc, (None, 0.4, 2.5)),
    (ellipdinc, (1.1, 0.4)), (ellipdinc, (None, 0.4, 2.5)),
    (ellippiinc, (0.7, 1.1, 0.4)), (ellippiinc, (0.7, None, 1.1, 2.5)),
    (elliprc, (1, 2)),
    (elliprd, (1, 2, 3)), (elliprf, (1, 2, 3)),
    (elliprg, (1, 2, 3)), (elliprj, (1, 2, 3, 4))
))
def test_grads_work_at_all(func: Callable, args: Union[Tensor, Iterable[Tensor]]):
    args = cgtensors(args)
    assert gradcheck(func, args)
    assert gradgradcheck(func, args)


@with_default_double
@mark.parametrize('func, _, nargs', ELLIPR_FUNCMAP[:-1])
@given(*3*(_cut_plane_(st.floats(1e-2, 1e2)),))
def test_ellipr_grads(func, nargs, _, x, y, z):
    args = (x, y, z)[:nargs]
    for _1, _2 in combinations(args, 2):
        assume(abs(_1-_2) > 1e-2)

    args = (*tensor(args, dtype=get_default_complex_dtype(), requires_grad=True),)
    assert gradcheck(func, args, raise_exception=False)
    assert gradgradcheck(func, args, raise_exception=False)


@with_default_double
@given(*4*(_positive_real_complex_(st.floats(1e-2, 1e2)),))
def test_elliprj_grad(x, y, z, p):
    args = x, y, z, p

    for _1, _2 in combinations(args, 2):
        assume(abs(_1-_2) > 1e-2)
    args = (*tensor(args, dtype=get_default_complex_dtype(), requires_grad=True),)

    assert gradcheck(elliprj, args, raise_exception=False)
    assert gradgradcheck(elliprj, args, raise_exception=False)
