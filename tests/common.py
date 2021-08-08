import sys
from cmath import exp, inf, pi
from functools import reduce, update_wrapper
from numbers import Number
from operator import and_
from typing import Callable, Iterable, Mapping

import torch
from hypothesis import assume, strategies as st
from pytest import fixture, mark
from torch import Tensor


JUST_FINITE = dict(allow_nan=False, allow_infinity=False)
BIG = 1e10
SMALL = 1e-9

_real_number = st.floats(-BIG, BIG, **JUST_FINITE)
_nonnegative_number = st.floats(0, BIG, **JUST_FINITE)
_positive_number = st.floats(SMALL, BIG, **JUST_FINITE)
_positive_numbers = (_positive_number,)


def _complex_number_(min_magnitude=0., max_magnitude=inf, **kwargs):
    return st.complex_numbers(
        min_magnitude=min_magnitude, max_magnitude=max_magnitude,
        **JUST_FINITE, **kwargs)


_complex_number = _complex_number_(0, BIG)
_complex_numbers = (_complex_number,)
_nonzero_complex_number = _complex_number_(SMALL, BIG)


def _positive_real_complex_(real_part):
    return st.tuples(real_part, _real_number).map(lambda args: complex(*args))


_positive_real_complex = _positive_real_complex_(_positive_number)


def _cut_plane_(magnitude):
    return st.tuples(magnitude, st.floats(-1+1e-6, 1-1e-6, **JUST_FINITE)).map(
        lambda r_arg: r_arg[0] * exp(1j*pi*r_arg[1]))


_cut_plane = _cut_plane_(_nonnegative_number)
_nonzero_cut_plane = _cut_plane_(_positive_number)


def with_default_double(func):
    def f(*args, **kwargs):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.double)
        ret = func(*args, **kwargs)
        torch.set_default_dtype(default_dtype)
        return ret
    return update_wrapper(f, func)


class BaseDtypeTest:
    dtype: torch.dtype
    cdtype: torch.dtype
    name: str

    @fixture(autouse=True, scope='class')
    def _set_default_dtype(self):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        yield
        torch.set_default_dtype(previous_dtype)

    @fixture(scope='class')
    def eps(self):
        return torch.finfo(self.dtype).eps


class ConcreteDtypeTest(BaseDtypeTest):
    pass


@mark.xfail(reason='8bit\'s dead, baby')
class FloatDtypeTest(ConcreteDtypeTest):
    dtype = torch.float
    cdtype = torch.cfloat
    name = 'Float'


class DoubleDtypeTest(ConcreteDtypeTest):
    dtype = torch.double
    cdtype = torch.cdouble
    name = 'Double'


class AllDtypeTest(BaseDtypeTest):
    def __init_subclass__(cls, **kwargs):
        if not issubclass(cls, ConcreteDtypeTest):
            vars(sys.modules[cls.__module__]).update(make_dtype_tests((cls,), cls.__name__.removesuffix('Test')))


def make_dtype_tests(bases, name):
    return {_name: type(_name, bases+(cls,), {})
            for cls in (FloatDtypeTest, DoubleDtypeTest)
            for _name in [f'Test{cls.name}{name}']}


CLOSE_KWARGS = dict(atol=1e-6, rtol=1e-6)


def close(a, b, close_func=torch.allclose, **kwargs):
    b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    if torch.is_complex(a) and kwargs.get('equal_nan', True):
        return (close(a.real, b.real, close_func, equal_nan=True, **kwargs) and
                close(a.imag, b.imag, close_func, equal_nan=True, **kwargs))
    return close_func(a, b, **{
        'atol': max(1e-8, 100 * torch.finfo(a.dtype).eps),
        'rtol': max(1e-5, 100 * torch.finfo(a.dtype).eps),
        **kwargs})


def nice_and_close(a, b, close_func=torch.allclose, **kwargs):
    b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    assume(not torch.isnan(a) and not torch.isnan(b))
    return close(a, b, close_func, **kwargs, equal_nan=False)


def close_complex_nan(a, b, close_func=torch.allclose, accs=(torch.abs, torch.angle), **kwargs):
    return reduce(and_, (
        close_func(acc(a), acc(b), **{**CLOSE_KWARGS, 'equal_nan': True, **kwargs})
        for acc in accs))


def n_tensors_strategy(n, elements: st.SearchStrategy = st.floats(min_value=1e-4, max_value=1e3), max_len=10):
    return st.integers(min_value=1, max_value=max_len).flatmap(lambda m: st.tuples(*(
        st.lists(elements, min_size=m, max_size=m) for i in range(n)
    ))).map(torch.tensor)


def n_complex_tensors_strategy(n, max_len=10, min_magnitude=1e-4, max_magnitude=1e3):
    return n_tensors_strategy(n, st.complex_numbers(min_magnitude=min_magnitude, max_magnitude=max_magnitude), max_len=max_len)


def process_cases(func, vals):
    ins, outs = map(torch.tensor, zip(*vals))
    return ins, outs, func(*ins.T)


class BaseCasesTest:
    @staticmethod
    def parametrize(cases: Mapping[Callable[..., Tensor], Iterable[tuple[Iterable, Number]]]):
        return mark.parametrize('func, args, truth', tuple(
            (func, args, truth)
            for func, vals in cases.items() for args, truth in vals
        ))(BaseCasesTest.test_case)

    def test_case(self, func, args, truth):
        assert close_complex_nan(res := func(*args), torch.tensor(truth, dtype=res.dtype)), res.item()
