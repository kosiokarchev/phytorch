from functools import reduce, update_wrapper
from operator import and_

import torch
from pytest import fixture
from hypothesis import assume, strategies as st


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


class BaseFloatTest(BaseDtypeTest):
    dtype = torch.float
    cdtype = torch.cfloat
    name = 'Float'


class BaseDoubleTest(BaseDtypeTest):
    dtype = torch.double
    cdtype = torch.cdouble
    name = 'Double'


def make_dtype_tests(bases, name):
    return {_name: type(_name, bases+(cls,), {})
            for cls in (BaseFloatTest, BaseDoubleTest)
            for _name in [f'Test{cls.name}{name}']}


CLOSE_KWARGS = dict(atol=1e-6, rtol=1e-6)
JUST_FINITE = dict(allow_nan=False, allow_infinity=False)


def nice_and_close(a, b, close_func=torch.allclose, **kwargs):
    b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    assume(not torch.isnan(a) and not torch.isnan(b))
    return close_func(a, b, **{
        'atol': max(1e-8, 100*torch.finfo(a.dtype).eps),
        'rtol': max(1e-5, 100*torch.finfo(a.dtype).eps),
        **kwargs})


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
