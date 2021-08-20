from cmath import exp, inf, pi
from itertools import repeat

import torch
from hypothesis import strategies as st


JUST_FINITE = dict(allow_nan=False, allow_infinity=False)
BIG = 1e10
SMALL = 1e-9


def _real_number_(min_value=-BIG, max_value=BIG, **kwargs):
    return st.floats(min_value, max_value, **JUST_FINITE, **kwargs)


_real_number = _real_number_()
_real_numbers = (_real_number,)
_nonnegative_number = st.floats(0, BIG, **JUST_FINITE)
_nonnegative_numbers = (_nonnegative_number,)
_positive_number = st.floats(SMALL, BIG, **JUST_FINITE)
_positive_numbers = (_positive_number,)


def _complex_number_(min_magnitude=0., max_magnitude=inf, **kwargs):
    return st.complex_numbers(
        min_magnitude=min_magnitude, max_magnitude=max_magnitude,
        **JUST_FINITE, **kwargs)


_complex_number = _complex_number_(0, BIG)
_complex_numbers = (_complex_number,)
_nonzero_complex_number = _complex_number_(SMALL, BIG)


def _positive_complex_(real_part, imag_part=_real_number):
    return st.tuples(real_part, imag_part).map(lambda args: complex(*args))


_positive_complex = _positive_complex_(_positive_number)


def _cut_plane_(magnitude):
    return st.tuples(magnitude, st.floats(-1 + 1e-6, 1 - 1e-6, **JUST_FINITE)).map(
        lambda r_arg: r_arg[0] * exp(1j * pi * r_arg[1]))


_cut_plane = _cut_plane_(_nonnegative_number)
_nonzero_cut_plane = _cut_plane_(_positive_number)


def n_tensors_strategy(n, elements: st.SearchStrategy = st.floats(min_value=1e-4, max_value=1e3), max_len=10):
    return st.integers(min_value=1, max_value=max_len).flatmap(lambda m: st.tuples(
        *repeat(st.lists(elements, min_size=m, max_size=m), n)
    )).map(torch.tensor)


def n_complex_tensors_strategy(n, max_len=10, min_magnitude=1e-4, max_magnitude=1e3):
    return n_tensors_strategy(n, st.complex_numbers(min_magnitude=min_magnitude, max_magnitude=max_magnitude),
                              max_len=max_len)
