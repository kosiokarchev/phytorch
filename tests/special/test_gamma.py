from cmath import log, pi, sqrt

import mpmath as mp
import torch
from hypothesis import assume, given, strategies as st
from pytest import mark
from scipy import special as sp
from torch import isclose, isnan, tensor

from phytorch.special.gamma import digamma, gamma, loggamma, polygamma
from tests.common import BaseDoubleTest, close_complex_nan, make_dtype_tests, with_default_double


euler = float(mp.euler)
cstrategy = st.complex_numbers(max_magnitude=100, allow_nan=False, allow_infinity=False)


@with_default_double
@mark.parametrize('ourfunc, theirfunc', (
    (gamma, sp.gamma), (loggamma, sp.loggamma), (digamma, sp.digamma)
))
@given(cstrategy)
def test_gammas_vs_scipy(ourfunc, theirfunc, z: complex):
    ress = ourfunc(z), tensor(theirfunc(z))
    assert all(map(isnan, ress)) or isclose(*ress)


@with_default_double
@mark.parametrize('z, truth', (
    # scipy/special/test_digamma.py
    # https://github.com/scipy/scipy/blob/70c8d80bd8ce97ea935d95111c779545c0aeb21e/scipy/special/tests/test_digamma.py#L27
    (1, -euler),
    (0.5, -2*log(2) - euler),
    (1/3, -pi/(2*sqrt(3)) - 3*log(3)/2 - euler),
    (1/4, -pi/2 - 3*log(2) - euler),
    (1/6, -pi*sqrt(3)/2 - 2*log(2) - 3*log(3)/2 - euler),
    (1/8, -pi/2 - 4*log(2) - (pi + log(2 + sqrt(2)) - log(2 - sqrt(2)))/sqrt(2) - euler)
))
def test_digamma_vals(z, truth):
    assert isclose(res := digamma(z), tensor(truth, dtype=res.dtype))


@with_default_double
@mark.parametrize('n, z, truth', (
    # scipy/special/tests/test_basic.py (from Table 6.2 (pg. 271) of A&S)
    # https://github.com/scipy/scipy/blob/3c35f0a3abdd07c2d2b8f2f1528709d6a393498e/scipy/special/tests/test_basic.py#L3024
    (2, 1, -2.4041138063),
    (3, 1, 6.4939394023),
    # TODO: polygamma broadcasting on n
    (0, 0.5, -1.9635100260214238),
    (1, 1.5, 0.93480220054467933),
    (2, 2.5, -0.23620405164172739)
))
def test_polygamma_vals(n, z, truth):
    assert isclose(res := polygamma(n, z), tensor(truth, dtype=res.dtype))


@with_default_double
@given(st.integers(min_value=0, max_value=10), cstrategy)  # TODO: polygamma arbitrary n
def test_polygamma(n, z):
    try:
        assume(not isnan(ours := polygamma(n, z)))
        assert isclose(ours, tensor(complex(mp.polygamma(n, z))))
    except (ValueError, ZeroDivisionError):
        assume(False)


class GammasTest:
    @staticmethod
    @given(st.integers(min_value=-1_000_000, max_value=0))
    def test_infinity(n):
        assert isnan(gamma(n))
        assert isnan(loggamma(n))
        assert isnan(digamma(n))
        assert isnan(polygamma(2, n))

    @staticmethod
    @given(st.floats(min_value=-300, max_value=-1e-2, allow_nan=False, allow_infinity=False))
    def test_loggamma_branch(x):
        assume(not isnan(loggamma(x)))
        eps = torch.finfo(torch.get_default_dtype()).eps
        assert loggamma(complex(x, eps)) == loggamma(complex(x, -eps)).conj()

    @staticmethod
    @given(cstrategy)
    def test_digamma_polygamma(z):
        assert close_complex_nan(digamma(z), polygamma(0, z))


globals().update(make_dtype_tests((GammasTest,), 'Gamma'))


class TestGammaDouble(BaseDoubleTest):
    @staticmethod
    @given(cstrategy)
    def test_gamma_loggamma(z):
        close_complex_nan(loggamma(z).exp(), gamma(z))

    @staticmethod
    @given(cstrategy)
    def test_loggamma_funceq(z: complex):
        assume(abs(z-round(z.real)) > 1e-3)
        assert close_complex_nan(loggamma(z+1), log(z) + loggamma(z))
