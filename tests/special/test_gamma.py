from cmath import log, pi, sqrt
from collections import Mapping
from numbers import Number
from typing import Any, Union

import mpmath as mp
import numpy as np
import torch
from hypothesis import assume, given, strategies as st
from pytest import mark
from scipy import special as sp
from scipy.special._mptestutils import exception_to_nan
from torch import allclose, isclose, isnan, tensor

from phytorch.special.gamma import digamma, gamma, loggamma, polygamma
from tests.common import BaseDoubleTest, BaseDtypeTest, close_complex_nan, make_dtype_tests, with_default_double


euler = float(mp.euler)
cstrategy = st.complex_numbers(max_magnitude=100, allow_nan=False, allow_infinity=False)


cases: Mapping[Any, tuple[tuple[Union[Number, int], ...], ...]] = {
    digamma: (
        # scipy/special/test_digamma.py
        # https://github.com/scipy/scipy/blob/70c8d80bd8ce97ea935d95111c779545c0aeb21e/scipy/special/tests/test_digamma.py#L27
        (1, -euler),
        (0.5, -2 * log(2) - euler),
        (1 / 3, -pi / (2 * sqrt(3)) - 3 * log(3) / 2 - euler),
        (1 / 4, -pi / 2 - 3 * log(2) - euler),
        (1 / 6, -pi * sqrt(3) / 2 - 2 * log(2) - 3 * log(3) / 2 - euler),
        (1 / 8, -pi / 2 - 4 * log(2) - (pi + log(2 + sqrt(2)) - log(2 - sqrt(2))) / sqrt(2) - euler)),
    polygamma: (
        # scipy/special/tests/test_basic.py (from Table 6.2 (pg. 271) of A&S)
        # https://github.com/scipy/scipy/blob/3c35f0a3abdd07c2d2b8f2f1528709d6a393498e/scipy/special/tests/test_basic.py#L3024
        (2, 1, -2.4041138063),
        (3, 1, 6.4939394023),
        # TODO: polygamma broadcasting on n
        (0, 0.5, -1.9635100260214238),
        (1, 1.5, 0.93480220054467933),
        (2, 2.5, -0.23620405164172739))
}


class GammasTest(BaseDtypeTest):
    @staticmethod
    @given(st.integers(min_value=-1_000_000, max_value=0))
    def test_infinity(n):
        assert isnan(gamma(n))
        assert isnan(loggamma(n))
        assert isnan(digamma(n))
        assert isnan(polygamma(2, n))

    @given(st.floats(min_value=-300, max_value=-1e-2, allow_nan=False, allow_infinity=False))
    def test_loggamma_branch(self, eps, x):
        assume(not isnan(loggamma(x)))
        eps = torch.finfo(torch.get_default_dtype()).eps
        assert loggamma(complex(x, eps)) == loggamma(complex(x, -eps)).conj()

    @mark.parametrize('z, truth', cases[digamma])
    def test_digamma_vals(self, z, truth):
        assert isclose(res := digamma(z), tensor(truth, dtype=res.dtype))

    @mark.parametrize('n, z, truth', cases[polygamma])
    def test_polygamma_vals(self, n, z, truth):
        assert isclose(res := polygamma(n, z), tensor(truth, dtype=res.dtype))

    @staticmethod
    @given(cstrategy)
    def test_digamma_polygamma(z):
        assert close_complex_nan(digamma(z), polygamma(0, z))

    def test_digamma_roots(self, eps):
        # from scipy/tests/special/test_mpmath.py
        # https://github.com/scipy/scipy/blob/86b27edca8306da6c9b3429e87b342b35200cbe7/scipy/special/tests/test_mpmath.py#L428
        z = tensor([1.4616321449683623, -0.50408300826445541]) + (
            torch.complex(*(tensor(d, dtype=self.dtype) for d in np.meshgrid(*2 * (
                np.r_[-0.24, -np.logspace(-1, -15, 10), 0, np.logspace(-15, -1, 10), 0.24],))))
        )[..., None]
        truth = np.vectorize(mp.digamma, otypes=(complex,))(z)

        assert allclose(digamma(z), tensor(truth, dtype=z.dtype), atol=10*eps)

    def test_digamma_boundary(self):
        # from scipy/tests/special/test_mpmath.py
        # https://github.com/scipy/scipy/blob/86b27edca8306da6c9b3429e87b342b35200cbe7/scipy/special/tests/test_mpmath.py#L470
        z = torch.complex(*(tensor(_, dtype=self.dtype) for _ in np.meshgrid(
            -np.logspace(10, -30, 100),  # TODO: range in scipy is bigger
            [-6.1, -5.9, 5.9, 6.1]
        ))).flatten()

        truth = np.vectorize(exception_to_nan(mp.digamma), otypes=(complex,))(z)

        assert allclose(digamma(z), tensor(truth, dtype=z.dtype))


globals().update(make_dtype_tests((GammasTest,), 'Gamma'))


class TestGammaDouble(BaseDoubleTest):
    @staticmethod
    @mark.parametrize('ourfunc, theirfunc', (
        (gamma, sp.gamma), (loggamma, sp.loggamma), (digamma, sp.digamma)
    ))
    @given(cstrategy)
    def test_gammas_vs_scipy(ourfunc, theirfunc, z: complex):
        ress = ourfunc(z), tensor(theirfunc(z))
        assert all(map(isnan, ress)) or isclose(*ress)

    @staticmethod
    @with_default_double
    @given(st.integers(min_value=0, max_value=10), cstrategy)  # TODO: polygamma arbitrary n
    def test_polygamma_vs_mpmath(n, z):
        try:
            assume(not isnan(ours := polygamma(n, z)))
            assert isclose(ours, tensor(complex(mp.polygamma(n, z))))
        except (ValueError, ZeroDivisionError):
            assume(False)

    @staticmethod
    @given(cstrategy)
    def test_gamma_loggamma(z):
        close_complex_nan(loggamma(z).exp(), gamma(z))

    @staticmethod
    @given(cstrategy)
    def test_loggamma_funceq(z: complex):
        assume(abs(z-round(z.real)) > 1e-3)
        assert close_complex_nan(loggamma(z+1), log(z) + loggamma(z))

    def test_digamma_negreal(self, eps):
        # from scipy/tests/special/test_mpmath.py
        # https://github.com/scipy/scipy/blob/86b27edca8306da6c9b3429e87b342b35200cbe7/scipy/special/tests/test_mpmath.py#L450
        z = torch.complex(*(tensor(_, dtype=self.dtype) for _ in np.meshgrid(
            -np.logspace(3, -30, 10),  # TODO: range in scipy is bigger
            np.r_[-np.logspace(0, -3, 5), np.logspace(-3, 0, 5)]
        ))).flatten()

        truth = np.vectorize(exception_to_nan(mp.digamma), otypes=(complex,))(z)

        assert allclose(digamma(z), tensor(truth, dtype=z.dtype))
