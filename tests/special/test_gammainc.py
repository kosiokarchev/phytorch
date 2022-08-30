from typing import Callable

import torch.special
from hypothesis import assume, given, strategies as st
from pytest import mark
from scipy import special as sp
from torch import isclose, isnan, tensor, Tensor

from phytorch.special.gammainc import gammainccinv, gammaincinv
from tests.common.cases import BaseCasesTest
from tests.common.closeness import nice_and_close
from tests.common.dtypes import AllDtypeTest, DoubleDtypeTest
from tests.common.strategies.numbers import _real_number, _real_number_


st_gammaincinv_params = st.floats(max_value=1e10), st.floats(min_value=0, max_value=1)


class GammaincinvCases(AllDtypeTest, BaseCasesTest):
    test_case = BaseCasesTest.parametrize({
        gammaincinv: (
            # scipy/special/tests/test_basic.py
            # https://github.com/scipy/scipy/blob/abd7f74160f03205c2bd4f2ec0e10d067e82e632/scipy/special/tests/test_basic.py#L2141
            ((10, 2.5715803516000736e-20), 0.05),
            ((50, 8.20754777388471303050299243573393e-18), 11.0)
        )
    })


class GammaincinvBasicTest(AllDtypeTest):
    @staticmethod
    @given(_real_number, _real_number_(0, 1))
    def test_complement(a: float, p: float):
        assert nice_and_close(gammaincinv(a, p), gammainccinv(a, 1-p))

    @staticmethod
    @mark.parametrize('func, inv', (
        (torch.special.gammainc, gammaincinv), (torch.special.gammaincc, gammainccinv)))
    @given(_real_number_(None, 10), _real_number_(0, 10))
    def test_roundtrip(func: Callable[[Tensor, Tensor], Tensor], inv: Callable[[Tensor, Tensor], Tensor], a: float, x: float):
        a, x = map(torch.as_tensor, (a, x))
        res = func(a, x)
        assume(res not in (0, 1))
        assert nice_and_close(x, inv(a, res))


class TestGammaincinvAdvanced(DoubleDtypeTest):
    @staticmethod
    @mark.parametrize('ourfunc, theirfunc', (
        (gammaincinv, sp.gammaincinv), (gammainccinv, sp.gammainccinv)))
    @given(st.floats(max_value=1e10), st.floats(min_value=0, max_value=1))
    def test_vs_scipy(ourfunc, theirfunc, a: float, p: float):
        assume(a == (-float('inf')) or not (0 > a == int(a)))
        ress = ourfunc(a, p), tensor(theirfunc(a, p))
        assert all(map(isnan, ress)) or isclose(*ress)
