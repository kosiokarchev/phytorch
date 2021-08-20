from cmath import isclose
from functools import partial
from itertools import chain, product
from numbers import Complex, Number, Real
from typing import Iterable, Sequence, Union

import mpmath as mp
from hypothesis import assume, given, strategies as st
from more_itertools import collapse
from pytest import fixture, param

from phytorch.special.elliptic_reduction.mp import MPEllipticReduction as ER
from tests.common.closeness import distinct
from tests.common.strategies import _cut_plane_, _positive_complex_, _real_number_


SMALL = 1e-3
BIG = 1e3


_reasonable_positive_real = _real_number_(SMALL, BIG)
_reasonable_cut_plane = _cut_plane_(_reasonable_positive_real)
_reasonable_positive_complex = _positive_complex_(_reasonable_positive_real, _real_number_(-BIG, BIG))


class BaseERTest:
    h: int
    n: int = None

    @staticmethod
    def _test_assertion(ours, theirs, err, rtol, atol):
        return isclose(ours, theirs, rel_tol=rtol, abs_tol=max(atol, 2*err))

    def _test(self, args, m, z1, z2, rtol=1e-5, atol=1e-8):
        er = ER(z2, z1, args, h=self.h)

        ours = complex(er.Im(m))
        theirs, err = mp.quad(partial(er.v, m), (er.y, er.x), error=True)

        assertion = self._test_assertion(ours, theirs, err, rtol, atol)

        if not assertion:
            print(m, args, z1, z2, ours, theirs)

        assert assertion

    # noinspection PyMethodMayBeStatic
    def m(self, request):
        return request.param

    mcases: Sequence[Union[param, Iterable[Number]]] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.n is not None and cls.mcases is None:
            cls.mcases = tuple(
                param(args, id=name)
                for args, name in (
                    ((*(i - 1) * (0,), q, *(cls.n - i) * (0,)),
                     f'Iqe({q},{i})' if (abs(q) > 1) else f'Ie({q * i})')
                    for q, i in chain(
                        ((0, 1),), *(
                            ((q, i), ((1 if (q > 0) else -1) * 3, i))
                            for q, i in product((1, -1), range(1, cls.n + 1)))
                    )
                )
            )
            cls.m = fixture(BaseERTest.m, scope='class', params=cls.mcases)


class ER3Test(BaseERTest):
    h = 3


class ER4Test(BaseERTest):
    h = 4


class BasicERTest(BaseERTest):
    basic_args: Sequence[Number]
    basic_interval: tuple[Real, Real]

    def test_basic(self, m):
        self._test(self.basic_args, m[:self.h], *self.basic_interval)
        self._test(self.basic_args, m[:self.h], *self.basic_interval[::-1])


class TestER3Basic(BasicERTest, ER3Test):
    n = 3
    basic_args = 1, 2, 3
    basic_interval = 3, 17


class TestER4Basic(BasicERTest, ER4Test):
    n = 4
    basic_args = 1, 2, 3, 4
    basic_interval = 3, 17


class StressfulERTest(BaseERTest):
    m_strategy = st.integers(-2, 3)
    bounds_strategies = 2*(_reasonable_positive_real,)

    _args_strategy: st.SearchStrategy[Sequence[Number]]
    args_strategy: Iterable[st.SearchStrategy[Number]] = None
    # TODO: S.real <= 0
    extrargs_strategy: st.SearchStrategy[Number] = _reasonable_positive_real

    @staticmethod
    def args(*args: Number) -> Sequence[Number]:
        return tuple(collapse((a, a.conjugate()) if isinstance(a, Complex) and not isinstance(a, Real) else a for a in args))

    def test_stress(self, m, args, z1, z2):
        assume(distinct(*args, d_min=SMALL))
        assume(abs(z1-z2) >= SMALL)
        self._test(args, m, z1, z2, rtol=1e-2, atol=1e-2)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.args_strategy is not None:
            cls._args_strategy = st.tuples(
                st.tuples(*cls.args_strategy).map(lambda args: cls.args(*args)),
                st.tuples(*(cls.n - cls.h)*(cls.extrargs_strategy,))
            ).map(partial(sum, start=()))
            cls.test_stress = given(cls._args_strategy, *cls.bounds_strategies)(StressfulERTest.test_stress)
            cls.test_stress_m = given(
                st.tuples(*cls.n*(cls.m_strategy,)),
                cls._args_strategy, *cls.bounds_strategies
            )(StressfulERTest.test_stress)


class BaseER3StressfulTest(StressfulERTest, ER3Test):
    n = 5


class TestER3StressfulOneReal(BaseER3StressfulTest):
    # TODO: could this be _reasonable_cut_plane?
    args_strategy = _reasonable_positive_real, _reasonable_positive_complex


class TestER3StressfulAllReal(BaseER3StressfulTest):
    args_strategy = 3*(_reasonable_positive_real,)


class BaseER4StressfulTest(StressfulERTest, ER4Test):
    n = 6


class TestER4StressfulAllComplex(BaseER4StressfulTest):
    # TODO: could this be _reasonable_cut_plane?
    args_strategy = 2*(_reasonable_positive_complex,)


class TestER4StressfulTwoReal(BaseER4StressfulTest):
    # TODO: could this be _reasonable_cut_plane?
    args_strategy = _reasonable_positive_real, _reasonable_positive_real, _reasonable_positive_complex

    @staticmethod
    def _test_assertion(ours, theirs, err, rtol, atol):
        # TODO: are these legit?
        return isclose(ours.real, theirs.real, rel_tol=rtol, abs_tol=max(atol, 2*err))


class TestER4StressfulAllReal(BaseER4StressfulTest):
    args_strategy = 4*(_reasonable_positive_real,)

    @staticmethod
    def _test_assertion(ours, theirs, err, rtol, atol):
        # TODO: are these legit?
        return isclose(ours.real, theirs.real, rel_tol=rtol, abs_tol=max(atol, 2*err))
