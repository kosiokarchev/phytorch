from fractions import Fraction
from itertools import starmap
from math import isclose, pi

from hypothesis import assume, given
from more_itertools import pairwise
from pytest import mark, raises

from phytorch.units.unit import Dimension, dimensions, Unit
from phytorch.utils._typing import ValueProtocol

from tests.common.strategies_units import units_strategy, values_strategy


# TODO:
#  - UnitBase constructor: handle 0-dims


@mark.parametrize('dim', dimensions)
def test_concrete_dimensions(dim):
    assert isinstance(dim, Dimension)
    assert hash(dim)


@given(units_strategy)
def test_make_unit_from_unit(u: Unit):
    assert u == u.value * Unit(u)
    assert (bu := Unit(u)).value == 1 and bu.dimension == u.dimension


@given(units_strategy)
def test_make_unit_dimensions(u: Unit):
    dims = dict(u)
    assert u.dimension == dims and not u == dims and not u == u.dimension
    assert all(u[d] == dims.get(d, 0) for d in dimensions)


@given(units_strategy, values_strategy)
def test_unit_arithmetic(u: Unit, val: float):
    for f in (lambda: val ** u, lambda: u + u, lambda: val - u):
        with raises(TypeError, match='unsupported operand'):
            f()

    assert all(starmap(isclose, pairwise(((u1 := val * u).value, (u * val).value, val * u.value)))) and u1.dimension == u.dimension
    assert isclose((u2 := u / val).value, u.value / val) and u2.dimension == u.dimension
    assert isclose((u3 := val / u).value, val / u.value) and u3.dimension == {key: -value for key, value in u.items()}
    assert isclose((u4 := ~u).value, 1 / u.value) and u4.dimension == u3.dimension

    assert isclose((u5 := u**val).value, u.value**val) and u5.dimension == {
        key: v for key, value in u.items()
        for v in [Fraction(value * val).limit_denominator()] if v != 0}

    assert isclose((u6 := u * u).value, u.value * u.value) and u6.dimension == {key: 2*value for key, value in u.items()}
    assert isclose((u7 := u / u).value, 1.) and not u7.dimension
    assert Unit.isclose(u / u**2, u4)


@given(units_strategy, values_strategy)
def test_unit_conversion(u: Unit, val: float):
    assume(u.value != 0)
    assert isinstance((to := (u1 := val * u).to(u)), ValueProtocol) and isclose(to, val)
    assert isclose((pi * u1).to(u), pi * val)


@given(units_strategy)
def test_unit_conversion_wrong(u: Unit):
    assume(u)
    with raises(TypeError, match='Cannot convert'):
        (u**2).to(u)
