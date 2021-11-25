from typing import Iterable

import torch
from _pytest.python_api import raises
from hypothesis import given
from torch import Tensor

from phytorch.quantities.tensor_quantity import TensorQuantity
from phytorch.units._si.base import meter, second
from phytorch.units.exceptions import UnitError
from phytorch.units.unit import Unit

from tests.common import are_same_view
from tests.common.strategies.tensors import n_broadcastable_random_tensors, random_tensors
from tests.common.strategies.units import units_strategy
from tests.quantities.quantity_utils import _VT, ConcreteQuantity


def assert_has_value_and_unit(q: ConcreteQuantity, value: _VT, unit: Unit):
    assert (q.value == value).all()
    assert q.unit == unit


@given(random_tensors, units_strategy)
def test_init(value: _VT, unit: Unit):
    def _test(q: ConcreteQuantity):
        assert are_same_view(q, value, q.value)
        assert q.unit is unit

    _test(TensorQuantity(value, unit=unit))
    _test(value * unit)
    _test(unit * value)

    assert are_same_view((q := value / unit), value, q.value) and q.unit == ~unit
    assert ((q := unit / value).value == 1 / value).all() and q.unit is unit

    assert ((value * unit) * unit).unit == unit * unit
    assert ((value * unit) / unit).unit == unit / unit


@given(n_broadcastable_random_tensors(2), units_strategy, units_strategy)
def test_basic_arithmetic(values: Iterable[ConcreteQuantity], unit1: Unit, unit2: Unit):
    v1, v2 = values
    q1, q12, q2 = v1 * unit1, v2 * unit1, v2 * unit2
    assert_has_value_and_unit(q1 + q12, q1.value + q12.value, q1.unit)
    assert_has_value_and_unit(q1 - q12, q1.value - q12.value, q1.unit)
    assert_has_value_and_unit(q1 * q2, q1.value * q2.value, q1.unit * q2.unit)
    assert_has_value_and_unit(q1 / q2, q1.value / q2.value, q1.unit / q2.unit)
    assert_has_value_and_unit(q1 ** 2, q1.value**2, q1.unit ** 2)


@given(random_tensors)
def test_comparisons(q: Tensor):
    q_ = q/2 * (2*second)
    qm = q * meter
    q = q * second

    assert (q == q_).all()
    assert torch.allclose(q, q_)
    assert torch.isclose(q, q_).all()

    with raises(UnitError):
        _ = q == qm
    with raises(UnitError):
        torch.allclose(q, qm)
    with raises(UnitError):
        torch.isclose(q, qm)