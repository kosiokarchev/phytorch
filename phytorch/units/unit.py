from __future__ import annotations

from fractions import Fraction
from itertools import chain
from math import isclose
from numbers import Number, Real
from operator import add, mul, neg
from typing import Any, cast, Iterable, TYPE_CHECKING, Union

from typing_extensions import Protocol, runtime_checkable, TypeAlias

from ..quantities import quantity
from ..utils._typing import _bop, _fractionable, _mop, upcast, ValueProtocol


class Dimension(str):
    pass


dimensions = tuple(map(Dimension, ('L', 'T', 'M', 'I', 'Θ')))  # type: tuple[Dimension, ...]
LENGTH, TIME, MASS, CURRENT, TEMPERATURE = dimensions


class UnitBase(dict):
    @classmethod
    def _make(cls, iterable: Iterable[tuple[Dimension, _fractionable]], **kwargs):
        return cls(((key, val) for key, val in iterable for val in [Fraction(val).limit_denominator()] if val != 0), **kwargs)

    def __missing__(self, key: Dimension):
        assert isinstance(key, Dimension),\
            f'Units can be indexed only by {Dimension.__name__} instances, '\
            f'got {key} ({type(key)})'
        return Fraction()

    def __repr__(self):
        return f'<{type(self).__name__}: {self!s}>'

    def __str__(self):
        return f'[{" ".join(f"{key}^({val})" for key, val in self.items())}]'

    def _operate_other(self, other, dim_op: _bop, **kwargs):
        return self._make(((key, dim_op(self[key], other[key])) for key in set(chain(self.keys(), other.keys()))), **kwargs)

    def _operate_self(self, dim_op: _mop, *args, **kwargs):
        return self._make(((key, dim_op(val, *args)) for key, val in self.items()), **kwargs)

    def __invert__(self, **kwargs):
        return self._operate_self(neg, **kwargs)

    def __pow__(self, power, modulo=None, **kwargs):
        if isinstance(power, Number):
            return self._operate_self(mul, power, **kwargs)
        return NotImplemented

    def __mul__(self, other: UnitBase, **kwargs):
        if isinstance(other, UnitBase):
            return self._operate_other(other, add, **kwargs)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        return self.__mul__(other)

    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def __truediv__(self, other):
        return self.__mul__(other**(-1))

    def __rtruediv__(self, other):
        return (~self).__mul__(other)

    __mod__ = __truediv__
    __rmod__ = __rtruediv__


class ValuedFloat(float):
    @property
    def value(self):
        return self


class Unit(UnitBase):
    def __init__(self, *args, value: Real = Fraction(1), name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = value
        self.name = name

        self.unit = self
        self.dimension = upcast(self, UnitBase)

    def set_name(self, name):
        self.name = name
        return self

    def to(self, other: Unit):
        if self.dimension != other.dimension:
            raise TypeError(f'Cannot convert {self}, aka {self.dimension}, to {other}, aka {other.dimension}')
        return ValuedFloat(self.value / other.value)

    def __str__(self):
        return self.name or f'{self.value} x {super().__str__()}'

    @property
    def bracketed_name(self):
        return f'({s})' if ' ' in (s := str(self)) else s

    def __invert__(self, **kwargs):
        return super().__invert__(value=1/self.value, name=f'{self.bracketed_name}^(-1)', **kwargs)

    def __pow__(self, power: _fractionable, modulo=None, **kwargs):
        return super().__pow__(power, modulo, value=self.value**power, name=f'{self.bracketed_name}^({Fraction(power).limit_denominator()})', **kwargs)

    def __mul__(self, other: _mul_other, **kwargs):
        if isinstance(other, Unit):
            return super().__mul__(other, value=self.value * other.value, name=f'{self!s} {other!s}', **kwargs)
        elif isinstance(other, Real):
            return self._make(self.items(), value=self.value * other, name=f'{other!s} {self.bracketed_name}', **kwargs)
        elif isinstance(other, quantity.GenericQuantity):
            return other.value * (other.unit * self)
        elif (cls := next((
            cls for cls in quantity.GenericQuantity._generic_quantity_subtypes.keys()
            if isinstance(other, cls)
        ), None)) is not None:
            return quantity.GenericQuantity._generic_quantity_subtypes[cls]._from_bare_and_unit(cast(cls, other), unit=self)
        return NotImplemented

    if TYPE_CHECKING:
        def __rmul__(self, other: _mul_other): ...

    def __eq__(self, other):
        return isinstance(other, Unit) and self.value == other.value and super().__eq__(other)

    def isclose(self, other, *args, **kwargs):
        return isinstance(other, Unit) and isclose(self.value, other.value, *args, **kwargs) and super().__eq__(other)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return next((
            t.__torch_function__(func, types, args, kwargs)
            for t in types if issubclass(t, quantity.GenericQuantity)
        ), NotImplemented)


_mul_other: TypeAlias = Union[Unit, Real, 'quantity.GenericQuantity']


@runtime_checkable
class Unitful(Protocol):
    value: Any
    unit: Unit

    def to(self, unit: Unit) -> ValueProtocol: ...
