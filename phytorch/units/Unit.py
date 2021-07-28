from __future__ import annotations

from fractions import Fraction
from itertools import chain
from numbers import Number, Real
from operator import add, mul, neg
from typing import Generic, ItemsView, Iterable, Type, Union

from .. import quantities
from ..utils._typing import _bop, _fractionable, _mop, _T, ValueProtocol


class Dimension(str):
    pass


LENGTH, TIME, MASS, CURRENT, TEMPERATURE = map(Dimension, ('L', 'T', 'M', 'I', 'Θ'))  # type: Dimension


class UnitBase(dict):
    # TODO: PyCharm bug https://youtrack.jetbrains.com/issue/PY-38897
    @classmethod
    def _make(cls, iterable: Union[Iterable[tuple[Dimension, _fractionable]], ItemsView[Dimension, _fractionable]], **kwargs):
        return cls(((key, val) for key, val in iterable for val in [Fraction(val).limit_denominator()] if val != 0), **kwargs)

    def __missing__(self, key: Dimension):
        assert isinstance(key, Dimension),\
            f'Units can be indexed only by {Dimension.__name__} instances, '\
            f'got {key} ({type(key)})'
        return self.get(key, Fraction())

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
        else:
            return NotImplemented

    def __mul__(self, other, **kwargs):
        if isinstance(other, Unit):
            return self._operate_other(other, add, **kwargs)
        elif not isinstance(other, quantities.quantity.GenericQuantity) and (cls := next((
            cls for cls in quantities.quantity.GenericQuantity._generic_quantiy_subtypes.keys()
            if isinstance(other, cls)
        ), None)) is not None:
            return quantities.quantity.GenericQuantity._generic_quantiy_subtypes[cls]._from_bare_and_unit(other, unit=self)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(other**(-1))

    def __rtruediv__(self, other):
        return (~self).__mul__(other)

    __mod__ = __truediv__
    __rmod__ = __rtruediv__


class EqualityWrapper(Generic[_T]):
    def __init__(self, wrapped: _T, cls: Type[_T]):
        self.wrapped = wrapped
        self.cls = cls

    def __eq__(self, other: _T):
        return self.cls.__eq__(self.wrapped, other.wrapped if isinstance(other, EqualityWrapper) else other)

    def __repr__(self):
        return self.cls.__repr__(self.wrapped)

    def __str__(self):
        return self.cls.__str__(self.wrapped)


class ValuedFloat(float):
    @property
    def value(self):
        return self


class Unit(UnitBase, ValueProtocol):
    def __init__(self, *args, value: Real = Fraction(1), name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = value
        self.name = name

        self.dimension = EqualityWrapper(self, UnitBase)

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
        s = str(self)
        return f'({s})' if ' ' in s else s

    def __invert__(self, **kwargs):
        return super().__invert__(value=1/self.value, name=f'{self.bracketed_name}^(-1)', **kwargs)

    def __pow__(self, power: _fractionable, modulo=None, **kwargs):
        return super().__pow__(power, modulo, value=self.value**power, name=f'{self.bracketed_name}^({Fraction(power).limit_denominator()})', **kwargs)

    def __mul__(self, other, **kwargs):
        if isinstance(other, Unit):
            return super().__mul__(other, value=self.value * other.value, name=f'{self!s} {other!s}', **kwargs)
        elif isinstance(other, Real):
            return self._make(self.items(), value=self.value * other, name=f'{other!s} {Unit.__str__(self)}', **kwargs)
        else:
            return super().__mul__(other, **kwargs)

    def __eq__(self, other):
        return isinstance(other, Unit) and self.value == other.value and super().__eq__(other)
