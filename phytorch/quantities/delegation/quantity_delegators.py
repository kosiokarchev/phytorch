from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import reduce
from numbers import Integral, Real
from typing import Callable, Iterable, Literal, Optional, Union

from more_itertools import collapse

from .delegator import Delegator
from .. import quantity
from ...units.unit import Unit


@dataclass(eq=False)
class QuantityDelegatorBase(Delegator):
    incremental: bool = False

    _filter: Callable[[quantity.GenericQuantity, Iterable], Iterable]\
        = staticmethod(lambda self, qty, args: collapse(args, base_type=qty._T))

    def filter(self, qty: quantity.GenericQuantity, args: Iterable) -> Iterable:
        return self._filter(self, qty, args)

    def finalize_get(self, qty: quantity.GenericQuantity, func, unit, *args, **kwargs):
        with qty.delegator_context:
            # TODO: Nasty hack -> https://github.com/pytorch/pytorch/issues/54983
            args = tuple(float(arg) if isinstance(arg, Real) and not isinstance(arg, Integral)
                         else arg for arg in args)
            if self.func_takes_self:
                args = (qty,) + args
            if self.incremental:
                func(*args, **kwargs)
                qty.unit = unit
                return qty
            else:
                return qty._meta_update(func(*args, **kwargs), unit=unit)


@dataclass(eq=False)
class QuantityDelegator(QuantityDelegatorBase):
    in_unit: Optional[Unit] = None
    out_unit: Union[Literal[False], None, Unit, Callable[[Unit], Unit]] = None

    strict: bool = True

    _to: Callable[[quantity.GenericQuantity, Iterable, Unit], Iterable]\
        = staticmethod(lambda self, qty, args, unit: qty._to(args, unit, self.strict))

    def to(self, qty: quantity.GenericQuantity, args: Iterable, unit: Unit) -> Iterable:
        return self._to(self, qty, args, unit)

    def _get(self, func):
        def f(qty: quantity.GenericQuantity, *args, **kwargs):
            in_unit = self.in_unit if self.in_unit is not None else qty.unit
            if self.func_takes_self:
                args = (qty,) + args
            assert all(
                a.unit.dimension == in_unit.dimension
                if isinstance(a, quantity.GenericQuantity) and a.unit is not None
                else not (self.strict and in_unit)
                for a in self.filter(qty, args)
            ), f'All inputs have to be of dimension {in_unit.dimension}.'
            args = self.to(qty, args, in_unit)
            if self.func_takes_self:
                qty, *args = args
            return self.finalize_get(qty, func, self.out_unit, *args, **kwargs)
        return f


@dataclass(eq=False)
class ProductDelegator(QuantityDelegatorBase):
    op: Callable = operator.mul
    flip: bool = False

    def _get(self, func):
        def f(qty: quantity.GenericQuantity, *args, **kwargs):
            _func = func
            if len(args) == 1:
                if isinstance(arg := args[0], Unit):
                    otherunit = arg
                    args = (1,)
                else:
                    otherunit = arg.unit if isinstance(arg, quantity.GenericQuantity) else Unit()
                unit = self.op(*((otherunit, qty.unit) if self.flip else (qty.unit, otherunit)))
            else:
                unit = reduce(self.op, (a.unit for a in self._filter(qty, args)
                                        if isinstance(a, quantity.GenericQuantity)))
            return self.finalize_get(qty, _func, unit, *args, **kwargs)
        return f


class PowerDelegator(QuantityDelegatorBase):
    def _get(self, func):
        def f(qty: quantity.GenericQuantity, other, *args, **kwargs):
            try:
                other = float(other)
                return self.finalize_get(qty, func, qty.unit**other, other, *args, **kwargs)
            except ValueError:
                raise AssertionError('Can only raise quantity to a scalar power.')
        return f
