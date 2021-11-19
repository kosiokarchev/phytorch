from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import reduce
from inspect import getattr_static
from numbers import Integral, Number, Real
from typing import Callable, Iterable, Literal, Union

from more_itertools import all_equal, collapse, first
from typing_extensions import TypeAlias

from .delegator import Delegator
from .. import quantity
from ...units.exceptions import UnitError
from ...units.unit import Unit


__unitT: TypeAlias = Union[Literal[False], None, Unit, Callable[[Unit], Unit]]
_unitT = Union[__unitT, Iterable[__unitT]]


@dataclass(eq=False)
class QuantityDelegatorBase(Delegator):
    incremental: bool = False

    _filter: Callable[[quantity.GenericQuantity, Iterable], Iterable]\
        = staticmethod(lambda self, qty, args: collapse(args, base_type=qty._T))

    def filter(self, qty: quantity.GenericQuantity, args: Iterable) -> Iterable:
        return self._filter(self, qty, args)

    def finalize_get(self, qty: quantity.GenericQuantity, func, unit: _unitT, *args, **kwargs):
        with qty.delegator_context:
            # TODO: Nasty hack -> https://github.com/pytorch/pytorch/issues/54983
            args = tuple(float(arg) if isinstance(arg, Real) and not isinstance(arg, Integral)
                         else arg for arg in args)
            if self.func_takes_self:
                args = (qty,) + args
            if self.incremental:
                if func(*args, **kwargs) is NotImplemented:
                    return NotImplemented
                qty.unit = unit
                return qty
            else:
                res = func(*args, **kwargs)
                _meta_update = getattr_static(qty, '_meta_update')

                if not isinstance(unit, Unit) and isinstance(unit, Iterable):
                    if isinstance(res, qty._T):
                        unit = first(unit)
                    else:
                        return type(res)(_meta_update(qty, r, unit=u)
                                         for r, u in zip(res, unit))
                return _meta_update(qty, res, unit=unit)


@dataclass(eq=False)
class QuantityDelegator(QuantityDelegatorBase):
    in_unit: Union[Unit, None, Literal[False]] = None
    out_unit: _unitT = None

    strict: bool = True

    _to: Callable[[quantity.GenericQuantity, Iterable, Unit], Iterable]\
        = staticmethod(lambda self, qty, args, unit: qty._to(args, unit, self.strict))

    def to(self, qty: quantity.GenericQuantity, args: Iterable, unit: Unit) -> Iterable:
        return self._to(self, qty, args, unit)

    def _get(self, func):
        def f(qty: quantity.GenericQuantity, *args, **kwargs):
            if self.in_unit is not False:
                if self.func_takes_self:
                    args = (qty,) + args
                units = [Unit() if unit is None else unit
                         for a in self.filter(qty, args)
                         for unit in [a.unit if isinstance(a, quantity.GenericQuantity) else False]
                         if self.strict or unit is not False]
                dims = [u.dimension if isinstance(u, Unit) else u for u in units]
                if not all_equal(dims):
                    raise UnitError(f'All inputs have to have the same dimension, but found {dims}.')
                args = self.to(qty, args, units[0])
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
            if self.func_takes_self and len(args) == 1:
                if isinstance(arg := args[0], Unit):
                    otherunit = arg
                    args = (1,)
                else:
                    otherunit = arg.unit if isinstance(arg, quantity.GenericQuantity) else Unit()
                unit = self.op(*((otherunit, qty.unit) if self.flip else (qty.unit, otherunit)))
            else:
                unit = reduce(self.op, (a.unit for a in self.filter(qty, args)
                                        if isinstance(a, quantity.GenericQuantity)))
            return self.finalize_get(qty, func, unit, *args, **kwargs)
        return f


class PowerDelegator(QuantityDelegatorBase):
    def _get(self, func):
        def f(qty: quantity.GenericQuantity, other, *args, **kwargs):
            if not isinstance(other, Number):
                raise UnitError('Can only raise quantity to a scalar power.')
            return self.finalize_get(qty, func, qty.unit**other, other, *args, **kwargs)
        return f
