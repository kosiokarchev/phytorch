import contextlib
import dataclasses
import operator
import typing as tp
from abc import ABC
from dataclasses import dataclass
from functools import reduce, update_wrapper

from more_itertools import collapse

from . import quantity
from ..units.Unit import Unit


class Delegating(ABC):
    @property
    def delegator_context(self):
        return contextlib.nullcontext()


@dataclass(eq=False)
class Delegator:
    name: str = dataclasses.field(init=False)
    func_takes_self: bool = True

    def __set_name__(self, owner, name):
        self.name = name

    def _get(self, func):
        def f(slf: Delegating, *args, **kwargs):
            with slf.delegator_context:
                if self.func_takes_self:
                    args = (slf,) + args
                return func(*args, **kwargs)
        return f

    def __get__(self, instance, owner: tp.Type['quantity.GenericQuantity']):
        if owner is quantity.GenericQuantity:
            return self
        func = getattr(owner._T, self.name)
        ret = update_wrapper(self._get(func), func)
        # print('getting', self.name, 'of', owner._T, 'result', ret)
        setattr(owner, self.name, ret)
        return getattr(instance, self.name) if instance is not None else ret


@dataclass(eq=False)
class QuantityDelegatorBase(Delegator):
    incremental: bool = False

    _filter: tp.Callable[['quantity.GenericQuantity', tp.Iterable], tp.Iterable]\
        = staticmethod(lambda self, qty, args: collapse(args, base_type=qty._T))

    def filter(self, qty: 'quantity.GenericQuantity', args: tp.Iterable) -> tp.Iterable:
        return self._filter(self, qty, args)

    def finalize_get(self, qty, func, unit, *args, **kwargs):
        with qty.delegator_context:
            if self.func_takes_self:
                args = (qty,) + args
            if self.incremental:
                func(*args, **kwargs)
                qty.unit = unit
                return qty
            else:
                return qty._fill_quantity(func(*args, **kwargs), unit=unit)


@dataclass(eq=False)
class QuantityDelegator(QuantityDelegatorBase):
    in_unit: tp.Optional[Unit] = None
    out_unit: tp.Union[tp.Literal[False], None, Unit, tp.Callable[[Unit], Unit]] = None

    strict: bool = True

    _to: tp.Callable[['quantity.GenericQuantity', tp.Iterable, Unit], tp.Iterable]\
        = staticmethod(lambda self, qty, args, unit: qty._to(args, unit, self.strict))

    def to(self, qty: 'quantity.GenericQuantity', args: tp.Iterable, unit: Unit) -> tp.Iterable:
        return self._to(self, qty, args, unit)

    def _get(self, func):
        def f(qty: 'quantity.GenericQuantity', *args, **kwargs):
            in_unit = self.in_unit if self.in_unit is not None else qty.unit
            if self.func_takes_self:
                args = (qty,) + args
            assert all(
                a.unit.dimension == in_unit.dimension if isinstance(a, quantity.GenericQuantity)
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
    op: tp.Callable = operator.mul
    flip: bool = False

    def _get(self, func):
        def f(qty: 'quantity.GenericQuantity', *args, **kwargs):
            if len(args) == 1:
                otherunit = args[0].unit if isinstance(args[0], quantity.GenericQuantity) else Unit()
                unit = self.op(*((otherunit, qty.unit) if self.flip else (qty.unit, otherunit)))
            else:
                unit = reduce(self.op, (a.unit for a in self._filter(qty, args)
                                        if isinstance(a, quantity.GenericQuantity)))
            return self.finalize_get(qty, func, unit, *args, **kwargs)
        return f


class PowerDelegator(QuantityDelegatorBase):
    def _get(self, func):
        def f(qty: 'quantity.GenericQuantity', other, *args, **kwargs):
            try:
                other = float(other)
                return self.finalize_get(qty, func, qty.unit**other, other, *args, **kwargs)
            except ValueError:
                raise AssertionError('Can only raise quantity to a scalar power.')
        return f
