import typing as tp
from numbers import Real

import numpy as np
import torch
from frozendict import frozendict
from torch import Tensor

from .functypes import DictByName, TORCH_FUNCTYPES_H
from ..delegator import Delegator, QuantityDelegator, QuantityDelegatorBase
from ..quantity import GenericQuantity, UnitFuncType
from ...units.Unit import Unit


class TorchQuantity(GenericQuantity[Tensor], Tensor):
    # Needs to exist in this class because of the the cls argument...
    def __new__(cls, *args, unit: Unit, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def as_subclass(self, cls):
        return self._fill_quantity(super().as_subclass(cls))

    @classmethod
    def _to(cls, args, unit: Unit, strict=False):
        return (
            args.to(unit=unit) if isinstance(args, GenericQuantity)
            # TODO: Nasty hack: https://github.com/pytorch/pytorch/issues/54983
            else args / float(unit.scale) if (strict and isinstance(args, (Real, Tensor)))
            else args if (isinstance(args, (str, torch.Size, Tensor, np.ndarray, tp.Iterator))
                          or not isinstance(args, tp.Iterable))
            else type(args)(cls._to(a, unit, strict=strict) for a in args))

    def to(self, unit: Unit = None, *args, **kwargs):
        if unit is not None and not isinstance(unit, Unit):
            unit = None
            args = (unit,) + args
        return super().to(unit, *args, **kwargs)

    @property
    def value(self) -> Tensor:
        with self.delegator_context:
            return super().view(self.shape)

    FUNCTYPES: tp.Mapping[tp.Any, UnitFuncType] = DictByName({
        v: key for key, val in TORCH_FUNCTYPES_H.items() for v in val})

    @property
    def delegator_context(self):
        return torch._C.DisableTorchFunction()

    hypot, dist, maximum, minimum = (QuantityDelegator() for _ in range(4))
    ger = GenericQuantity.outer
    storage = QuantityDelegator(out_unit=None)
    view, movedim, narrow = (QuantityDelegator(strict=False) for _ in range(3))

    new, new_tensor, new_full, new_ones, new_zeros, new_empty = (Delegator() for _ in range(6))

    def __iter__(self):
        with self.delegator_context:
            return map(self._fill_quantity, super().__iter__())

    def __torch_function__(self, func, types, args=(), kwargs=frozendict()):
        if ((_func := getattr(type(self), func.__name__, None)) is not None
                and _func is not func):
            with self.delegator_context:
                return super().__torch_function__(_func, types, args, kwargs)

        functype = self.FUNCTYPES.get(func, UnitFuncType.SAME)
        # if func is not Tensor.__repr__:
        #     print(func)
        #     print(f'self: {self is args[0] or [self is a for a in args[0]]}')
        #     print(f'__torch_function__[{functype}]:', func)
        #     print('\targs:  ', args)
        #     print('\tkwargs:', kwargs)

        if isinstance(functype, QuantityDelegatorBase):
            return functype._get(func)(self, *args, **kwargs)
        elif functype is UnitFuncType.NOUNIT:
            with self.delegator_context:
                return super().__torch_function__(func, types, args, kwargs)
        else:
            res = super().__torch_function__(func, types, args, kwargs)
            return self._to(res, self.unit)
