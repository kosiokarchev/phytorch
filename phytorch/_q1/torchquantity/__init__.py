from __future__ import annotations

from fractions import Fraction
from itertools import chain, repeat
from numbers import Real
from operator import truediv
from typing import Any, Iterable, Iterator, Mapping, overload, TypeVar, Union

import numpy as np
import torch
from frozendict import frozendict
from more_itertools import collapse, padded
from torch import Tensor

from ..delegation.delegator import Delegator
from ..delegation.quantity_delegators import ProductDelegator, QuantityDelegator
from ..quantity import GenericQuantity
from ...meta import Meta
from ...units.angular import radian
from ...units.exceptions import UnitError
from ...units.unit import Unit


class TorchQuantityMeta(type(GenericQuantity), type(Tensor)):
    pass


class TorchQuantity(GenericQuantity[Tensor], Tensor, metaclass=TorchQuantityMeta):
    def __new__(cls, *args, unit: Unit, **kwargs):
        return super().__new__(cls, *args, unit=unit, **kwargs)

    def as_subclass(self, cls):
        return self._meta_update(self._T.as_subclass(self, cls))

    @classmethod
    @overload
    def _to(cls, args: _TQ_T, unit: Unit, strict=False) -> _TQ_T: ...

    @classmethod
    @overload
    def _to(cls, args: Real, unit: Unit, strict=False) -> Real: ...

    @classmethod
    @overload
    def _to(cls, args: Tensor, unit: Unit, strict=False) -> Tensor: ...

    @classmethod
    @overload
    def _to(cls, args: _TQ_to_iterable, unit: Union[Unit, Iterable[Unit]], strict=False) -> _TQ_to_iterable: ...

    @classmethod
    def _to(cls, args, unit, strict=False):
        return (
            (args.value if isinstance(args, GenericQuantity) else args) if unit is False else
            args.to(unit=unit) if isinstance(args, GenericQuantity)
            # TODO: Nasty hack: https://github.com/pytorch/pytorch/issues/54983
            else args / float(unit.value) if (strict and isinstance(args, (Real, Tensor)))
            else args if (isinstance(args, (str, torch.Size, Tensor, np.ndarray, Iterator))
                          or not isinstance(args, Iterable))
            else type(args)(
                cls._to(a, u, strict=strict) for a, u in zip(
                    args, repeat(unit) if isinstance(unit, Unit) else unit)))

    def to(self, unit: Unit = None, *args, **kwargs):
        if unit is not None and not isinstance(unit, Unit):
            args = (unit,) + args
            unit = None
        return super().to(unit, *args, **kwargs)

    @property
    def value(self) -> Tensor:
        with self.delegator_context:
            return super().view(self.shape)

    @classmethod
    @property
    def delegator_context(cls):
        return torch._C.DisableTorchFunction()

    ger = GenericQuantity.outer
    rsub = GenericQuantity.__rsub__
    storage, tolist = (QuantityDelegator() for _ in range(2))
    (nextafter, hypot, dist, minimum, maximum, fmin, fmax,
     hardshrink) = (
        QuantityDelegator() for _ in range(8))

    min, max, cummin, cummax, kthvalue = (
        QuantityDelegator(out_unit=(None, False), strict=False) for _ in range(5))
    nonzero = QuantityDelegator(out_unit=repeat(False))
    signbit, histc = (QuantityDelegator(out_unit=False) for _ in range(2))
    histogram = QuantityDelegator(out_unit=(False, None))
    frexp = QuantityDelegator(out_unit=(None, False))

    unique, unique_consecutive = (QuantityDelegator(out_unit=(None, False, False)) for _ in range(2))

    (bernoulli, softmax, log_softmax,
     logdet, slogdet) = (
        QuantityDelegator(in_unit=Unit(), out_unit=False, strict=False) for _ in range(5))
    multinomial = QuantityDelegator(out_unit=False, strict=False)

    cholesky = QuantityDelegator(out_unit=lambda unit: unit**Fraction(1, 2))
    cholesky_inverse = QuantityDelegator(out_unit=lambda unit: unit**Fraction(-2))
    lu, eig, symeig = (QuantityDelegator(out_unit=(None, False)) for _ in range(3))
    qr = QuantityDelegator(out_unit=(False, None))
    svd = QuantityDelegator(out_unit=(False, None, False))

    def rad2deg(self):
        raise NotImplementedError(f'Please use \'{GenericQuantity.to.__name__}\'.')

    def deg2rad(self):
        raise NotImplementedError(f'Please use \'{GenericQuantity.to.__name__}\'.')

    def renorm(self, p, dim, maxnorm):
        raise NotImplementedError('Please use `value.renorm() * unit` if you'
                                  ' really want to do something like this.')

    def __iter__(self):
        with self.delegator_context:
            return map(self._meta_update, super().__iter__())

    FUNCTYPES: Mapping[Any, Delegator] = {
        v: key.set_func_takes_self(False) for key, val in (
            (Delegator(), (torch.broadcast_tensors, torch.meshgrid)),
            (QuantityDelegator(), (torch.complex, torch.cartesian_prod)),
            (QuantityDelegator(out_unit=False), (
                torch.isin, torch.bucketize, torch.searchsorted,
                torch.matrix_rank
            )),
            (QuantityDelegator(out_unit=(lambda u: u**2, None)), (torch.var_mean,)),
            (ProductDelegator(), (
                torch.einsum, torch.trapz, torch.trapezoid, torch.cumulative_trapezoid,
                torch.chain_matmul
            )),
            (QuantityDelegator(strict=False), (
                torch.cat, torch.concat,
                torch.stack, torch.dstack, torch.hstack, torch.vstack, torch.row_stack, torch.column_stack,
                torch.block_diag, torch.cdist
            )),
            (QuantityDelegator(out_unit=(False, None, False)), (torch.pca_lowrank,))
        ) for v in val
    }

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=frozendict()):
        if func is torch.where:
            return Tensor.where(args[1], args[0], args[2])
        elif ((_func := getattr(cls, func.__name__, None)) is not None
                and _func is not func):
            return (_func(*args, **kwargs) if callable(_func) else
                    _func.__get__(args[0], type(args[0])) if hasattr(_func, '__get__') else
                    NotImplemented)

        # TODO: normalise args, kwargs
        self = next(a for a in collapse(chain(args, kwargs.values()), base_type=cls._T) if isinstance(a, cls))

        if func in cls.FUNCTYPES:
            res = cls.FUNCTYPES[func]._get(func)(self, *args, **kwargs)

            if func in (torch.broadcast_tensors, torch.meshgrid):
                res = tuple(a._meta_update(r.as_subclass(type(a)))
                            if isinstance(a, Meta) else r
                            for a, r in zip(args, res))
            elif func in (torch.trapz, torch.trapezoid, torch.cumulative_trapezoid):
                if isinstance(dx := kwargs.get('dx', None), (GenericQuantity, Unit)):
                    res.unit *= dx.unit
        else:
            unit = self.unit

            if func is torch.polar:
                assert len(args) <= 2
                arg1, arg2 = padded(args, fillvalue=None, n=2)
                arg1, arg2 = kwargs.get('abs', arg1), kwargs.get('angle', arg2)
                if isinstance(arg2, GenericQuantity):
                    arg2 = arg2.to(radian)
                args, kwargs = (arg1, arg2), {key: kwargs[key] for key in kwargs.keys()
                                              if key not in ('abs', 'angle')}

            elif func in (Tensor.ldexp, Tensor.xlogy) and isinstance(args[1], GenericQuantity) and args[1].unit.dimension:
                raise UnitError(f'second argument to \'{func.__name__}\' must be dimensionless.')

            elif func is Tensor.lerp:
                if not (isinstance(args[0], GenericQuantity) and isinstance(args[1], GenericQuantity)
                        and args[0].unit.dimension == args[1].unit.dimension):
                    raise UnitError('\'start\' and \'end\' given to \'{func.__name__}\''
                                    ' must have the same unit dimensions.')
                if isinstance(args[2], GenericQuantity) and args[2].unit.dimension:
                    raise UnitError(f'argument \'weight\' to \'{func.__name__}\' must be dimensionless.')

                args = cls._to(args, unit=2*(args[0].unit,) + (False,))

            # TODO: heaviside(quantity, tensor) -> tensor
            elif func is Tensor.heaviside:
                unit = args[1].unit if isinstance(args[1], GenericQuantity) else Unit()

            elif func in (Tensor.scatter, Tensor.scatter_add):
                if not (isinstance(args[0], GenericQuantity) and isinstance(args[3], GenericQuantity)
                        and args[0].unit.dimension == args[3].unit.dimension):
                    raise UnitError(f'\'input\' and \'src\' given to \'{func.__name__}\''
                                    ' must have the same unit dimensions.')
                unit = args[0].unit
                args = (args[0].to(unit), *args[1:3], args[3].to(unit))

            elif func is Tensor.where:
                if not (isinstance(args[0], GenericQuantity) and isinstance(args[2], GenericQuantity)
                        and args[0].unit.dimension == args[2].unit.dimension):
                    raise UnitError(f'\'x\' and \'y\' given to \'{func.__name__}\''
                                    ' must have the same unit dimensions.')
                unit = args[0].unit
                args = (args[0].to(unit), args[1], args[2].to(unit))

            elif func in (Tensor.put, Tensor.index_put, Tensor.masked_scatter):
                if not (isinstance(args[0], GenericQuantity) and isinstance(args[2], GenericQuantity)
                        and args[0].unit.dimension == args[2].unit.dimension):
                    raise UnitError(f'source and target given to \'{func.__name__}\''
                                    ' must have the same unit dimensions.')
                unit = args[0].unit
                args = (args[0].to(unit), args[1], args[2].to(unit))

            elif func in (Tensor.index_copy, Tensor.index_add):
                if not (isinstance(args[0], GenericQuantity) and isinstance(args[3], GenericQuantity)
                        and args[0].unit.dimension == args[3].unit.dimension):
                    raise UnitError(f'source and target given to \'{func.__name__}\''
                                    ' must have the same unit dimensions.')
                unit = args[0].unit
                args = (args[0].to(unit), *args[1:3], args[3].to(unit))

            elif func is Tensor.prelu:
                if isinstance(args[1], GenericQuantity) and args[1].unit.dimension:
                    raise UnitError(f'\'weight\' parameter given to \'{func.__name__}\''
                                    ' must be dimensionless.')

            elif func in (Tensor.solve, Tensor.triangular_solve, Tensor.lstsq):
                unit1, unit2 = (_.unit if isinstance(_, GenericQuantity) else Unit() for _ in args[:2])
                unit = unit1 / unit2, unit2 if unit2.dimension else False

            elif func is Tensor.cholesky_solve:
                unit1, unit2 = (_.unit if isinstance(_, GenericQuantity) else Unit() for _ in args[:2])
                unit = unit1 * unit2**Fraction(-2)

            elif func is Tensor.lu_solve:
                if isinstance(args[2], GenericQuantity) and args[2].unit.dimension:
                    raise UnitError('\'LU_pivots\' should be dimensionless.')
                unit = truediv(*(_.unit if isinstance(_, GenericQuantity) else Unit() for _ in args[:2]))

            elif func is torch.lu_unpack:
                if isinstance(args[1], GenericQuantity) and args[1].unit.dimension:
                    raise UnitError('\'LU_pivots\' should be dimensionless.')
                unit = (False,) + 2*(args[0].unit**Fraction(1, 2) if isinstance(args[0], GenericQuantity) else False,)

            elif func is torch.svd_lowrank:
                M = args[3] if len(args) == 4 else kwargs['M']
                if isinstance(M, GenericQuantity) and M.unit.dimension != args[0].unit.dimension:
                    raise UnitError(f'\'A\' and \'M\' parameters given to \'{func.__name__}\''
                                    ' must have the same unit dimensions.')
                unit = (False, args[0].unit, False)
                kwargs = {**kwargs, 'M': M}
                args = args[:3]

            elif func is Tensor.det:
                unit = unit ** Fraction(self.shape[-1])

            res = cls._to(super().__torch_function__(func, types, args, kwargs), unit)
        return res


_TQ_T = TypeVar('_TQ_T', bound=TorchQuantity)
_TQ_to_arg = TypeVar('_TQ_to_arg', TorchQuantity, Real, Tensor, Any)
_TQ_to_iterable = TypeVar('_TQ_to_iterable', bound=Iterable[_TQ_to_arg])
