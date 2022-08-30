from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property, partial, reduce
from itertools import chain, repeat
from numbers import Number
from typing import (
    Any, Callable, Iterable, Literal, Mapping, MutableMapping, Optional,
    Sequence, TypeVar, Union
)

import torch
from more_itertools import padded
from torch import Tensor
from typing_extensions import TypeAlias

from . import quantity, tensor_quantity
from ..units.angular import radian
from ..units.exceptions import UnitError
from ..units.unit import Unit
from ..utils import AutoUnpackable, pytree


_Unitful = (quantity.GenericQuantity, Unit)


@dataclass
class Eval:
    expr: str
    name: str = None

    @cached_property
    def compiled_expr(self):
        return compile(self.expr, '', 'eval')

    def __call__(self, ctx: Union[Mapping, MutableMapping]) -> __OutputUnitSpecT:
        ret = eval(self.compiled_expr, {}, ctx)
        if self.name is not None:
            ctx[self.name] = ret
        return ret


_T = TypeVar('_T')
_UnitSpecT: TypeAlias = Union[Unit, str, Eval, None]
_OUnitSpecT: TypeAlias = Union[_UnitSpecT, Literal[False]]
_InputUnitSpecT: TypeAlias = Union[Iterable['_InputUnitSpecT'], Mapping[str, '_InputUnitSpecT'], _UnitSpecT]
__OutputUnitSpecT: TypeAlias = Union[Iterable['__OutputUnitSpecT'], Mapping[str, '__OutputUnitSpecT'], _OUnitSpecT]
_OutputUnitSpecT: TypeAlias = Union[__OutputUnitSpecT, str, Callable[[...], __OutputUnitSpecT]]
_ctxVT: TypeAlias = Any


class AbstractDimSpec(ABC):
    @abstractmethod
    def __call__(self, *args) -> tuple[Sequence, BasePostprocessor]: ...


@dataclass
class DimSpecDispatcher(AbstractDimSpec):
    i: int
    specs: Iterable[tuple[type, AbstractDimSpec]]

    def __call__(self, *args):
        return next(spec(*args) for typ, spec in self.specs if isinstance(args[self.i], typ))


@dataclass
class BasicDimSpec(AutoUnpackable, AbstractDimSpec):
    params: Optional[_InputUnitSpecT] = None
    ret: _OutputUnitSpecT = None
    flipped: bool = False

    def flip(self):
        ret = copy(self)
        ret.flipped = not self.flipped
        return ret

    def __call__(self, *args) -> tuple[Sequence, BasePostprocessor]:
        return args, BasePostprocessor(
            pytree.tree_map(lambda a: a.unit if isinstance(a, _Unitful) else False, args[0])
            if self.ret is None else self.ret)


class DetDimSpec(AbstractDimSpec):
    def __call__(self, arg):
        return (arg,), BasePostprocessor(arg.unit ** arg.shape[-1])


@dataclass
class BasePostprocessor:
    ret: Unit

    @staticmethod
    def filter_only_float_tensors(t):
        return isinstance(t, tensor_quantity.TensorQuantity) and (torch.is_floating_point(t) or torch.is_complex(t))

    filter = filter_only_float_tensors
    dimless_to_value = False

    @staticmethod
    def process_filtered(output):
        return output.value if isinstance(output, quantity.GenericQuantity) else output

    def process(self, output, spec):
        _outputs, _tree = pytree.tree_flatten(output)
        return pytree.tree_unflatten([
            self.process_filtered(o) if (
                s is False
                or (self.dimless_to_value and isinstance(s, Unit)
                    and not s.dimension and s.value == 1)
                or not self.filter(o)
            ) else o._meta_update(o, unit=s)
            if isinstance(o, quantity.GenericQuantity) else o
            for _spec in [pytree._broadcast_to_and_flatten(spec, _tree)]
            for o, s in zip(_outputs, (_spec if _spec is not None else spec))
        ], _tree)

    def __call__(self, output):
        return self.process(output, self.ret)


class MultiproductDimSpec(BasicDimSpec):
    def __call__(self, *args) -> tuple[Sequence, BasePostprocessor]:
        _args, treespec = pytree.tree_flatten(args)

        retargs, units = zip(*(
            (arg.value, arg.unit) if isinstance(arg, _Unitful) else (arg, None)
            for arg in _args))
        return pytree.tree_unflatten(retargs, treespec), BasePostprocessor(reduce(operator.mul, filter(
            partial(operator.is_not, None), units
        )))


@dataclass
class PowerDimSpec(BasicDimSpec):
    def __call__(self, base, exponent, *args, **kwargs):
        if self.flipped:
            base, exponent = exponent, base

        if isinstance(exponent, _Unitful):
            if exponent.unit.dimension:
                raise UnitError('can only raise to dimensionless power.')
            exponent = exponent.to(Unit()).value

        return (base, exponent) if not self.flipped else (exponent, base), BasePostprocessor(
            base.unit**(exponent if isinstance(exponent, Number) else float(exponent))
            if isinstance(base, _Unitful) else False)


@dataclass
class DimSpec(BasicDimSpec):
    params: _InputUnitSpecT = ('unit',)

    def __call__(self, *args) -> tuple[Sequence, Postprocessor]:
        ctx: dict[str, _ctxVT] = {}

        if self.params is not None:
            retargs = []

            for _arg, _unit in zip(args, padded(self.params, None)):
                _args, _tree = pytree.tree_flatten(_arg)
                _units = pytree._broadcast_to_and_flatten(_unit, _tree)

                _retarg = []
                for arg, unit in zip(_args, _units):
                    if isinstance(unit, str):
                        if unit in ctx:
                            unit = ctx[unit]
                        else:
                            ctx[unit] = arg.unit if isinstance(arg, _Unitful) else Unit()
                    elif isinstance(unit, Eval):
                        unit = unit(ctx)

                    if isinstance(unit, Unit):
                        if isinstance(arg, _Unitful):
                            if not arg.unit.dimension == unit.dimension:
                                raise UnitError(f'expected {unit.dimension} but got {arg.unit.dimension}.')
                            arg = arg.to(unit)
                        # TODO: better ignored args
                        elif isinstance(arg, (Number, Tensor)) and not isinstance(arg, bool):
                            if unit.dimension:
                                raise UnitError(f'expected {unit.dimension} but got dimensionless.')
                            if unit.value != 1:
                                # TODO: nasty hack
                                arg = arg / float(unit.value)

                    _retarg.append(arg)
                retargs.append(pytree.tree_unflatten(_retarg, _tree))
        else:
            retargs = args

        return retargs, Postprocessor(self.ret, ctx)


@dataclass
class Postprocessor(BasePostprocessor):
    ret: _OutputUnitSpecT
    ctx: Mapping[str, _ctxVT]

    def process_spec(self, spec):
        return (
            self.ctx[spec] if isinstance(spec, str) else
            spec(**self.ctx) if callable(spec) else spec
        )

    def __call__(self, output: _T) -> _T:
        return self.process(output, pytree.tree_map(self.process_spec, self.ret))


dimless = Unit()

default_dimspec = BasicDimSpec()

same = DimSpec(repeat('unit'), 'unit')
rad_to_dimless = DimSpec((radian,), False)
trans_to_radian = DimSpec((dimless,), radian)
trans = DimSpec((dimless,), False)
nounit = BasicDimSpec(None, False)
same_to_nounit = DimSpec(repeat('unit'), False)


class _dimspecs:
    (eq, equal, ne, not_equal,
     gt, greater, ge, greater_equal,
     lt, less, le, less_equal) = same_to_nounit

    # TODO: atol of closeness
    allclose = isclose = DimSpec(('unit', 'unit', None, None), False)

    add = sub = subtract = DimSpec(('unit', 'unit2', Eval('unit / unit2')), 'unit')
    rsub = DimSpec(('unit2', 'unit', Eval('unit / unit2')), 'unit')
    __rsub__ = same

    minimum = maximum = fmin = fmax = hypot = hardshrink = nextafter = same
    dist = cdist = DimSpec(('unit', 'unit'), 'unit')

    (mul, multiply, matmul, __rmatmul__,
     cross, dot, mm, bmm, mv, inner, outer, ger, kron, vdot, tensordot
     ) = DimSpec(('unit1', 'unit2'), lambda unit1, unit2: unit1 * unit2)
    chain_matmul = einsum = MultiproductDimSpec()
    trapz = trapezoid = cumulative_trapezoid = MultiproductDimSpec()

    # gradient(..., spacing: Iterable, dim: Iterable)
    #   -> *(gradient(..., spacing=..., dim=int)[0] for ...),
    gradient = div = divide = true_divide = DimSpec(('unit1', 'unit2'), lambda unit1, unit2: unit1 / unit2)

    __rmod__ = fmod = remainder = same
    __floordiv__ = __rfloordiv__ = floor_divide = same_to_nounit

    pow = matrix_power = float_power = PowerDimSpec()
    __rpow__ = PowerDimSpec(flipped=True)

    (block_diag, cat, concat,
     stack, hstack, dstack, vstack, row_stack, column_stack,
     cartesian_prod,
     # atleast(*args) -> *map(atleast, args), in __torch__function__
     # atleast_1d, atleast_2d, atleast_3d
     ) = DimSpec(ret='unit')

    clip = clamp = clamp_min = clamp_max = nan_to_num = complex = same

    bucketize = isin = searchsorted = same_to_nounit

    sign = sgn = corrcoef = nounit
    angle = BasicDimSpec(None, radian)
    atan2 = DimSpec(('unit', 'unit'), radian)

    sin = cos = tan = sinc = rad_to_dimless
    asin = arcsin = acos = arccos = atan = arctan = trans_to_radian

    (exp, exp2, expm1, matrix_exp,
     log, log2, log10, log1p, logsumexp, logcumsumexp,
     lgamma, digamma, mvlgamma,
     erf, erfc, erfinv, logit, sigmoid, i0,
     sinh, asinh, arcsinh, cosh, acosh, arccosh, tanh, atanh, arctanh,
     bernoulli, logdet, slogdet, softmax, log_softmax,
     ceil, floor, round, fix, trunc, frac
     ) = trans
    polygamma = DimSpecDispatcher(i=0, specs=(
        (int, DimSpec((None, dimless), trans.ret)),
        (object, trans)
    ))

    logaddexp = logaddexp2 = igamma = igammac = DimSpec((dimless, dimless), False)

    det = DetDimSpec()

    histc = DimSpec(('unit', None, 'unit', 'unit'), False)
    histogram = DimSpecDispatcher(i=1, specs=(
        (int, _hist_spec := DimSpec(('unit',), (False, 'unit'))),
        (object, DimSpec(('unit', 'unit'), _hist_spec.ret))
    ))

    diff = DimSpec(('unit', None, None, 'unit', 'unit'), 'unit')

    polar = DimSpec(('unit', radian), 'unit')

    (where, masked_scatter, masked_fill,
     put, index_put
     ) = DimSpec(('unit', None, 'unit'), 'unit')

    index_copy = index_fill = scatter = scatter_add = DimSpec(('unit', None, None, 'unit'), 'unit')
    index_add = DimSpec(('unit', None, None, 'unit', dimless), 'unit')

    ldexp = xlogy = prelu = DimSpec(('unit', dimless), 'unit')
    lerp = DimSpec(('unit', 'unit', dimless), 'unit')
    # heaviside: can't assure func(..., val/2 * (2*unit)) == func(..., val)
    heaviside = DimSpec((None, dimless), False)

    sqrt = DimSpec(ret=lambda unit: unit ** Fraction(1, 2))
    rsqrt = DimSpec(ret=lambda unit: unit ** Fraction(-1, 2))
    square = var = cov = DimSpec(ret=lambda unit: unit ** Fraction(2))
    var_mean = DimSpec(ret=lambda unit: (unit ** Fraction(2), unit))

    inverse = pinverse = reciprocal = DimSpec(ret=lambda unit: ~unit)

    lu = eig = symeig = lobpcg = DimSpec(ret=('unit', False))
    triangular_solve = lstsq = DimSpec(('unit1', 'unit2'), lambda unit1, unit2: (
        unit1 / unit2, unit2
    ))
    linalg_solve = DimSpec(('unit1', 'unit2'), lambda unit1, unit2: unit2 / unit1)
    lu_solve = DimSpec(('unit1', 'unit2', dimless), lambda unit1, unit2: unit1 / unit2)
    lu_unpack = DimSpec(('unit', dimless), (False, False, 'unit'))

    cholesky = sqrt
    cholesky_inverse = DimSpec(ret=lambda unit: unit ** Fraction(-2))
    cholesky_solve = DimSpec(('unit1', 'unit2'), lambda unit1, unit2: unit1 * unit2 ** Fraction(-2))

    qr = DimSpec(ret=(False, 'unit'))
    svd = pca_lowrank = DimSpec(ret=(False, 'unit', False))
    svd_lowrank = DimSpec(('unit', None, None, 'unit'), (False, 'unit', False))


    addmm = addbmm = baddbmm = addmv = addr = DimSpec(('a', 'b1', 'b2', 'beta', Eval('beta * a / (b1 * b2)')), lambda a, beta, **kwargs: a * beta)
    addcmul = DimSpec(('unit', 'unit1', 'unit2', Eval('unit / (unit1 * unit2)')), 'unit')
    addcdiv = DimSpec(('unit', 'unit1', 'unit2', Eval('unit * unit2 / unit1')), 'unit')


dimspecs: Mapping[Callable, AbstractDimSpec] = _dimspecs.__dict__


def _torch_and_tensor(*torch_funcs):
    """Given a set of ``torch.*`` functions, return it, augmented with the ``Tensor.*`` mirrors"""
    return set(chain(torch_funcs, (ret for func in torch_funcs for ret in [getattr(Tensor, func.__name__, None)] if ret is not None)))


forbidden = _torch_and_tensor(
    torch.bincount,
    torch.bitwise_not, torch.bitwise_and, torch.bitwise_or, torch.bitwise_xor, torch.bitwise_left_shift, torch.bitwise_right_shift,
    torch.prod, torch.cumprod,
    torch.deg2rad, torch.rad2deg,
    torch.gcd, torch.lcm,
    torch.norm, torch.renorm, torch.frexp,

    Tensor.align_as, Tensor.align_to, Tensor.rename, Tensor.refine_names,

    torch.fake_quantize_per_channel_affine, torch.fake_quantize_per_tensor_affine,
    torch.quantized_batch_norm, torch.quantized_max_pool1d, torch.quantized_max_pool2d,
    torch.dequantize, torch.int_repr, torch.q_per_channel_axis, torch.q_per_channel_scales, torch.q_per_channel_zero_points,

    Tensor.coalesce, Tensor.is_coalesced, Tensor.to_dense, Tensor.to_sparse,
    Tensor.col_indices, Tensor.crow_indices, Tensor.indices, Tensor.values,
    Tensor.dense_dim, Tensor.sparse_dim, Tensor.sparse_mask,
)
