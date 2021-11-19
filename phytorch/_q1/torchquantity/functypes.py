from itertools import chain
from typing import Any, Dict, Protocol, TYPE_CHECKING

import torch
from more_itertools import first
from torch import Tensor


from ..quantity import UnitFuncType  # Must come before delegator import!
from ..delegation.quantity_delegators import ProductDelegator, QuantityDelegator


if TYPE_CHECKING:
    from torch._C._VariableFunctions import *

globals().update({key: getattr(torch._C._VariableFunctions, key) for key in dir(torch._C._VariableFunctions)})


__all__ = 'TORCH_FUNCTYPES_H',


def _torch_and_tensor(*torch_funcs):
    """Given a set of ``torch.*`` functions, return it, augmented with the ``Tensor.*`` mirrors"""
    return set(chain(torch_funcs, (getattr(Tensor, func.__name__) for func in torch_funcs)))


class NamedProtocol(Protocol):
    __name__: str


class DictByName(Dict[NamedProtocol, Any]):
    notfound = object()

    def _get_alias(self, item: NamedProtocol):
        return first((key for key in self.keys()
                      if key.__name__ == item.__name__),
                     self.notfound)

    def __missing__(self, key):
        if (alias := self._get_alias(key)) is self.notfound:
            raise KeyError(key)
        return self.setdefault(key, self[alias])

    def __contains__(self, item):
        return super().__contains__(item) or self._get_alias(item) is not self.notfound

    def get(self, key, default=None):
        return self[key] if key in self else default


# https://pytorch.org/docs/stable/torch.html#math-operations
TORCH_FUNCTYPES_H = {
    # UnitFuncType.REJECT: _torch_and_tensor(prod, cumprod),
    UnitFuncType.NOUNIT: {
        Tensor.__repr__, Tensor.__dir__, Tensor.__reduce__, Tensor.__reduce_ex__
    },
    UnitFuncType.SAME: {
        Tensor.requires_grad_, Tensor.to, Tensor.expand
    } | _torch_and_tensor(
        # nan_to_num
        abs, absolute, ceil, clamp, clip, clone, conj, fix, trunc, floor, frac,
        real, imag, neg, negative, nextafter, round,
        index_select, masked_select, take,
        reshape, squeeze, unsqueeze, t, transpose, flatten,
        diag, diag_embed, diagflat, diagonal, flip, fliplr, flipud, rot90, roll,
        chunk, split, unbind, repeat_interleave,
        mean, median, mode, norm, nansum, quantile, nanquantile, std, sum,
        trace, tril, triu,
        torch.unique, torch.unique_consecutive, sort,
        max, amax, min, amin, topk, sum, cumsum,
    ) | {atleast_1d, atleast_2d, atleast_3d, broadcast_tensors, std_mean, tensordot, combinations, view_as_real, view_as_complex},
    # UnitFuncType.SUM: {cat, stack, dstack, hstack, vstack, block_diag, cdist},
    QuantityDelegator(func_takes_self=False, strict=False): {
        cat, stack, dstack, hstack, vstack, block_diag, cdist},
    ProductDelegator(func_takes_self=False): {
        cartesian_prod, trapz},
    QuantityDelegator(func_takes_self=False, out_unit=False): {
        bucketize, searchsorted, tril_indices, triu_indices}
    # UnitFuncType.SUM: _torch_and_tensor(  # Done!
    #     add, sub, subtract, hypot, dist,
    #     eq, equal, ne, not_equal, ge, greater_equal, gt, greater, le, less_equal, lt, less,
    #     maximum, minimum,
    # ) | {cat, stack, dstack, hstack, vstack, block_diag, cdist},
    # UnitFuncType.PRODUCT: _torch_and_tensor(  # Done!
    #     mul, multiply, cross, dot, matmul, mm, mv, outer, vdot,
    # ) | {cartesian_prod, trapz},
    # UnitFuncType.POWER: _torch_and_tensor(  # Done!
    #     pow, matrix_power
    # ),
    # UnitFuncType.TRANS: _torch_and_tensor(  # Done!
    #     # igamma, igammac,
    #     exp, exp2, expm1, log, log10, log2, log1p, logaddexp, logaddexp2,
    #     logsumexp, logcumsumexp, matrix_exp,
    #     sinh, cosh, tanh, asinh, arcsinh, acosh, arccosh, atanh, arctanh,
    #     digamma, erf, erfc, erfinv, lgamma, logit, i0,
    #     mvlgamma, polygamma, sigmoid,
    # ),
    # UnitFuncType.RAD: _torch_and_tensor(  # Done!
    #     sin, cos, tan,
    # ),
    # UnitFuncType.INVRAD: _torch_and_tensor(  # Done!
    #     asin, arcsin, acos, arccos, atan, arctan,
    # ),
    # UnitFuncType.SAMETORAD: _torch_and_tensor(  # Done!
    #     angle, atan2
    # ),
    # UnitFuncType.SAMETODIMLESS: _torch_and_tensor(  # Done!
    #     sign, argmax, argmin, count_nonzero, bincount, argsort,
    #     allclose, isclose, isfinite, isinf, isposinf, isneginf, isnan, isreal,
    #     histc,
    # ) | {bucketize, searchsorted, tril_indices, triu_indices}
}

# TODO: important!!
#   meshgrid

# TODO: torch 1.8?
#   copysign, float_power, ldexp, nan_to_num, igamma, igammac
# TODO:
#   lerp -- maybe ensure weight is dimensionless
#   cummax, cummin, gcd, lcm,
#   where, var_mean, kthvalue
