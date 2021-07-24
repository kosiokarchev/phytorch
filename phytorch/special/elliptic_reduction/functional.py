from __future__ import annotations

from itertools import chain
from typing import Optional, Sequence

import torch
from phytorch.utils.complex import as_complex_tensors
from torch import Tensor
from torch.autograd import Function

from .core import EllipticReduction
from ...utils.function_context import TorchFunctionContext


# noinspection PyMethodOverriding,PyAbstractClass
class EllipticIntegral(Function):
    @staticmethod
    def forward(ctx: TorchFunctionContext, m: Sequence[int], h, z1, z2, *a) -> Tensor:
        er = EllipticReduction(z2, z1, a, len(a) * (1,), h=h)
        ctx.er = er
        ctx.m = m
        return er.Im(m)

    @staticmethod
    def backward(ctx: TorchFunctionContext, grad_output) -> tuple[Optional[Tensor], ...]:
        er, m = ctx.er, ctx.m
        er: EllipticReduction
        m: Sequence[int]
        return tuple(
            None if grad is None else grad_output * (grad.conj() if torch.is_complex(grad) else grad)
            for grad in chain(2*(None,), (
                f(m) if nig else None for nig, f in zip(ctx.needs_input_grad[2:], (er.vy, er.vx))
            ), (
                (m[j] - (j < er.h) / 2) * EllipticIntegral.apply(
                    tuple(m[i] - (i == j) for i in range(er.n)), er.h, er.y, er.x, *er.a)
                if nig else None
                for j, nig in enumerate(ctx.needs_input_grad[4:])
            )))


def elliptic_integral(z1, z2, *a, m: Sequence[int] = 4 * (0,), h=4):
    return EllipticIntegral.apply(m, h, *map(torch.as_tensor, (z1, z2)), *as_complex_tensors(*a))
