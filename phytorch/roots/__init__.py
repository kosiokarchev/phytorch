from functools import partial, reduce
from itertools import combinations
from operator import mul
from typing import Optional, Sequence

import torch
from torch import Tensor

from ..utils.complex import complex_typemap
from ..utils.function_context import TorchFunctionContext


product = partial(reduce, mul)


def vieta(rts: Sequence[Tensor]) -> Sequence[Tensor]:
    return [torch.ones_like(rts[0]), *(
        (-1)**k * sum(map(product, combinations(rts, k)))
        for k in range(1, len(rts) + 1)
    )]


def companion_matrix(p: Tensor) -> Tensor:
    return torch.cat((
        -p.unsqueeze(-2),
        torch.eye(p.shape[-1]-1, p.shape[-1]).unsqueeze(-3).unflatten(0, *p.shape[:-1])
    ), -2)


class Roots(torch.autograd.Function):
    """
    Calculate roots of a polynomial, given its (complex) coefficients.

    Note:
        It is assumed that the leading coefficient is unity and should not be
        given!
    """

    from ..extensions import roots as _roots
    _rootfuncs = {2: _roots.roots2, 3: _roots.roots3, 4: _roots.roots4}

    @staticmethod
    def forward(ctx: TorchFunctionContext, *coeffs: Tensor) -> tuple[Tensor, ...]:
        n = len(coeffs)
        if n == 0:
            raise ValueError('At least 2 coefficients are needed.')
        if n == 1:
            raise NotImplementedError('You already have the answer...')
        if 2 <= n <= 4:
            rts = (Roots._rootfuncs[len(coeffs)])(*(c.type(complex_typemap.get(c.dtype, c.dtype)) for c in coeffs))
        else:
            assert not any(len(c.shape) > 1 for c in coeffs),\
                'PyTorch does not support batched general eigenvalue calculation,'\
                'which is needed for general root finding.'
            rts = companion_matrix(torch.stack(torch.broadcast_tensors(coeffs), -1)).eig(eigenvectors=False).eigenvalues
            if not torch.is_complex(rts):
                rts = torch.complex(*rts.movedim(-1, 0))
        ctx.save_for_backward(*rts)
        return rts

    @staticmethod
    def backward(ctx: TorchFunctionContext, *grad_outputs: Optional[Tensor]) -> tuple[Tensor]:
        rts = ctx.saved_tensors
        grads = -torch.stack([
            c for r in combinations(rts[::-1], len(rts) - 1)
            for c in vieta(r)
        ], -1).unflatten(-1, (len(rts), len(rts))).inverse().movedim((-2, -1), (0, 1))
        return (grads.conj() * torch.stack(grad_outputs)).sum(1).unbind(0)


roots = Roots.apply

__all__ = 'vieta', 'companion_matrix', 'roots'
