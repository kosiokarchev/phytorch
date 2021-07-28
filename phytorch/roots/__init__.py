from itertools import combinations
from typing import Optional, Sequence

import torch
from torch import Tensor

from ..utils._typing import _TN
from ..utils.complex import as_complex_tensors
from ..utils.function_context import TorchFunctionContext
from ..utils.symmetry import elementary_symmetric


def vieta(rts: Sequence[Tensor]) -> Sequence[Tensor]:
    return [torch.ones_like(rts[0]), *(
        (-1)**k * elementary_symmetric(k, rts)
        for k in range(1, len(rts) + 1)
    )]


def companion_matrix(*coeffs: Tensor) -> Tensor:
    return torch.cat((
        - (p := torch.stack(torch.broadcast_tensors(*coeffs), -1).unsqueeze(-2)),
        torch.eye(len(coeffs) - 1, len(coeffs)).expand(*p.shape[:-2], len(coeffs) - 1, len(coeffs))
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
    def _roots_via_companion(*coeffs: Tensor):
        return tuple(torch.linalg.eigvals(companion_matrix(*coeffs)).unbind(-1))

    @staticmethod
    def forward(ctx: TorchFunctionContext, force_numeric=False, *coeffs: _TN) -> tuple[Tensor, ...]:
        n = len(coeffs)
        if n == 0:
            raise ValueError('At least 2 coefficients are needed.')
        if n == 1:
            raise NotImplementedError('You already have the answer...')
        ctx.save_for_backward(*(
            rts := (
                Roots._roots_via_companion if force_numeric else
                Roots._rootfuncs.get(n, Roots._roots_via_companion)
            )(*coeffs)
        ))
        return rts

    @staticmethod
    def backward(ctx: TorchFunctionContext, *grad_outputs: Optional[Tensor]) -> tuple[Tensor]:
        rts = ctx.saved_tensors
        grads = -torch.stack([
            c for r in combinations(rts[::-1], len(rts) - 1)
            for c in vieta(r)
        ], -1).unflatten(-1, (len(rts), len(rts))).inverse().movedim((-2, -1), (0, 1))
        return (grads.conj() * torch.stack(grad_outputs)).sum(1).unbind(0)


def roots(*coeffs: _TN, force_numeric=False):
    return Roots.apply(force_numeric, *as_complex_tensors(*coeffs))


def sroots(*coeffs: _TN, dim: int = 0, force_numeric=False):
    return torch.stack(roots(*coeffs, force_numeric=force_numeric), dim=dim)


__all__ = 'vieta', 'companion_matrix', 'roots', 'sroots'
