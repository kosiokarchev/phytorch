"""
Auto-differentiable polynomial-root finding.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence, Collection

import torch
from more_itertools import prepend
from torch import Tensor
from torch.overrides import wrap_torch_function

from ..utils._typing import _TN
from ..utils.complex import as_complex_tensors
from ..utils.function_context import TorchFunctionContext
from ..utils.symmetry import elementary_symmetric


def vieta(rts: Collection[Tensor]) -> Sequence[Tensor]:
    r"""Calculate coefficients of a polynomial from its roots (inverse of |roots|). [Vieta]_

    Parameters
    ----------
    rts
        roots

    Returns
    -------
    :
        :arg:`coeffs`, as for |roots|, i.e. in order of decreasing power and
        assuming the leading coefficient is unity (and is, hence, skipped)
    """
    return tuple((-1)**k * elementary_symmetric(k, rts) for k in range(1, len(rts)+1))


def companion_matrix(*coeffs: Tensor) -> Tensor:
    r"""Create a `companion matrix <https://en.wikipedia.org/wiki/Companion_matrix>`_ from polynomial coefficients as for |roots|."""
    return torch.cat((
        - (p := torch.stack(torch.broadcast_tensors(*coeffs), -1).unsqueeze(-2)),
        torch.eye(len(coeffs) - 1, len(coeffs), dtype=p.dtype, device=p.device).expand(*p.shape[:-2], len(coeffs) - 1, len(coeffs))
    ), -2)


class Roots(torch.autograd.Function):
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
            c for r in combinations(reversed(rts), len(rts) - 1)
            for c in prepend(torch.ones_like(r[0]), vieta(r))
        ], -1).unflatten(-1, (len(rts), len(rts))).inverse().movedim((-2, -1), (0, 1))
        return (None,) + tuple((grads.conj() * torch.stack(grad_outputs)).sum(1).unbind(0))


@wrap_torch_function(lambda *args, **kwargs: args)
def roots(*coeffs: _TN, force_numeric=False) -> Collection[Tensor]:
    r"""Compute the roots of a polynomial. Result always has a complex `dtype`.

    Parameters
    ----------
    *coeffs: `~torch.Tensor`\ s or `~numbers.Number`\ s
        coefficients of the polynomial, *arranged in order of decreasing powers*,
        and *omitting the leading coefficient, assumed unity*:

        .. math::
            P(x) = x^N + \mathtt{coeffs[} 0 \mathtt{]} x^{N-1}
                       + \mathtt{coeffs[} 1 \mathtt{]} x^{N-2} + ...
            \\ ... + \mathtt{coeffs[} N-1 \mathtt{]} x
                   + \mathtt{coeffs[} N \mathtt{]}.

    force_numeric: bool
        whether to always use the numeric algorithm or to allow the analytic
        solutions for polynomials of degree up to 4 (requires the
        `~phytorch.extensions.roots` extensions)

    Returns
    -------
    tuple[~torch.Tensor, ...]
        A tuple of :math:`N` complex `~torch.Tensor`\ s, where :math:`N` is the
        number of coefficients given (equal to the degree of the polynomial).
        No guarantees are made about the order in which the roots are returned.

    See Also
    --------
    numpy.roots
    """
    return Roots.apply(force_numeric, *as_complex_tensors(*coeffs))


@wrap_torch_function(lambda *args, **kwargs: args)
def sroots(*coeffs: _TN, dim: int = 0, force_numeric=False) -> Tensor:
    r"""Like |roots| but ``s``\ tacks the roots along a (new) dimension :arg:`dim`, thus returning a single `~torch.Tensor`.

    Notes
    -----
        The order of the roots along :arg:`dim` is undefined, as for |roots|.
    """
    return torch.stack(roots(*coeffs, force_numeric=force_numeric), dim=dim)


__all__ = 'vieta', 'companion_matrix', 'roots', 'sroots'
