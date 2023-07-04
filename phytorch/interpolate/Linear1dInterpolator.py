from __future__ import annotations

from functools import cached_property
from itertools import permutations
from typing import Callable

import torch
from torch import Tensor

from phytorchx import aligned_expand, broadcast_gather, broadcast_left

from .abc import AbstractBatchedInterpolator


def merge_grids(left: Tensor, right: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    left, right = broadcast_left(left, right, ndim=-1)
    size = left.shape[-1] + right.shape[-1]
    mright = left.new_zeros(*left.shape[:-1], size, dtype=torch.bool)
    mright[torch.searchsorted(left, right) + torch.arange(right.shape[-1])] = True
    mleft = ~mright

    grid = left.new_empty(mright.shape)
    grid[mleft], grid[mright] = left, right
    return grid, (mleft, mright)


class Linear1dInterpolator(AbstractBatchedInterpolator):
    def __init__(self, x: Tensor, y: Tensor, channel_ndim=-1, **kwargs):
        super().__init__((x.shape,), y.shape, channel_ndim, **kwargs)
        self.x, self.y = x, y

    @cached_property
    def dx(self) -> Tensor:
        return torch.diff(self.x, dim=-1)

    @cached_property
    def dydx(self) -> Tensor:
        return torch.diff(self.y, dim=-1) / self._unsqueeeze_channels(self.dx, pos=-2)

    def derivative(self, x, return_idx=False):
        idx = self.project_one(x, self.x)
        deriv = broadcast_gather(self.dydx, -1, self._unsqueeeze_channels(idx, pos=-2))
        return (idx, deriv) if return_idx else deriv

    def __call__(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        idx, deriv = self.derivative(x, return_idx=True)
        dy = self._unsqueeeze_channels(x - broadcast_gather(self.x, -1, idx), pos=-2) * deriv
        return (broadcast_gather(self.y, -1, self._unsqueeeze_channels(idx, pos=-2)) + dy).squeeze(-1)

    def channel_first(self, t: Tensor):
        return t.movedim(tuple(range(self.batch_ndim, self.batch_ndim + self.channel_ndim)), tuple(range(self.channel_ndim)))

    def merge(self, other: Linear1dInterpolator, op: Callable[[Tensor, Tensor], Tensor]):
        assert self.ndim == other.ndim == 1
        x, (mleft, mright) = merge_grids(self.x, other.x)

        batch_shape = torch.broadcast_shapes(self.batch_shape, other.batch_shape)
        channel_shape = torch.broadcast_shapes(self.channel_shape, other.channel_shape)

        y = x.new_empty(2, *channel_shape, *batch_shape, x.shape[-1])
        for i, ((s, ms), (o, mo)) in enumerate(permutations(((self, mleft), (other, mright)))):
            y[i, ..., ms], y[i, ..., mo] = (
                aligned_expand(s.channel_first(_y), ndims=(s.channel_ndim, s.batch_ndim, s.ndim),
                               shapes=(channel_shape, batch_shape, _l.grid_shape))
                for _y, _l in zip((s.values, s(o.x)), (s, o))
            )

        return type(self)(x, op(*y).movedim(
            tuple(range(len(channel_shape))),
            tuple(range(len(batch_shape), len(batch_shape)+len(channel_shape)))
        ), channel_ndim=len(channel_shape))
