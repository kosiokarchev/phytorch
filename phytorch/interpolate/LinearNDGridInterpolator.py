from itertools import product
from typing import Sequence

import torch
from torch import Tensor

from phytorchx import broadcast_gather

from .abc import AbstractBatchedInterpolator


class LinearNDGridInterpolator(AbstractBatchedInterpolator):
    def __init__(self, grids: Sequence[Tensor], values: Tensor, channel_ndim=-1, **kwargs):
        super().__init__((g.shape for g in grids), values.shape, channel_ndim, **kwargs)

        self.grids, self.values = grids, values
        self.dgrids = [torch.diff(g, dim=-1) for g in grids]

        self.di = self.values.new_tensor(tuple(product(*self.ndim*((0, 1),))), dtype=int)
        self.strides = self.values.new_tensor(
            [p for _p in [1] for s in self.values.shape[-self.ndim:][::-1]
             for p in [_p] for _p in [_p*s]][::-1],
            dtype=int)

    def __call__(self, x: Tensor):
        assert x.shape[-1] == self.ndim

        idxs, ws = (
            self._unsqueeeze_channels(torch.stack(_, -1).unsqueeze(-2))
            for _ in zip(*(
                (idx, w)
                for g, dg, _x in zip(self.grids, self.dgrids, x.split(1, -1))
                # TODO: handle out of bounds
                for idx in [self.project_one(_x, g)]
                for dgrid in [broadcast_gather(dg, -1, idx)]
                for w in [1 - (_x - broadcast_gather(g, -1, idx)) / dgrid]
            ))
        )

        idxs = ((idxs + self.di) * self.strides).sum(-1)
        ws = (self.di - ws).prod(-1).abs()

        return ((ws * broadcast_gather(self.values.flatten(-self.ndim), -1, idxs, index_ndim=2)).sum(-1) / ws.sum(-1)).squeeze(-1)
