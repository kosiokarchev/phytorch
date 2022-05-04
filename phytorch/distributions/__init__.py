from typing import Sequence

import torch
from more_itertools import last
from torch import Size, Tensor
from torch.distributions import Distribution

from ..interpolate import Linear1dInterpolator, LinearNDGridInterpolator
from ..utils import _mid_many


class NDIDistribution(Distribution):
    arg_constraints = {}

    def __init__(self, grids: Sequence[Tensor], probs: Tensor, validate_args=None):
        self.grid_shape = Size(map(len, grids))
        self.ndim = len(self.grid_shape)

        assert self.grid_shape == probs.shape[-self.ndim:]

        self.grids = grids
        self.probs = probs

        self.marginal_cprobs = [
            torch.cat((torch.zeros_like(norm), cprob / norm), i)
            # (cprob - cprob.narrow(i, 0, 1)) / (cprob.narrow(i, -1, 1) - cprob.narrow(i, 0, 1))
            for i in range(-self.ndim, 0) for p in [_mid_many(self.probs, (i,))] for mprob in [
                p.sum(tuple(range(-self.ndim, i)))
                if i > -self.ndim else p
            ] for cprob in [mprob.cumsum(i)] for norm in [cprob.narrow(i, -1, 1)]
        ][::-1]
        self.cinterps = [
            LinearNDGridInterpolator(self.grids[-i:], self.marginal_cprobs[i], channel_ndim=1)
            for i in range(1, self.ndim)
        ]

        self.interp0 = Linear1dInterpolator(self.marginal_cprobs[0], self.grids[-1], channel_ndim=0)

        super().__init__(batch_shape=torch.broadcast_shapes(
            *(g.shape[:-1] for g in self.grids), self.probs.shape[:-self.ndim]
        ), event_shape=Size((self.ndim,)), validate_args=validate_args)

    def rsample(self, sample_shape=Size()):
        return self.icdf(torch.rand((self.ndim,) + Size(sample_shape) + self.batch_shape))

    def log_prob(self, value):
        pass

    def cdf(self, value):
        pass

    def icdf(self, value: Sequence[Tensor]):
        return last(
            x for x in [self.interp0(value[0]).unsqueeze(-1)]
            for cinterp, grid, _y in zip(self.cinterps, self.grids[-2::-1], value[1:])
            for x in [torch.cat((
                Linear1dInterpolator(cinterp(x), grid, channel_ndim=0)(_y).unsqueeze(-1),
                x
            ), -1)]
        )
