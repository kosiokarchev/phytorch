from abc import ABC, abstractmethod
from typing import Iterable, Union

import torch
from more_itertools import first
from torch import searchsorted, Size, Tensor

from ..utils.broadcast import broadcast_except


# TODO: Abstract further
class AbstractBatchedInterpolator(ABC):
    def __init__(self, grid_shapes: Iterable[Size], values_shape: Size, channel_ndim=-1, **kwargs):
        super().__init__(**kwargs)
        grid_shapes = tuple(grid_shapes)
        self.grid_shape = Size(g[-1] for g in grid_shapes)
        self.ndim = len(self.grid_shape)
        assert self.grid_shape == values_shape[-self.ndim:]

        if channel_ndim < 0:
            channel_ndim = len(values_shape) - self.ndim

        self.channel_shape = values_shape[(0 if channel_ndim<0 else -self.ndim-channel_ndim):-self.ndim]
        self.channel_ndim = len(self.channel_shape)
        self.batch_shape = torch.broadcast_shapes(*(g[:-1] for g in grid_shapes), values_shape[:-self.ndim - channel_ndim])
        self.batch_ndim = len(self.batch_shape)

    @staticmethod
    def interp_input(*args: Tensor):
        return torch.stack(torch.broadcast_tensors(*map(torch.as_tensor, args)), -1)

    def _unsqueeeze_channels(self, *tensors, pos=-4) -> Union[Tensor, Iterable[Tensor]]:
        res = ((t.unsqueeze(pos).unflatten(pos, (1,) * self.channel_ndim) for t in tensors)
               if self.channel_ndim else tensors)
        return res if len(tensors) > 1 else first(res)

    # TODO: handle out of bounds
    @staticmethod
    def project_one(x: Tensor, grid: Tensor) -> Tensor:
        return searchsorted(*broadcast_except(grid, x, dim=-1), right=True).clamp_(1, grid.shape[-1]-1).sub_(1)

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor: ...
