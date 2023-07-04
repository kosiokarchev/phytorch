from itertools import product
from typing import Any, Type

import torch
from pytest import mark
from torch import Tensor

from phytorchx import insert_dims

from phytorch.interpolate.abc import AbstractBatchedInterpolator


class BaseTestShapes:
    ndim: int
    explicit_dim = True
    cls: Type[AbstractBatchedInterpolator]
    channel_shape = torch.Size((2, 3, 4))
    batch_shape = torch.Size((15, 17))

    x: Any
    values: Tensor

    def channeled(self, values):
        return insert_dims(values, -self.ndim - 1, self.channel_shape)

    def _test(self, interp: AbstractBatchedInterpolator, eval_shape: tuple):
        assert interp(torch.rand(eval_shape + ((self.ndim,) if self.explicit_dim else ()))).shape == torch.broadcast_shapes(interp.batch_shape, eval_shape) + interp.channel_shape

    @mark.parametrize('channel_ndim', tuple(range(-1, len(channel_shape)+1)))
    def test_channels(self, channel_ndim: int):
        interp = self.cls(self.x, self.channeled(self.values), channel_ndim=channel_ndim)
        assert interp.batch_shape == self.channel_shape[:len(self.channel_shape)-interp.channel_ndim]
        assert interp.channel_shape == self.channel_shape[len(self.channel_shape)-interp.channel_ndim:]
        self._test(interp, self.batch_shape + interp.batch_shape)

    def test_batched(self, xy, channels):
        x, y = xy
        interp = self.cls(
            x, self.channeled(y) if channels else y,
            channel_ndim=len(self.channel_shape) * channels)
        assert interp.batch_shape == self.batch_shape
        assert interp.channel_shape == (self.channel_shape if channels else ())
        self._test(interp, ())
        self._test(interp, self.batch_shape + (1,)*interp.batch_ndim)

    @classmethod
    def _make_batched_xs(cls, x):
        return [x.expand(*cls.batch_shape[-1:], -1),
                x.expand(*BaseTestShapes.batch_shape, -1)]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.test_batched = mark.parametrize('xy, channels', tuple(product((
            (cls.x, cls.values.expand(cls.batch_shape + cls.values.shape)),
            *zip(cls._make_batched_xs(cls.x), (
                cls.values.expand(*cls.batch_shape[:-1], 1, *cls.values.shape),
                cls.values
            ))
        ), (True, False))))(cls.test_batched)
