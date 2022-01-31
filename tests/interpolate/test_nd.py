import torch

from phytorch.interpolate import LinearNDGridInterpolator
from tests.interpolate.common import BaseTestShapes


class TestShapes(BaseTestShapes):
    x = grids = [torch.linspace(0, 1, 11), torch.linspace(0, 1, 12), torch.linspace(0, 1, 13)]
    values = torch.rand(*map(len, grids))

    ndim = len(grids)
    cls = LinearNDGridInterpolator

    @classmethod
    def _make_batched_xs(cls, x):
        return zip(*map(super()._make_batched_xs, cls.grids))
