import torch
from scipy import interpolate as spi

from phytorch.interpolate import Linear1dInterpolator
from tests.interpolate.common import BaseTestShapes


def test_basic():
    x = torch.linspace(2.6, 4.2, 13) + torch.rand(13) * 0.1
    y = torch.rand_like(x)

    xout = 3 + torch.rand_like(x) * (4-3)

    assert torch.allclose(
        Linear1dInterpolator(x, y)(xout),
        torch.tensor(spi.interp1d(x.numpy(), y.numpy())(xout.numpy()))
    )


class TestShapes(BaseTestShapes):
    ndim = 1
    explicit_dim = False
    cls = Linear1dInterpolator
    x = torch.linspace(0, 1, 51)
    values = torch.rand(x.shape[-1])
