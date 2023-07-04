"""Classes for univariate and multidimensional interpolation. Batched, of course."""

import torch

from .Linear1dInterpolator import Linear1dInterpolator
from .LinearNDGridInterpolator import LinearNDGridInterpolator


def interp2d(x, y, z, extent_x, extent_y, **kwargs):
    """

    Parameters
    ----------
    x: (batch_shape...)
    y: (batch_shape...)
    z: (batch_shape..., N_x, N_y)
    extent_x
    extent_y

    Returns
    -------

    """
    batch_shape = torch.broadcast_shapes(x.shape, y.shape, x.shape[:-2])
    return torch.nn.functional.grid_sample(
        z.unsqueeze(-3).expand(*batch_shape, 1, *z.shape[-2:]).flatten(end_dim=-4).swapaxes(-2, -1),
        torch.stack(torch.broadcast_tensors(
            2 * (x - extent_x[0]) / (extent_x[1] - extent_x[0]) - 1,
            2 * (y - extent_y[0]) / (extent_y[1] - extent_y[0]) - 1,
        ), -1).expand(*batch_shape, 2).reshape(-1, 1, 1, 2),
        **kwargs
    ).reshape(batch_shape)
