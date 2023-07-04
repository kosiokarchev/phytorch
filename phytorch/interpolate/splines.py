import operator
from dataclasses import dataclass
from functools import cached_property
from itertools import chain, starmap
from typing import Sequence

import torch
from more_itertools import ichunked
from torch import LongTensor, Tensor

from phytorchx import polyval, broadcast_except, broadcast_gather

from ..utils.symmetry import product


def polyderiv_coeff(polydegree, derivdegree):
    return product(range(polydegree, polydegree-derivdegree, -1))


@dataclass
class Spline:
    """Interpolating univariate spline.

    Attributes
    ----------
    x0: `~torch.Tensor` ``(batch_shape..., N)``
        locations of the knots along the *last* dimension.
        *Must be sorted!*
    degree
        degree of the spline. Only ``degree = 3`` works flawlessly for now.

    npoints: int
        ``= N``: number of knots
    npolys: int
        ``= N-1``: number of interpolating polynomials
    nvarsp: int
        ``= degree + 1``: number of coefficients per polynomial
    nvars: int
        ``= nvarsp * npolys``: total number of coefficients
    batch_shape: torch.Size
        ``= x0.shape[:-1]``: batch shape of the knots
    """

    x0: Tensor
    degree: int = 3

    def __post_init__(self):
        self.npoints = self.x0.shape[-1]
        self.npolys = self.npoints - 1
        self.nvarsp = self.degree + 1
        self.nvars = self.nvarsp * self.npolys

        self.batch_shape = self.x0.shape[:-1]

    def _new_zeros(self, shape: tuple):
        return self.x0.new_zeros(()).expand(shape)

    @cached_property
    def X0(self) -> Tensor:
        """Linear operator representing *all* constraint equations.

        Returns
        -------
        `~torch.Tensor` ``(batch_shape..., nvars, nvars)``
        """

        x0 = self.x0.movedim(-1, 0)
        x0p = tuple(x0**i for i in range(self.degree + 1))
        return torch.stack([
            torch.cat((
                torch.stack([
                    x0p[d][_i] for d in range(self.degree, -1, -1)
                ]),
                self._new_zeros((self.nvars-self.nvarsp, *self.batch_shape))
            ), dim=0).roll(i * self.nvarsp, 0)
            for i in range(self.npolys)
            for _i in (i, i+1)
        ] + [
            torch.cat((
                torch.stack([
                    m * polyderiv_coeff(d, dd) * x0p[d-dd][i] if d>=dd else self._new_zeros(self.batch_shape)
                    for m in (1, -1) for d in range(self.degree, -1, -1)
                ]),
                self._new_zeros((self.nvars - 2 * self.nvarsp, *self.batch_shape))
            ), dim=0).roll((i-1) * self.nvarsp, 0)
            for dd in range(1, self.degree)
            for i in range(1, self.npolys)
        ] + [
            torch.cat((
                torch.stack([
                    polyderiv_coeff(d, dd) * x0p[d-dd][i] if d>=dd else self._new_zeros(self.batch_shape)
                    for d in range(self.degree, -1, -1)]),
                self._new_zeros((self.nvars - self.nvarsp, *self.batch_shape))
            ), dim=0).roll(i * self.nvarsp, 0)
            for dd in range(self.degree - 1, 1, -1)
            for i in (0, -1)
        ]).movedim((0, 1), (-2, -1))

    @cached_property
    def A(self) -> Tensor:
        """Linear operator, which multiplies the values to get the coefficients:

        .. math:
            A y_0 = c

        It is a `nvars`-by-`npoints` matrix calculated as a linear combination
        of the first ``2 * npolys`` columns of the inverse of `X0`.

        Returns
        -------
        `~torch.Tensor` ``(batch_shape..., nvars, npoints)``
        """
        X0inv = self.X0.inverse()[..., :2*self.npolys]
        return torch.stack((
            X0inv[..., 0], *starmap(
                operator.add, ichunked(X0inv[..., 1:-1].unbind(-1), 2)
            ), X0inv[..., -1]
        ), -1)

    def pick_poly(self, x: Tensor) -> LongTensor:
        """Index of the polynomial that each element of :arg:`x` corresponds to.

        Parameters
        ----------
        x: `~torch.Tensor` ``(batch..., M)``
            Points at which interpolation is to be performed.
            Can have a last dimension of arbitrary length. The rest need to
            broadcast with the `batch_shape`.

        Returns
        -------
        `~torch.Tensor` ``(additional_batch..., batch_shape..., M)``
            `~torch.LongTensor` of the (broadcasted) shape of :arg:`x` which
            indexes into the polynomials that comprise the spline. Points
            outside the range of `x0`'s are extrapolated, i.e. assigned to the
            boundary polynomials.
        """
        return torch.searchsorted(*broadcast_except(self.x0, x)).sub_(1).clamp_(min=0, max=self.x0.shape[-1]-2)

    def Y0(self, y0: Tensor) -> Tensor:
        """Right-hand side of *all* constraint equations in the appropriate order.

        Parameters
        ----------
        y0: `~torch.Tensor` ``(batch..., npoints)``
            Values of the interpolated function.

        Returns
        -------
        `~torch.Tensor` ``(batch..., nvars)``

            Vector(s) of length `nvars` consisting of

            - the first end-point,
            - duplicated internal points,
            - the last end-point,
            - zero padding.
        """
        return torch.cat((y0.movedim(-1, 0).repeat_interleave(2, dim=0)[1:-1], y0.new_zeros((self.nvars - 2*self.npolys, *y0.shape[:-1])))).movedim(0, -1)

    def coeffs(self, y0: Tensor) -> Tensor:
        """Coefficients for the interpolating polynomials.

        Parameters
        ----------
        y0: `~torch.Tensor` ``(batch..., npoints)``
            Values of the interpolated function. Must broadcast with `x0`.

        Returns
        -------
        `~torch.Tensor` ``(batch..., npolys, nvarsp)``
            Solution to :math:`X_0 c = Y_0`, where :math:`X_0` and :math:`Y_0`
            are constructed from :math:`x_0` and :math:`y_0`, respectively.
        """
        return torch.linalg.solve(self.X0, self.Y0(y0).unsqueeze(-1)).squeeze(-1).unflatten(-1, (self.npolys, self.nvarsp))

    def pick_coeffs(self, x: Tensor, y0: Tensor) -> Sequence[Tensor]:
        """Coefficients of the interpolating polynomials, corresponding to each point in :arg:`x`.

        Parameters
        ----------
        x: `~torch.Tensor` ``(batch..., M)``
            Points at which interpolation is to be performed.
            Can have a last dimension of arbitrary length. The rest need to
            broadcast with the `batch_shape`.
        y0: `~torch.Tensor` ``(batch..., npoints)``
            Values of the interpolated function. Must broadcast with `x0`.

        Returns
        -------
        `~torch.Tensor` ``(nvarsp, batch..., M)``
            Coefficients (in the first dimension) corresponding to the
            particular polynomial that covers each point in :arg:`x`. The spline
            can then be evaluated by using `polyval`:

            .. code:: python3

                polyval(Spline(x0).pick_coeffs(x, y0), x)

            which is exactly the code of `evaluate`.
        """
        idx = self.pick_poly(x)
        return broadcast_gather(self.coeffs(y0), -2, idx).movedim(-1, 0)

    def weights(self, x: Tensor):
        r"""Weights that directly combine :math:`\{y_{0, i}\}` into :math:`y(x)`.

        Parameters
        ----------
        x: `~torch.Tensor` ``(batch..., M)``
            Points at which interpolation is to be performed.
            Can have a last dimension of arbitrary length. The rest need to
            broadcast with the `batch_shape`.

        Returns
        -------
        `~torch.Tensor` ``(batch..., M, npoints)``
            Weights (in the last dimension) that multiply a given :math:`y_0`
            to give :math:`y(x)` at each given :arg:`x`. This leads to an
            alternative way to evaluate the spline:

            .. math::
                y(x) = w(x_0, x)^T y_0.
        """
        return polyval(broadcast_gather(self.A.unflatten(-2, (self.npolys, self.nvarsp)), -3, self.pick_poly(x)).movedim((-2, -1), (0, 1)), x).movedim(0, -1)

    def evaluate(self, x: Tensor, y0: Tensor):
        """Evaluate the spline (that interpolates :arg:`y0` from :arg:`x0`) at :arg:`x`.

        Parameters
        ----------
        x: `~torch.Tensor` ``(batch..., M)``
            Points at which interpolation is to be performed.
            Can have a last dimension of arbitrary length. The rest need to
            broadcast with the `batch_shape`.
        y0: `~torch.Tensor` ``(batch..., npoints)``
            Values of the interpolated function. Must broadcast with `x0`.

        Returns
        -------
        `~torch.Tensor` ``(batch..., M)``
            The spline which passes through ``(x0, y0)`` evaluated at :arg:`x`.
        """
        return polyval(self.pick_coeffs(x, y0), x)


class SplineNd:
    r"""N-dimensional spline as the product of univariate `Spline`\ s.

    Parameters
    ----------
    x0s
        ``[(batch_shape_1..., npoints_1), (batch_shape_2..., npoints_2)), ...]``
        knots in each dimension. The batch shapes should broadcast.
    degree
        degree of the spline. Passed to the univariate `Spline`\ s.

    Attributes
    ----------
    splines: list[Spline]
        the individual univariate `Spline`\ s
    grid_shape: torch.Size
        shape of the grid: ``(npoints_1, npoints_2, ...)``
    batch_shape: torch.Size
        broadcasted `~Spline.batch_shape` of all the 1D grids.
    ndim: int
        ``= len(x0s) = len(splines) = len(grid_shape)``: number of dimensions
    """
    def __init__(self, *x0s: Tensor, degree: int = 3):
        self.splines = [Spline(_x0, degree=degree) for _x0 in x0s]
        self.grid_shape = torch.Size((_x0.shape[-1] for _x0 in x0s))
        self.batch_shape = torch.broadcast_shapes(*(_x0.shape[:-1] for _x0 in x0s))
        self.ndim = len(self.grid_shape)

    def evaluate(self, y0, *xs):
        r"""Evaluate the n-dimensional spline as a multilinear "product" of 1-dim splines:

        .. math::
            y(x_1, x_2, ...) = \sum_{i=1, j=1, \ldots}^{N_1, N_2, \ldots} y_{0, ij\ldots} w_{1,i}(x_1) w_{2,i}(x_2) \ldots

        Parameters
        ----------
        y0: `~torch.Tensor` ``(batch..., npoints_1, npoints_2, ...)``
            Values of the interpolated function as a grid in the last `ndim`
            dimensions. The rest must broadcast with `batch_shape` and with the
            batch shapes of :arg:`xs`.
        xs
            ``[(batch..., M), (batch..., M), ...]``
            Coordinates of the points where the spline is to be evaluated.
            Last dimension must match, and the batch dimensions must broadcast
            among each other, with :arg:`y0`, and with `batch_shape`.

        Returns
        -------
        `~torch.Tensor` ``(batch..., M)``
            The spline evaluated at ``xs = (x_1, x_2, ...)``.
        """
        return torch.einsum(*chain.from_iterable(
            (s.weights(x), [..., i]) for i, (s, x) in enumerate(zip(self.splines, xs))
        ), y0, [..., *range(self.ndim)])
