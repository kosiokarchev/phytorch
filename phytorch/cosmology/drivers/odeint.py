from abc import ABC
from functools import partialmethod
from itertools import chain
from typing import Any, Iterable, Mapping, TYPE_CHECKING

import torch
from more_itertools import first
from torch import Size, Tensor
from torchdiffeq import odeint_adjoint

from .. import special
from ..core import FLRW, FLRWDriver
from ...utils._typing import _TN


class OdeintFLRWDriver(FLRWDriver, ABC):
    odeint_adjoint_kwargs: Mapping[str, Any] = {}

    @property
    def adjoint_params(self) -> Iterable:
        return self.parameters.values()

    @property
    def _param_shape(self) -> Size:
        return torch.broadcast_shapes(*shapes) if (
            shapes := tuple(p.shape for p in self.adjoint_params if torch.is_tensor(p))
        ) else Size()

    def _integrate(self, z1: _TN, z2: _TN, func):
        #
        params = tuple(filter(torch.is_tensor, chain(self.adjoint_params, (z1, z2))))
        p = first(filter(torch.is_floating_point, params), params[0]) if params else None
        dtype, device = (torch.get_default_dtype(), None) if p is None else (
            p.dtype if p.is_floating_point() else torch.get_default_dtype(),
            p.device
        )
        z1, z2 = (
            z if torch.is_tensor(z) else
            torch.tensor(z, dtype=dtype, device=device)
            for z in (z1, z2)
        )  # type: Tensor

        assert z1.numel() == 1
        z = torch.cat(torch.atleast_1d(z1.flatten(), z2.flatten()))

        return odeint_adjoint(
            lambda z, *args: func(self, z),
            y0=z.new_zeros(self._param_shape), t=z,
            adjoint_params=params
        )[1:].reshape((*z2.shape, *self._param_shape))

    if TYPE_CHECKING:
        def comoving_distance_dimless_z1z2(self, z1: _TN, z2: _TN) -> _TN: ...
        def lookback_time_dimless(self, z: _TN) -> _TN: ...
        def absorption_distance_dimless(self, z: _TN) -> _TN: ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.comoving_distance_dimless_z1z2 = partialmethod(cls._integrate, func=cls.inv_efunc)
        cls.lookback_time_dimless = partialmethod(cls._integrate, 0., func=cls.lookback_time_integrand)
        cls.absorption_distance_dimless = partialmethod(cls._integrate, 0., func=cls.abs_distance_integrand)


globals().update(_clss := {
    key: type(key, (OdeintFLRWDriver, obj), {})
    for key in dir(special) for obj in [getattr(special, key, None)]
    if isinstance(obj, type) and issubclass(obj, FLRW)
})
__all__ = *_clss.keys(),
