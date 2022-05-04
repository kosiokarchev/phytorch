from __future__ import annotations

from abc import ABC

from .. import special
from ..core import FLRW
from ...utils._typing import _TN


class AbstractCosmology(FLRW, ABC):
    """Allows access to properties other than distances (E-function, etc.)"""

    def lookback_time_dimless(self, z: _TN) -> _TN: ...
    def age_dimless(self, z: _TN) -> _TN: ...
    def absorption_distance_dimless(self, z: _TN) -> _TN: ...


globals().update(_clss := {
    key: type(key, (AbstractCosmology, obj), {})
    for key in dir(special) for obj in [getattr(special, key, None)]
    if isinstance(obj, type) and issubclass(obj, FLRW)
})
__all__ = *_clss.keys(),
