from torch import Tensor

from ..utils._typing import _TN


def roots(*coeffs: _TN, force_numeric: bool = False) -> tuple[Tensor, ...]: ...
