from functools import update_wrapper
from typing import Callable, TypeVar

import torch
from torch import Tensor


complex_typemap = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128
}


def get_default_complex_dtype():
    return complex_typemap[torch.get_default_dtype()]


def to_complex(tensor: Tensor):
    return tensor if torch.is_complex(tensor) else torch.complex(
        tensor := tensor if torch.is_floating_point(tensor) else tensor.to(torch.get_default_dtype()),
        tensor.new_zeros(()))


def as_complex_tensors(*args: Tensor):
    return (to_complex(torch.as_tensor(a)) if a is not None else None for a in args)


_T = TypeVar('_T')
# TODO: Python 3.10: ArgSpec
def with_complex_args(f: _T) -> _T:
    def _f(*args: Tensor):
        return f(*as_complex_tensors(*args))
    return update_wrapper(_f, f)
