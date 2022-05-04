from inspect import signature
from typing import Generic, Type, TypeVar

import torch
from torch.nn import Module

from .core import Cosmology


_T = TypeVar('_T', bound=Cosmology)


class CosmologyModule(Module, Generic[_T]):
    def __init__(self, cls: Type[_T], method: str, unsqueeze=0):
        super().__init__()
        self.obj = cls.__new__(cls)

        self.method = getattr(self.obj, method)
        self.method_kwarg_names = tuple(
            name
            for name, param in signature(self.method).parameters.items()
            if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        )

        self.unsqueeze = unsqueeze

    def forward(self, *args, **kwargs):
        method_kwargs = {
            key: kwargs.pop(key)
            for key in self.method_kwarg_names
            if key in kwargs
        }
        self.obj._set_params(**{
            key: val.reshape(*val.shape, *self.unsqueeze*(1,))
            if torch.is_tensor(val) else val
            for key, val in kwargs.items()
        })

        return self.method(*args, **method_kwargs)
