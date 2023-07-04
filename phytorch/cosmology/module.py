from inspect import signature
from typing import Generic, Type, TypeVar

import torch
from torch.nn import Module

from .core import Cosmology


_T = TypeVar('_T', bound=Cosmology)


class AbstractCosmologyModule(Module, Generic[_T]):
    def __init__(self, cls: Type[_T], unsqueeze=0):
        super().__init__()
        self.obj = cls.__new__(cls)
        cls.__init__.__wrapped__(self.obj)
        self.unsqueeze = unsqueeze

    def set_params(self, **kwargs) -> _T:
        return self.obj._set_params(**{
            key: val.reshape(*val.shape, *self.unsqueeze*(1,))
            if torch.is_tensor(val) else val
            for key, val in kwargs.items()
        })


class CosmologyModule(AbstractCosmologyModule[_T], Generic[_T]):
    def __init__(self, cls: Type[_T], method: str, unsqueeze=0):
        super().__init__(cls, unsqueeze)

        self.method = getattr(self.obj, method)
        self.method_kwarg_names = tuple(
            name
            for name, param in signature(self.method).parameters.items()
            if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        )

    def forward(self, *args, **kwargs):
        method_kwargs = {
            key: kwargs.pop(key)
            for key in self.method_kwarg_names
            if key in kwargs
        }
        self.set_params(**kwargs)
        return self.method(*args, **method_kwargs)
