from functools import cached_property
from typing import Sequence

import torch
from frozendict import frozendict
from torch import Tensor

from .meta import Meta
from .quantities.torchquantity import TorchQuantity


class RaggedTensorMeta(type(Meta), type(Tensor)):
    pass


class RaggedTensor(Meta, Tensor, metaclass=RaggedTensorMeta):
    sizes: Meta.meta_attribute(Sequence[int])

    def __new__(cls, *args, sizes: Sequence[int], **kwargs):
        return super().__new__(cls, *args, sizes=sizes, **kwargs)

    def _meta_update(self, other: Meta, /, **kwargs):
        if 'sizes' not in kwargs and hasattr(self, 'sizes'):
            kwargs['sizes'] = self.sizes
        return super()._meta_update(other, **kwargs)

    def __torch_function__(self, func, types, args=(), kwargs=frozendict()):
        ret = super().__torch_function__(func, types, args, kwargs)
        return self._meta_update(ret) if isinstance(ret, type(self)) else ret

    @cached_property
    def reduce_indices(self):
        return torch.broadcast_to(
            torch.repeat_interleave(torch.arange(len(self.sizes)).to(self.device), torch.tensor(self.sizes).to(self.device)),
            self.shape)

    @cached_property
    def reduced_shape(self):
        return self.shape[:-1] + (len(self.sizes),)

    def reduce_add(self):
        return self.new_zeros(self.reduced_shape).scatter_add_(-1, self.reduce_indices, self)


class RaggedQuantityMeta(type(TorchQuantity), type(RaggedTensor)):
    pass


class RaggedQuantity(TorchQuantity, RaggedTensor, metaclass=RaggedQuantityMeta):
    def __new__(cls, arg, *args, sizes, unit=None, **kwargs):
        if isinstance(arg, TorchQuantity) and unit is None:
            unit = arg.unit
        return super().__new__(cls, arg, *args, sizes=sizes, unit=unit, **kwargs)
