from typing import Iterable

import torch
from more_itertools import zip_equal
from torch import Tensor

from .dimspec import default_dimspec, dimspecs, forbidden
from .quantity import GenericQuantity
from .signatures import sigdict
from ..units.unit import Unit


class TensorQuantityMeta(type(GenericQuantity), type(Tensor)):
    pass


class TensorQuantity(GenericQuantity[Tensor], Tensor, metaclass=TensorQuantityMeta):
    """A `GenericQuantity` backed by  a `~torch.Tensor`."""

    @property
    def value(self) -> Tensor:
        with self.delegator_context:
            return super().view(self.shape)

    def to(self, unit: Unit = None, *args, **kwargs):
        if unit is not None and not isinstance(unit, Unit):
            args = (unit,) + args
            unit = None
        return super().to(unit, *args, **kwargs)

    # noinspection PyPropertyDefinition
    @classmethod
    @property
    def delegator_context(cls):
        return torch._C.DisableTorchFunction()

    # noinspection PyUnusedLocal
    @classmethod
    def delegate(cls, func, types, args, kwargs):
        with cls.delegator_context:
            return func(*args, **kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in forbidden:
            return NotImplemented

        # TODO: resolve args, kwargs before permuting arguments
        if func is torch.gradient and isinstance(kwargs.setdefault('dim', tuple(range(args[0].ndim))), Iterable):
            kwargs.setdefault('spacing', len(dim := kwargs.pop('dim')) * (1,))
            if not isinstance(spacing := kwargs.pop('spacing'), (tuple, list)):
                spacing = len(dim) * (spacing,)
            return *(torch.gradient(args[0], spacing=sp, dim=dm, **kwargs)[0]
                     for sp, dm in zip_equal(spacing, dim)),
        elif func is torch.where:
            return Tensor.where(args[1], args[0], args[2])

        if fancy := None not in (
                (dimspec := dimspecs.get(func.__name__, default_dimspec)),
                (sig := sigdict.get(func, None))):
            bargs = sig.bind(*args, **kwargs)
            bargs_withdef = sig.bind(*args, **kwargs)
            bargs_withdef.apply_defaults()

            *_args, = map(bargs_withdef.arguments.__getitem__, sig.parameters.keys())
            _nargs, post = dimspec(*_args)
            for key, _arg, _narg in zip(sig.parameters.keys(), _args, _nargs):
                if key in bargs.arguments.keys() or _arg is not _narg:
                    bargs.arguments[key] = _narg
            args, kwargs = bargs.args, bargs.kwargs

        ret = super().__torch_function__(func, tuple(
            t for t in types if t is not Unit
        ), args, kwargs)

        return post(ret) if fancy else ret
