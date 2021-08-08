import threading
from inspect import getattr_static
from typing import Any, Callable, ClassVar, Iterable, Optional, Sequence, Union

import torch
from more_itertools import always_iterable, padded
from torch import is_complex, Tensor
from torch.autograd.function import _ContextMethodMixin

from phytorch.utils.complex import as_complex_tensors


class TorchFunctionContext(_ContextMethodMixin):
    saved_tensors: tuple[Tensor, ...]
    needs_input_grad: tuple[bool, ...]


class TorchFunction(torch.autograd.Function):
    devil_may_care = False

    differentiable = False
    save_output: ClassVar[bool] = True

    @staticmethod
    def saved_tensors(ctx: TorchFunctionContext, *args: Any) -> Iterable[Tensor]:
        return args

    @staticmethod
    def _forward(ctx: TorchFunctionContext, *args: Any) -> Union[Tensor, tuple[Tensor, ...]]:
        raise NotImplementedError

    @classmethod
    def forward(cls, ctx: TorchFunctionContext, *args: Any) -> Union[Tensor, tuple[Tensor, ...]]:
        ctx.set_materialize_grads(False)
        saved_tensors = tuple(cls.saved_tensors(ctx, *args))
        output = cls._forward(ctx, *saved_tensors, *args)
        if isinstance(outputs := output, Tensor):
            outputs = (outputs,)
        if cls.save_output:
            ctx.save_for_backward(*saved_tensors, *outputs)
        else:
            ctx.save_for_backward(*saved_tensors)
        if not cls.differentiable:
            ctx.mark_non_differentiable(*outputs)
        # TODO: devil may care
        ctx.devil_may_care = cls.devil_may_care
        return output

    # TODO: Python 3.10: ArgSpec
    _gradfunc_T = Callable[[TorchFunctionContext, ...], Optional[Tensor]]
    _gradfuncs_auto = False
    gradfuncs: ClassVar[Sequence[_gradfunc_T]] = None
    ninputs: ClassVar[int] = None

    @classmethod
    def process_grad(cls, grad: Tensor) -> Tensor:
        return grad

    @classmethod
    def _backward(cls, ctx) -> Optional[Iterable[Optional[Tensor]]]:
        return (
            gf(ctx, *ctx.saved_tensors) if nig and gf is not None else None
            for nig, gf in zip(ctx.needs_input_grad, cls.gradfuncs)
        )

    @classmethod
    def backward(cls, ctx: TorchFunctionContext, *grad_outputs: Optional[Tensor]) -> Union[tuple[Optional[Tensor], ...], Optional[Tensor]]:
        return (
            tuple(padded((
                sum(grad * grad_output for grad_output in grad_outputs)
                if grad is not None else None
                for grad in always_iterable(cls._backward(ctx))
                for grad in [cls.process_grad(grad) if grad is not None else None]
            ), None, cls.ninputs))
            if (grad_outputs := [g for g in grad_outputs if g is not None])
            else (cls.ninputs * (None,)))

    @classmethod
    def _update_gradfuncs(cls): ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.gradfuncs is None or cls._gradfuncs_auto:
            cls.gradfuncs = list({
                key: val for t in cls.mro()[::-1] for key, val in vars(t).items()
                if key.startswith('grad_')
            }.values()) or None
            cls._update_gradfuncs()
            if cls.ninputs is None or cls._gradfuncs_auto:
                cls.ninputs = len(cls.gradfuncs) if cls.gradfuncs else None
            if cls.gradfuncs:
                cls._gradfuncs_auto = True
        if cls.gradfuncs is not None:
            cls.gradfuncs = [
                gf.__get__(None, cls) if isinstance(gf, (staticmethod, classmethod)) else gf
                for gf in cls.gradfuncs]
            if cls.ninputs is None:
                cls.ninputs = len(cls.gradfuncs)
            cls.differentiable = True

        if any(getattr_static(cls, fname) is not getattr_static(TorchFunction, fname)
               for fname in ('backward', '_backward')):
            cls.differentiable = True


class CimplMixin(TorchFunction):
    _impl_func: Callable[[Tensor, ...], Union[Tensor, tuple[Tensor, ...]]]

    @classmethod
    def _forward(cls, ctx, *args):
        return cls._impl_func(*args[:cls.ninputs])


class AlternativeForwardMixin(TorchFunction):
    save_output = False
    differentiable = True

    @staticmethod
    def _alternative_forward(*args):
        raise NotImplementedError

    @classmethod
    def _backward(cls, ctx) -> Optional[Iterable[Optional[Tensor]]]:
        relevant_args = [arg for nig, arg in zip(ctx.needs_input_grad, ctx.saved_tensors) if nig]
        if relevant_args:
            with torch.autograd.enable_grad():
                grads = iter(torch.autograd.grad(
                    cls._alternative_forward(*ctx.saved_tensors), relevant_args,
                    only_inputs=True, create_graph=ctx.requires_grad
                ))
            return (next(grads) if nig else None for nig in ctx.needs_input_grad)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.ninputs is None:
            raise TypeError(
                f'Subclasses of {AlternativeForwardMixin.__name__}'
                'must specify ".ninputs"!')


class CargsMixin(TorchFunction):
    @classmethod
    def process_grad(cls, grad: Tensor) -> Tensor:
        return grad.conj() if is_complex(grad) else grad

    @classmethod
    def apply(cls, *args):
        return super().apply(*as_complex_tensors(*args))
