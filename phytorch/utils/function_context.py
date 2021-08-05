from typing import Any, Callable, ClassVar, Iterable, Optional, Sequence, Union

import torch
from more_itertools import padded
from torch import is_complex, Tensor
from torch.autograd.function import _ContextMethodMixin


class TorchFunctionContext(_ContextMethodMixin):
    saved_tensors: tuple[Tensor, ...]
    needs_input_grad: tuple[bool, ...]


class TorchFunction(torch.autograd.Function):
    differentiable = True
    save_output: ClassVar[bool] = True

    @staticmethod
    def saved_tensors(ctx: TorchFunctionContext, *args: Any) -> Iterable[Tensor]:
        return ()

    @staticmethod
    def _forward(ctx: TorchFunctionContext, *args: Any) -> Tensor:
        raise NotImplementedError

    @classmethod
    def forward(cls, ctx: TorchFunctionContext, *args: Any) -> Any:
        ctx.set_materialize_grads(False)
        saved_tensors = tuple(cls.saved_tensors(ctx, *args))
        output = cls._forward(ctx, *saved_tensors, *args)
        if cls.save_output:
            ctx.save_for_backward(*saved_tensors, output)
        else:
            ctx.save_for_backward(*saved_tensors)
        if not cls.differentiable:
            ctx.mark_non_differentiable(output)
        return output

    # TODO: Python 3.10: ArgSpec
    _gradfunc_T = Callable[[TorchFunctionContext, ...], Optional[Tensor]]
    gradfuncs: ClassVar[Sequence[_gradfunc_T]] = None
    ninputs: ClassVar[int] = None

    @classmethod
    def process_grad(cls, grad: Tensor) -> Tensor:
        return grad

    @classmethod
    def backward(cls, ctx: TorchFunctionContext, *grad_outputs: Optional[Tensor]) -> Union[tuple[Optional[Tensor], ...], Optional[Tensor]]:
        return (
            tuple(padded((
                sum(grad * grad_output for grad_output in grad_outputs)
                if grad is not None else None
                for nig, gf in zip(ctx.needs_input_grad, cls.gradfuncs)
                for grad in [cls.process_grad(gf(ctx, *ctx.saved_tensors)) if nig and gf is not None else None]
            ), None, cls.ninputs))
            if (grad_outputs := [g for g in grad_outputs if g is not None])
            else (cls.ninputs * (None,)))

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.gradfuncs is not None:
            cls.gradfuncs = [
                gf.__get__(None, cls) if isinstance(gf, (staticmethod, classmethod)) else gf
                for gf in cls.gradfuncs]
            if cls.ninputs is None:
                cls.ninputs = len(cls.gradfuncs)
        else:
            cls.differentiable = False


class ComplexTorchFunction(TorchFunction):
    @classmethod
    def process_grad(cls, grad: Tensor) -> Tensor:
        return grad.conj() if is_complex(grad) else grad
