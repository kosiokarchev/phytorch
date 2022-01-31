import ast
import inspect
import types
from itertools import repeat
from typing import Iterable

import torch
from more_itertools import last
from torch import Tensor


class AutoUnpackable:
    def __iter__(self):
        """ Allows writing `a, b, c = DimSpec()` with automatic counting of targets."""
        frame = inspect.currentframe().f_back
        line, *lines = inspect.getsourcelines(inspect.getmodule(frame))[0][frame.f_lineno-1:]
        line = line.lstrip(' \t')
        lines = lines[::-1]
        while True:
            try:
                stmt = ast.parse(line)
                break
            except SyntaxError:
                line += lines.pop()
        n = len(stmt.body[0].targets[0].elts)
        return repeat(self, n)


def copy_func(f):
    # https://stackoverflow.com/questions/6527633/how-can-i-make-a-deepcopy-of-a-function-in-python/30714299#30714299
    fn = types.FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    fn.__dict__.update(f.__dict__)
    return fn


def _mid_many(a: Tensor, axes: Iterable[int]) -> Tensor:
    axes = [ax % a.ndim for ax in axes]
    return last(
        _a for _a in [a] for ax in axes
        for _a in [torch.narrow(_a, ax, 0, _a.shape[ax]-1) + torch.narrow(_a, ax, 1, _a.shape[ax]-1)]
    ) / 2**len(axes) if axes else a
