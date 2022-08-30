from functools import partial
from numbers import Number
from typing import Callable, Iterable, Mapping

import torch
from pytest import mark
from torch import Tensor

from phytorch.utils import copy_func
from tests.common.closeness import close_complex_nan


def process_cases(func, vals):
    ins, outs = map(torch.tensor, zip(*vals))
    return ins, outs, func(*ins.T)


class BaseCasesTest:
    @staticmethod
    def parametrize(cases: Mapping[Callable[..., Tensor], Iterable[tuple[Iterable, Number]]]):
        return mark.parametrize('func, args, truth', tuple(
            (func, args, truth)
            for func, vals in cases.items() for args, truth in vals
        ))(
            # TODO: pytest fails with "duplicate 'func'"
            #  if you run multiple files that have dtyped cases
            #  if BaseCasesTest.test_case is not copied?!?....
            copy_func(BaseCasesTest.test_case),
        )

    # noinspection PyMethodMayBeStatic
    def test_case(self, func, args, truth):
        assert close_complex_nan(res := func(*args), torch.tensor(truth, dtype=res.dtype)), res.item()
