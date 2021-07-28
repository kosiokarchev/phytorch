from functools import partial
from math import inf, nan
from typing import Union

import hypothesis.extra.numpy as npst
import numpy as np
import torch
from hypothesis import given, strategies as st
from hypothesis.strategies import SearchStrategy
from pytest import fixture, mark
from torch import allclose, as_tensor, isclose, tensor

from phytorch.roots import companion_matrix, roots, sroots, vieta
from tests.common import make_dtype_tests, with_default_double


def compare_sets(a, b, **kwargs):
    return isclose(a.unsqueeze(-1), as_tensor(b, dtype=a.dtype).unsqueeze(-2), **kwargs).any(-1).all(-1)


def coeffs_strategy(
        n: Union[int, SearchStrategy[int]] = st.integers(min_value=2, max_value=4),
        dtype=np.complex,
        elements=st.complex_numbers(min_magnitude=1e-6, max_magnitude=1e6, allow_nan=False, allow_infinity=False)):
    if isinstance(n, int):
        return coeffs_strategy(st.integers(min_value=2, max_value=n))
    return n.flatmap(lambda n: npst.mutually_broadcastable_shapes(num_shapes=n, max_dims=3, max_side=16).flatmap(
        lambda shapes: st.tuples(*(npst.arrays(dtype, shape, elements=elements).map(lambda arr: tensor(arr)) for shape in shapes.input_shapes))
    ))


def test_companion_matrix():
    assert (companion_matrix(tensor(-1), tensor(-2), tensor(-3)) == tensor([
        [1, 2, 3],
        [1, 0, 0],
        [0, 1, 0]
    ])).all()


@given(coeffs_strategy())
def test_companion_matrix_batched(coeffs):
    assert companion_matrix(*coeffs).shape == torch.broadcast_shapes(*(c.shape for c in coeffs)) + 2*(len(coeffs),)


@mark.xfail(reason='flaky', strict=False)
@with_default_double
@given(coeffs_strategy())
def test_vieta(coeffs):
    for c, _c in zip(coeffs, vieta(roots(*coeffs))[1:]):
        assert allclose(_c, c, rtol=1e-3, atol=1e-3)


@with_default_double
@given(coeffs_strategy())
def test_analytic_vs_numeric(coeffs):
    assert compare_sets(
        sroots(*coeffs, dim=-1),
        sroots(*coeffs, dim=-1, force_numeric=True),
        rtol=1e-3, atol=1e-3
    ).all()


class RootsTest:
    @mark.parametrize('coeffs, vals', (
        ((0, 0), (0, 0)),
        ((1, 0), (-1, 0)),
        ((0, 1), (-1j, 1j)),

        ((0, 0, 0), (0, 0, 0)),
        ((1, 0, 0), (-1, 0, 0)),
        ((0, 1, 0), (0, 1j, -1j)),
        ((0, 0, 1), (-1, (-1)**(1 / 3), -(-1)**(2 / 3))),
        ((1, 1, 0), (0, (-1)**(2 / 3), -(-1)**(1 / 3))),

        ((0, 0, 0, 0), (0, 0, 0, 0)),
        ((1, 0, 0, 0), (0, -1, 0, -1)),
        ((0, 1, 0, 0), (-1j, 0, 1j, 0)),
        ((0, 0, 1, 0), (-1, 0, (-1)**(1 / 3), -(-1)**(2 / 3))),
        ((0, 0, 0, 1), (-(-1)**(1 / 4), (-1)**(3 / 4), (-1)**(1 / 4), -(-1)**(3 / 4))),
        ((1, 1, 0, 0), (-(-1)**(1 / 3), (-1)**(2 / 3), 0, 0)),
        ((0, 1, 0, 1), (-(-1)**(1 / 3), (-1)**(2 / 3), -(-1)**(2 / 3), (-1)**(1 / 3))),
        ((1, 1, 1, 0), (-1, 0, 1j, -1j))
    ))
    def test_special(self, coeffs, vals):
        assert compare_sets(sroots(*coeffs), vals).all()

    @staticmethod
    def test_finite():
        # Any NaN or infinite coefficient should return NaN
        for n in (2, 3, 4):
            assert sroots(*(n-1)*(1,)+(nan,)).isnan().all()
            assert sroots(*(n-1)*(1,)+(inf,)).isnan().all()


class ForceNumericRootsTest(RootsTest):
    @fixture(autouse=True, scope='class')
    def _set_force_numeric(self):
        # see e.g. https://github.com/pytest-dev/pytest/issues/363
        # for why this workaround is needed
        from _pytest.monkeypatch import MonkeyPatch
        mpatch = MonkeyPatch()
        mpatch.setitem(globals(), 'roots', partial(roots, force_numeric=True))
        yield
        mpatch.undo()

    # @mark.xfail(reason='NaN in eig (https://github.com/pytorch/pytorch/issues/61251)', strict=True)
    @mark.skip(reason='segfaults, so cannot recover...')
    def test_finite(self): ...


globals().update(make_dtype_tests((RootsTest,), 'Roots'))
globals().update(make_dtype_tests((ForceNumericRootsTest,), 'ForceNumericRoots'))
