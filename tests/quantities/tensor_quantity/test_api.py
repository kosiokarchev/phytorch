import itertools
import operator
import string
from abc import ABC, abstractmethod
from functools import partial, reduce, update_wrapper
from itertools import chain, repeat, starmap
from random import choices
from typing import Callable, Iterable, Sequence, TYPE_CHECKING, Union

import pytest
import torch
from hypothesis import given, strategies as st
from more_itertools import consume, take
from pytest import fixture, mark, raises
from torch import Tensor

from phytorch.quantities.quantity import GenericQuantity
from phytorch.units.angular import deg, radian
from phytorch.units.exceptions import UnitError
from phytorch.units.si import centimeter, gram, second
from phytorch.units.unit import Dimension, Unit
from phytorch.utils import copy_func, pytree

from tests.common.strategies.tensors import random_tensors
from tests.common.strategies.units import dimful_units_strategy, units_strategy
from tests.quantities.quantity_utils import _nestedVT, _VT, ConcreteQuantity


def deterministic(func):
    def f(*args, **kwargs):
        torch.manual_seed(42)
        return func(*args, **kwargs)
    return update_wrapper(f, func)


def with_complex_arg(func):
    def f(arg, *args, **kwargs):
        return func(arg*(1+1j), *args, **kwargs)
    return update_wrapper(f, func)


def postpartial(func: Callable, *args, **kwargs):
    return update_wrapper((lambda *_args, **_kwargs: func(*_args, *args, **_kwargs, **kwargs)), func)


def postpartials(funcs: Iterable[Callable], *args, **kwargs):
    return (postpartial(f, *args, **kwargs) for f in funcs)


def partialparam(func: Callable, *args, **kwargs):
    return pytest.param(partial(func, *args, **kwargs), id=func.__name__)


def kwargsparam(func, **kwargs):
    return pytest.param((func, kwargs), id=func.__name__)


_tfparam = lambda args, i=0: pytest.param(*args, id=args[i].__name__)

_wrap_addmm = lambda f: update_wrapper(lambda *args: f(*args[:3], beta=args[3], alpha=args[4]), f)

_indexed_rearg = lambda f: update_wrapper(lambda val, index, val2: f(val, -1, index, val2), f)


class TestQfuncsBase:
    shape = (6,)
    expanded_shape = (3, 6)
    reshaped_shape = (2, 3)
    chunks = 3

    def _val(self):
        return torch.rand(self.shape)

    def _val_matrix(self):
        return self._val().diag_embed()

    def _unit(self, unit):
        return unit

    val = fixture(scope='class', name='val')(_val)
    val2 = fixture(scope='class', name='val2')(_val)
    val3 = fixture(scope='class', name='val3')(_val)
    val_matrix = fixture(scope='class', name='val_matrix')(_val_matrix)
    val2_matrix = fixture(scope='class', name='val2_matrix')(_val_matrix)
    val3_matrix = fixture(scope='class', name='val3_matrix')(_val_matrix)

    unit = fixture(scope='class', name='unit')(lambda self: self._unit(gram))
    unit2 = fixture(scope='class', name='unit2')(lambda self: self._unit(centimeter))
    unit3 = fixture(scope='class', name='unit3')(lambda self: self._unit(second))
    unit4 = fixture(scope='class', name='unit4')(lambda self: self._unit(Unit({Dimension('K'): 1}, value=42.)))

    if TYPE_CHECKING:
        @fixture
        def val(self) -> _VT: ...
        @fixture
        def val2(self) -> _VT: ...
        @fixture
        def val2(self) -> _VT: ...
        @fixture
        def val_matrix(self) -> _VT: ...
        @fixture
        def val2_matrix(self) -> _VT: ...
        @fixture
        def val3_matrix(self) -> _VT: ...

        @fixture()
        def unit(self) -> Unit: ...
        @fixture()
        def unit2(self) -> Unit: ...
        @fixture()
        def unit3(self) -> Unit: ...

    def q(self, val, unit):
        return val * unit

    @fixture(scope='class', name='q')
    def q_fixture(self, val, unit):
        return self.q(val, unit)

    @fixture(scope='class', name='q2')
    def q2_fixture(self, val2, unit2):
        return self.q(val2, unit2)

    @fixture(scope='class')
    def q_matrix(self, val_matrix, unit):
        return self.q(val_matrix, unit)

    @fixture(scope='class')
    def q2_matrix(self, val2_matrix, unit2):
        return self.q(val2_matrix, unit2)

    @staticmethod
    def _compare(a1, a2, assert_values=True):
        assert type(a1) == type(a2)
        if isinstance(a1, GenericQuantity):
            assert a1.unit.dimension == a2.unit.dimension
        if assert_values:
            if isinstance(a1, Tensor):
                assert a1.allclose(a2)
            else:
                assert a1 == a2

    @classmethod
    def _tree_compare(cls, val1, val2, assert_values=True):
        args1, tree1 = pytree.tree_flatten(val1)
        args2, tree2 = pytree.tree_flatten(val2)
        assert tree1 == tree2

        consume(starmap(partial(cls._compare, assert_values=assert_values), zip(args1, args2)))

    @mark.parametrize('func, out_unit', tuple(map(_tfparam, (
        (with_complex_arg(torch.angle), radian),
    ))))
    def test_to_unit(self, func: Callable, out_unit: Unit, val: _VT, unit: Unit):
        self._compare(func(val/2 * (unit * 2)), func(val) * out_unit)


class TestForbidden(TestQfuncsBase):
    @mark.parametrize('func', (
        torch.bincount, torch.deg2rad, torch.rad2deg,
        torch.norm, postpartial(torch.renorm, 1, 0, 1), torch.frexp,
        torch.prod, postpartial(torch.cumprod, -1),
        torch.bitwise_not,
    ))
    def test_forbidden(self, func, q):
        with raises(TypeError, match='no implementation'):
            func(q)

    @mark.parametrize('func', (
        torch.gcd, torch.lcm,
        torch.bitwise_and, torch.bitwise_or, torch.bitwise_xor,
        torch.bitwise_left_shift, torch.bitwise_right_shift,
    ))
    def test_forbidden_binary(self, func, q, q2):
        for args in ((q, q2), (q, q2.value), (q.value, q2)):
            with raises(TypeError, match='no implementation'):
                func(*args)


class TestCustom(TestQfuncsBase):
    @mark.parametrize('func', (torch.det,))
    def test_det(self, func, q_matrix):
        self._compare(func(q_matrix), func(q_matrix.value) * q_matrix.unit**q_matrix.shape[-1])

    @mark.parametrize('func', (torch.histc,))
    def test_histc(self, func, q: ConcreteQuantity):
        self._compare(
            func(q, bins=10, min=q.min(), max=q.max()),
            func(q.value, bins=10, min=q.value.min(), max=q.value.max())
        )

        with raises(UnitError):
            torch.histc(q)

    @mark.parametrize('func', (torch.polar,))
    def test_polar(self, func, val, unit, val2):
        res = func(val, val2)
        self._compare(func(val * unit, val2 * radian), res * unit)
        self._compare(func(val/2 * (2*unit), val2), res * unit)
        self._compare(func(val, val2 * Unit()), res * Unit())
        with raises(UnitError):
            func(val, val2 * unit)

    @mark.parametrize('func', (torch.copysign,))
    def test_copysign(self, func, val, val2, unit, unit2):
        res = func(val, val2)
        self._compare(func(val * unit, val2), res * unit)
        self._compare(func(val * unit, val2 * unit2), res * unit)
        self._compare(func(val, val2 * unit2), res)

    @mark.parametrize('func', (torch.ldexp, torch.xlogy))
    def test_unit_dimless_unit(self, func, val, val2, unit):
        res = func(val, val2)
        self._compare(func(val * unit, val2), res * unit)
        self._compare(func(val/2 * (2*unit), val2), res * unit)
        self._compare(func(val, val2 * Unit()), res * Unit())
        with raises(UnitError):
            func(val, val2 * unit)
        with raises(UnitError):
            func(val * unit, val2 * unit)

    @mark.parametrize('func', (torch.prelu,))
    def test_unit_dimless_unit_scalar(self, func, val, unit):
        self.test_unit_dimless_unit(func, val, torch.tensor(0.5), unit)

    @mark.parametrize('func', (torch.lerp,))
    def test_unit_unit_dimless(self, func, val, val2, unit, unit2):
        val3 = torch.rand_like(val2)
        res = func(val, val2, val3)
        self._compare(func(val * unit, val2 * unit, val3), res * unit)
        self._compare(func(val * unit, val2 * unit, val3 * Unit()), res * unit)
        self._compare(func(val, val2/2 * (2*Unit()), val3 * Unit()), res * Unit())
        self._compare(func(val, val2, val3 * Unit()), res * Unit())
        with raises(UnitError):
            func(val * unit, val2 * unit2, val3)
        with raises(UnitError):
            func(val * unit, val2, val3)
        with raises(UnitError):
            func(val, val2, val3 * unit)

    @mark.parametrize('func', (torch.heaviside,))
    def test_heaviside(self, func, val, val2, unit, unit2):
        res = func(val, val2)
        self._compare(func(val * unit, val2), res)
        self._compare(func(val * unit, val2 * Unit()), res)
        self._compare(func(val, val2 * Unit()), res)
        with raises(UnitError):
            func(val, val2 * unit2)

    @mark.parametrize('func, transform', tuple(map(_tfparam, (
        *((f, lambda u: u**2) for f in (torch.square, torch.var, torch.cov)),
        (torch.sqrt, lambda u: u**(1/2)), (torch.rsqrt, lambda u: u**(-1/2)),
        (torch.reciprocal, lambda u: 1/u),
    ))))
    def test_lambda(self, func: Callable, transform: Callable[[Unit], Unit], val, unit):
        self.test_to_unit(func, transform(unit), val, unit)

    @mark.parametrize('func, transform', tuple(map(_tfparam, (
        *((f, lambda u: 1/u) for f in (torch.inverse, torch.pinverse)),
        (torch.cholesky, lambda u: u**(1/2)), (torch.cholesky_inverse, lambda u: 1/u**2),
    ))))
    def test_lambda_matrix(self, func: Callable, transform: Callable[[Unit], Unit], val_matrix, unit):
        self.test_lambda(func, transform, val_matrix, unit)

    @mark.parametrize('func', (
        torch.solve, torch.lstsq,
        update_wrapper(lambda B, A: torch.lu_solve(B, *torch.lu(A)), torch.lu_solve),
        update_wrapper(lambda B, A: torch.cholesky_solve(B, torch.cholesky(A)), torch.cholesky_solve)
    ))
    def test_solve(self, func: Callable, val_matrix, val2_matrix, unit, unit2):
        res = func(val_matrix, val2_matrix)

        def res_units(u1, u2):
            return type(res)(
                (res[0] * u1, res[1] * u2)
            ) if isinstance(res, tuple) else res * u1

        self._tree_compare(func(val_matrix * unit, val2_matrix * unit2),
                           res_units(unit/unit2, unit2))
        self._tree_compare(func(val_matrix * unit, val2_matrix),
                           res_units(unit, Unit()))
        self._tree_compare(func(val_matrix, val2_matrix * unit2),
                           res_units(1/unit2, unit2))
        self._tree_compare(func(val_matrix, val2_matrix/2 * (2*Unit())),
                           res_units(Unit(), Unit()))

    @mark.parametrize('func', (torch.triangular_solve,))
    def test_solve_triangular(self, func, val_matrix, val2_matrix, unit, unit2):
        self.test_solve(func, val_matrix.tril(), val2_matrix, unit, unit2)

    @mark.parametrize('transform, func', tuple(map(partial(_tfparam, i=1), (
        (lambda res, u: (res[0] * u**2, res[1] * u), torch.var_mean),
        (lambda res, u: (res[0], res[1] * u), torch.histogram)
    ))))
    def test_tree_to_units(self, transform: Callable[[tuple, Unit], tuple], func: Callable, val, unit, **kwargs):
        if isinstance(func, tuple):
            func, kwargs = func
        res = func(val)
        self._tree_compare(func(val * unit), type(res)(transform(res, unit)), **kwargs)
        self._tree_compare(func(val/2 * (2*Unit())), type(res)(transform(res, Unit())), **kwargs)

    @mark.parametrize('transform, func', (*map(partial(_tfparam, i=1), (
        *((lambda res, unit: (res[0] * unit, res[1]), f)
          for f in (torch.lu, torch.eig, torch.symeig)),
        (lambda res, unit: (res[:2] + (res[2] * unit,)),
         update_wrapper(lambda A: torch.lu_unpack(*torch.lu(A)), torch.lu_unpack)),
        (lambda res, unit: (res[0], res[1] * unit),
         torch.qr),
        (lambda res, unit: (res[0], res[1] * unit, res[2]), torch.svd)
    )), *(
        pytest.param(lambda res, unit: (res[0], res[1] * unit, res[2]), (f, dict(assert_values=False)), id=f.__name__)
        for f in (torch.svd_lowrank, torch.pca_lowrank)
    )))
    def test_tree_to_units_matrix(self, transform: Callable[[tuple, Unit], tuple], func: Callable, val_matrix, unit, **kwargs):
        self.test_tree_to_units(transform, func, val_matrix, unit, **kwargs)

    @mark.parametrize('func, op', tuple(map(_tfparam, starmap(
        (lambda f, op: (
            update_wrapper(lambda *args: f(*args[:3], value=args[3]), f),
            op
        )), ((torch.addcmul, lambda unit, unit2, unit3: unit / (unit2 * unit3)),
             (torch.addcdiv, lambda unit, unit2, unit3: unit * unit3 / unit2))
    ))))
    def test_addc(self, func: Callable, op: Callable[..., Unit], val, val2, val3, unit, unit2, unit3):
        value = torch.tensor(0.5)
        res = func(val, val2, val3, value)
        self._compare(func(val * unit, val2 * unit2, val3 * unit3, value * op(unit, unit2, unit3)), res * unit)
        self._compare(func(val, val2 * unit2, val3 * unit3, value * op(Unit(), unit2, unit3)), res * Unit())
        self._compare(func(val/2 * (2*Unit()), val2, val3, value), res * Unit())
        self._compare(func(val * Unit(), val2, val3, value/2 * (2*Unit())), res * Unit())

        with raises(UnitError):
            func(val, val2, val3 * unit, value)

    def _test_add_multiply(self, func, val, val2, val3, unit, unit2, unit3, unit4):
        v1, v2 = torch.rand(2)
        res = func(val, val2, val3, v1, v2)
        self._compare(func(val * unit, val2 * unit2, val3 * unit3, v1 * unit4, v2 * (unit * unit4 / (unit2 * unit3))), res * unit * unit4)
        self._compare(func(val * unit, val2, val3, v1 / unit, v2), res * Unit())
        self._compare(func(val/2 * (2*Unit()), val2, val3, v1, v2), res * Unit())
        self._compare(func(val * Unit(), val2, val3, v1/2 * (2*Unit()), v2/3 * (3*Unit())), res * Unit())

        with raises(UnitError):
            func(val, val2, val3 * unit, v1, v2)

    @mark.parametrize('func', tuple(map(_wrap_addmm, (
        torch.addmm,
        *((lambda f: update_wrapper(lambda *args, **kwargs: f(args[0], *(
            v.expand(3, *v.shape) for v in args[1:3]
        ), **kwargs), f))(f) for f in (torch.addbmm, torch.baddbmm))
    ))))
    def test_add_multiply(self, func, val_matrix, val2_matrix, val3_matrix, unit, unit2, unit3, unit4):
        self._test_add_multiply(func, val_matrix, val2_matrix, val3_matrix, unit, unit2, unit3, unit4)

    @mark.parametrize('func', tuple(map(_wrap_addmm, (torch.addmv,))))
    def test_addmv(self, func, val, val2_matrix, val3, unit, unit2, unit3, unit4):
        self._test_add_multiply(func, val, val2_matrix, val3, unit, unit2, unit3, unit4)

    @mark.parametrize('func', tuple(map(_wrap_addmm, (torch.addr,))))
    def test_addr(self, func, val_matrix, val2, val3, unit, unit2, unit3, unit4):
        self._test_add_multiply(func, val_matrix, val2, val3, unit, unit2, unit3, unit4)


class SelectorTestBase(TestQfuncsBase, ABC):
    @abstractmethod
    def _selector(self): ...

    @fixture(scope='class')
    def selector(self):
        return self._selector()

    unary_funcs: Sequence[Callable[[_VT, _VT], _VT]] = ()
    binary_funcs: Sequence[Callable[[_VT, _VT, _VT], _VT]] = ()
    binary_scalar_funcs: Sequence[Callable[[_VT, _VT, _VT], _VT]] = ()

    def __init_subclass__(cls, **kwargs):
        # wtf, doesn't work with partial...
        cls.test_unary = mark.parametrize('func', cls.unary_funcs)(copy_func(cls.test_unary))
        cls.test_binary = mark.parametrize('func', cls.binary_funcs)(copy_func(cls.test_binary))
        cls.test_binary_scalar = mark.parametrize('func', cls.binary_scalar_funcs)(copy_func(cls.test_binary_scalar))


    def test_unary(self, func, selector, val, unit):
        res = func(val, selector)
        self._compare(func(val * unit, selector), res * unit)
        self._compare(func(val, selector * unit), res)

    def test_binary(self, func, selector, val, val2, unit, unit2):
        res = func(val, selector, val2)
        self._compare(func(val * unit, selector, val2 * unit), res * unit)
        self._compare(func(val * unit, selector * unit2, val2 * unit), res * unit)
        self._compare(func(val, selector, val2 / 2 * (2 * Unit())), res * Unit())
        self._compare(func(val * Unit(), selector, val2), res * Unit())
        self._compare(func(val, selector * unit, val2), res * Unit())

        with raises(UnitError):
            func(val * unit, selector, val2 * unit2)

    def test_binary_scalar(self, func, selector, val, unit, unit2):
        self.test_binary(func, selector, val, torch.tensor(0.5), unit, unit2)


class TestMasked(SelectorTestBase):
    def _selector(self):
        return torch.randint(0, 2, self.shape, dtype=torch.bool)

    unary_funcs = (torch.masked_select,)
    binary_funcs = (torch.Tensor.where, torch.masked_scatter)
    binary_scalar_funcs = (torch.masked_fill,)


class TestIndexed(SelectorTestBase):
    def _selector(self):
        return torch.randperm(self.shape[-1])

    unary_funcs = (
        *((lambda f: update_wrapper(lambda val, index: f(val, -1, index), f))(f)
          for f in (
              torch.gather, torch.index_select,
              update_wrapper(lambda val, dim, index: torch.select(val, dim, index[0]), torch.select)
          )),
        torch.take, postpartial(torch.take_along_dim, -1)
    )

    binary_funcs = (
        torch.put, update_wrapper(lambda val, index, val2: torch.index_put(val, (index,), val2), torch.index_put),
        *(_indexed_rearg(f)
          for f in (postpartial(torch.index_add, alpha=0.42), torch.index_copy,
                    torch.scatter, torch.scatter_add)),
    )
    binary_scalar_funcs = (_indexed_rearg(torch.index_fill),)


class TestDefault(TestQfuncsBase):
    @mark.parametrize('func', (
        torch.abs, torch.absolute, torch.neg, torch.negative, torch.positive,
        partialparam(torch.amin, dim=-1), partialparam(torch.amax, dim=-1), partialparam(torch.aminmax, dim=-1),
        update_wrapper(lambda x: torch.as_strided(x, x.shape, x.stride()), torch.as_strided),
        torch.atleast_1d, torch.atleast_2d, torch.atleast_3d,
        Tensor.bfloat16, Tensor.cdouble, Tensor.cfloat, Tensor.double, Tensor.float, Tensor.half,
        postpartial(Tensor.type, torch.double), postpartial(Tensor.type_as, torch.empty((), dtype=torch.double)),
        postpartial(torch.broadcast_to, TestQfuncsBase.expanded_shape),
        *postpartials((torch.chunk, torch.unsafe_chunk), TestQfuncsBase.chunks),
        torch.clone, partialparam(torch.combinations, r=2),
        *map(with_complex_arg, (torch.conj, torch.conj_physical)),
        Tensor.contiguous, torch.detach,
        torch.diag_embed, torch.diagflat, torch.diff,
        postpartial(Tensor.expand, TestQfuncsBase.expanded_shape),
        postpartial(Tensor.expand_as, torch.empty(TestQfuncsBase.expanded_shape)),
        torch.flatten, torch.ravel, torch.gradient,
        with_complex_arg(torch.real), with_complex_arg(torch.imag),
        torch.sum, torch.nansum, postpartial(torch.cumsum, -1), update_wrapper(lambda x: x.sum_to_size(*x.shape[:-1], 1), Tensor.sum_to_size),
        torch.mean, torch.nanmean, postpartial(torch.std, -1, True), torch.std_mean, torch.median, torch.nanmedian,
        *postpartials((torch.quantile, torch.nanquantile), 0.2),
        torch.msort,
        *postpartials((torch.narrow, torch.narrow_copy), -1, 1, 2),
        kwargsparam(postpartial(Tensor.new_empty, (3, 4)), assert_values=False),
        *starmap(postpartial, map(partial(filter, None), zip(
            (Tensor.new_zeros, Tensor.new_ones, Tensor.new_full, Tensor.new_tensor),
            repeat((3, 4)), (None, None, 42, None)))),
        kwargsparam(torch.empty_like, assert_values=False),
        torch.zeros_like, torch.ones_like, postpartial(torch.full_like, 42),
        *map(deterministic, (torch.randn_like, torch.randn_like, postpartial(torch.randint_like, 5))),
        Tensor.pin_memory,
        deterministic(torch.poisson), torch.relu,
        *postpartials((Tensor.repeat, Tensor.repeat_interleave), 42),
        *postpartials((torch.reshape, Tensor.view), TestQfuncsBase.reshaped_shape),
        *postpartials((Tensor.reshape_as, Tensor.view_as), torch.rand(TestQfuncsBase.reshaped_shape)),
        torch.resolve_conj, torch.resolve_neg, postpartial(torch.roll, 2),
        torch.squeeze, postpartial(torch.unsqueeze, -1),
        *postpartials((torch.stft, update_wrapper(lambda x, *args, **kwargs: torch.istft(torch.stft(x, *args, **kwargs), *args, **kwargs), torch.istft)), n_fft=2, hop_length=2, return_complex=True),
        postpartial(torch.tile, (3,)), torch.unbind, postpartial(Tensor.unfold, 0, 2, 1),
        with_complex_arg(torch.view_as_real), update_wrapper(lambda x: torch.view_as_complex(with_complex_arg(torch.view_as_real)(x)), torch.view_as_complex)
    ))
    def test_default(self, func: Callable[[_VT], _nestedVT], q: ConcreteQuantity, **kwargs):
        if isinstance(func, tuple):
            func, kwargs = func
        self._tree_compare(func(q), pytree.tree_map(q.unit.__mul__, func(q.value)), **kwargs)

    @mark.parametrize('func', (
        torch.diag, torch.diagonal, torch.rot90,
        *postpartials((torch.moveaxis, torch.movedim, torch.swapaxes, torch.swapdims, torch.transpose), -1, 0),
        postpartial(torch.permute, (1, 0)), torch.t,
        torch.trace, torch.tril, torch.triu
    ))
    def test_default_matrix(self, func: Callable[[_VT], _nestedVT], q_matrix: ConcreteQuantity):
        self.test_default(func, q_matrix)

    @mark.parametrize('func', (*chain.from_iterable(map(postpartial(postpartial, arg), funcs) for funcs, arg in (
        ((torch.split, torch.tensor_split, torch.unsafe_split,
          torch.hsplit, torch.dsplit, torch.vsplit), TestQfuncsBase.chunks),
        # TODO: _with_sizes have no signatures and/or docs
        # ((torch.split_with_sizes, torch.unsafe_split_with_sizes), (2, 2, 2))
    )), postpartial(torch.flip, (-1,)), torch.fliplr, torch.flipud))
    def test_nd(self, func: Callable[[_VT], _nestedVT], q: ConcreteQuantity):
        self.test_default(func, q.expand(3*q.shape))

    @mark.parametrize('func', (
        torch.min, torch.max, torch.mode, torch.sort,
        *postpartials((torch.topk, torch.kthvalue), 2),
        *postpartials((torch.min, torch.max, torch.cummin, torch.cummax), dim=-1),
        *chain.from_iterable(
            (postpartial(f, return_inverse=r1, return_counts=r2)
             for r1, r2 in itertools.product(*2*((True, False),)))
            for f in (torch.unique,
                      # TODO: unique_consecutive (https://github.com/pytorch/pytorch/issues/68610)
                      # torch.unique_consecutive
                      )
        )
    ))
    def test_nonfloat_in_out(self, func: Callable[[_VT], _nestedVT], q: ConcreteQuantity):
        self._tree_compare(func(q), pytree.tree_map(
            (lambda arg: arg * q.unit if torch.is_floating_point(arg) or torch.is_complex(arg) else arg),
            func(q.value)))

    @mark.parametrize('func', (
        torch.atleast_1d, torch.atleast_2d, torch.atleast_3d,
        torch.broadcast_tensors, torch.meshgrid
    ))
    def test_mixed(self, func, q: ConcreteQuantity, unit2):
        args = q, q.value * unit2, q.value
        consume(
            self._compare(r1, r2 * a.unit if isinstance(a, GenericQuantity) else r2)
            for a, r1, r2 in zip(args, func(*args), func(*(
                a.value if isinstance(a, GenericQuantity) else a for a in args)))
        )

    def test_diff(self, val, unit, unit2):
        pre, app = torch.rand(2, 1)
        res = torch.diff(val, prepend=pre, append=app)
        self._compare(torch.diff(val * unit, prepend=pre/2 * (2*unit), append=app*2 * (unit/2)), res * unit)
        self._compare(torch.diff(val * Unit(), prepend=pre, append=app), res * Unit())
        with raises(UnitError):
            torch.diff(val * unit, prepend=pre * unit2)
            torch.diff(val * unit, append=app)

    def test_gradient(self, val_matrix, unit, unit2, unit3):
        q = val_matrix * unit

        res = torch.gradient(val_matrix)
        res1 = torch.gradient(val_matrix, dim=-1)

        def _test(ours, theirs=res1, out_unit: Union[Unit, Iterable[Unit]] = unit):
            self._tree_compare(ours, type(theirs)(r * u for r, u in zip(
                theirs, (repeat(out_unit) if isinstance(out_unit, Unit) else out_unit))))

        for sp in (1., torch.tensor(1.) * Unit()):
            _test(torch.gradient(q, dim=-1))
            _test(torch.gradient(q, dim=-1, spacing=sp))
            _test(torch.gradient(q, dim=(-1,), spacing=sp))
            _test(torch.gradient(q, dim=(-1,), spacing=(sp,)))
        # tensor inside a tuple doesn't work when dim is not a tuple as well:
        _test(torch.gradient(q, dim=-1, spacing=(1.,)))

        _test(torch.gradient(q), res)
        _test(torch.gradient(q, spacing=torch.tensor(0.5) * (2*unit2)), res, unit / unit2)
        _test(torch.gradient(q, spacing=(torch.tensor(1.) * unit2, torch.tensor(0.5) * (2*unit3))),
              res, (unit / unit2, unit / unit3))


class TestNoUnit(TestQfuncsBase):
    @mark.parametrize('func', (
        Tensor.bool, Tensor.byte, Tensor.char, Tensor.short, Tensor.int, Tensor.long, postpartial(Tensor.type, torch.long),
        torch.argmin, torch.argmax, torch.argsort,
        torch.all, torch.any, torch.count_nonzero, Tensor.data_ptr,
        kwargsparam(Tensor.storage, assert_values=False), Tensor.storage_offset, Tensor.storage_type,
        Tensor.size, pytest.param(Tensor.shape.__get__, id=Tensor.shape.__name__),
        Tensor.dim, Tensor.ndimension, Tensor.nelement, torch.numel, Tensor.element_size,
        Tensor.get_device, Tensor.has_names,
        torch.is_complex, torch.is_conj, Tensor.is_contiguous, torch.is_distributed, torch.is_floating_point,
        torch.is_inference, torch.is_neg, Tensor.is_pinned,
        postpartial(torch.is_same_size, torch.empty(())), postpartial(Tensor.is_set_to, torch.empty(())),
        Tensor.is_shared, torch.is_signed,
        torch.isfinite, torch.isinf, torch.isneginf, torch.isposinf, torch.isnan,
        torch.isreal,
        postpartial(deterministic(torch.multinomial), num_samples=2),
        torch.nonzero, postpartial(torch.nonzero, as_tuple=True),
        torch.sign, torch.sgn, torch.signbit,
        torch.corrcoef, torch.logical_not
    ))
    def test_nounit(self, func, q: ConcreteQuantity, **kwargs):
        if isinstance(func, tuple):
            func, kwargs = func
        self._tree_compare(func(q), func(q.value), **kwargs)

    @mark.parametrize('func', (torch.matrix_rank,))
    def test_nounit_matrix(self, func, q_matrix: ConcreteQuantity):
        self.test_nounit(func, q_matrix)

    def test_is_nonzero(self):
        assert torch.is_nonzero(torch.tensor(3.14) * second) and not torch.is_nonzero(torch.tensor(0.) * second)

    @mark.parametrize('func', (torch.logical_and, torch.logical_or, torch.logical_xor))
    def test_nounit_binary(self, func, q: ConcreteQuantity, q2: ConcreteQuantity):
        res = func(q.value, q2.value)
        self._compare(func(q, q2), res)
        self._compare(func(q, q2.value), res)
        self._compare(func(q.value, q2), res)


class TestTransEtAl(TestQfuncsBase):
    @mark.parametrize('func', (torch.sin, torch.cos, torch.tan, torch.sinc))
    def test_trig(self, func, val: _VT, unit: Unit):
        with raises(UnitError):
            _ = func(val * unit)

        q = val * radian

        res = func(q)
        assert not isinstance(res, GenericQuantity)
        assert (res == func(val)).all()
        assert func(val * deg).allclose(func(torch.deg2rad(val)))
        assert func(val * (Unit() / (scale := 3.14))).allclose(func(val / scale))

    @mark.parametrize('func', tuple(map(with_complex_arg, (
        torch.asin, torch.arcsin, torch.acos, torch.arccos, torch.atan, torch.arctan))))
    def test_inv_trig(self, func, val: _VT, unit: Unit):
        with raises(UnitError):
            _ = func(val * unit)
        self.test_to_unit(func, radian, val, Unit())

    @mark.parametrize('func', (
        torch.exp, torch.exp2, torch.expm1,
        torch.log, torch.log2, torch.log10, torch.log1p,
        partialparam(torch.logsumexp, dim=-1), partialparam(torch.logcumsumexp, dim=-1),
        partialparam(torch.polygamma, 2), partialparam(torch.mvlgamma, p=1),
        torch.lgamma, torch.digamma,
        torch.erf, torch.erfc, torch.erfinv, torch.logit, torch.sigmoid, torch.i0,
        torch.sinh, torch.asinh, torch.arcsinh, torch.cosh, torch.tanh, torch.atanh, torch.arctanh,
        deterministic(torch.bernoulli),
        *(postpartial(f, -1) for f in (torch.softmax, torch.log_softmax)),
        torch.floor, torch.ceil, torch.round, torch.fix, torch.trunc, torch.frac
    ))
    def test_trans(self, func, val: _VT, unit: Unit):
        with raises(UnitError):
            _ = func(val * unit)

        self._compare(res := func(val * Unit()), func(val))
        self._compare(res, func(val/2 * (Unit() * 2)))

    @mark.parametrize('func', (torch.matrix_exp, torch.logdet, torch.slogdet))
    def test_trans_matrix(self, func, val_matrix: _VT, unit: Unit):
        self.test_trans(func, val_matrix, unit)

    @mark.parametrize('func', (torch.acosh, torch.arccosh))
    def test_acosh(self, func, val: _VT, unit: Unit):
        return self.test_trans(func, val + 1, unit)

    @mark.parametrize('func', (
        torch.logaddexp, torch.logaddexp2,
        torch.igamma, torch.igammac
    ))
    def test_trans_twoargs(self, func, val: _VT, val2: _VT, unit: Unit, unit2):
        for vs in ((val, val2*unit), (val*unit, val2), (val*unit, val2*unit), (val*unit, val * unit2)):
            with raises(UnitError):
                func(*vs)

        truth = func(val, val2)
        for res in starmap(func, ((val * Unit(), val2),
                                  (val, val2 * Unit()),
                                  (val * Unit(), val2 * Unit()),
                                  (val, val2/2 * (2*Unit())))):
            assert not isinstance(res, GenericQuantity)
            assert res.allclose(truth)

    def test_atan2(self, val: _VT, val2: _VT, unit: Unit, unit2):
        func = torch.atan2

        for vs in ((val*unit, val2 * unit2), (val, val2 * unit2), (val * unit, val2)):
            with raises(UnitError):
                func(*vs)

        truth = func(val, val2) * radian
        for res in starmap(func, ((val*unit, val2*unit),
                                  (val*Unit(), val2),
                                  (val, val2*Unit()),
                                  (val*unit, val2/2 * (2*unit)))):
            assert isinstance(res, GenericQuantity) and res.unit is radian
            assert res.allclose(truth)


class TestSame(TestQfuncsBase):
    @staticmethod
    def _test_raise_not_same(func, val, val2, unit, unit2, only_dispatches_on_first=False):
        with raises(UnitError):
            func(val * unit, val2 * unit2)
        with raises(UnitError):
            func(val * unit, val2)

        if not only_dispatches_on_first:
            with raises(UnitError):
                func(val, val2 * unit2)

    _to_nounit = lambda func: pytest.param((func, False), id=func.__name__)

    @mark.parametrize('func', (
        Tensor.__add__, Tensor.__radd__, torch.add,
        Tensor.__sub__, Tensor.__rsub__, torch.sub, torch.subtract, torch.rsub,
        Tensor.__mod__, Tensor.__rmod__, torch.fmod, torch.remainder,
        torch.complex,
        torch.dist, update_wrapper(lambda *args: torch.cdist(*torch.atleast_2d(*args)), torch.cdist),
        torch.fmin, torch.fmax, torch.minimum, torch.maximum,
        torch.hypot, torch.nextafter,
        *map(_to_nounit, (
            torch.allclose, torch.isclose,
            torch.bucketize, torch.isin, torch.searchsorted,
            Tensor.__eq__, torch.eq, torch.equal, Tensor.__ne__, torch.ne,
            torch.gt, torch.greater, torch.ge, torch.greater_equal,
            torch.lt, torch.less, torch.le, torch.less_equal,
            Tensor.__floordiv__, Tensor.__rfloordiv__, Tensor.floor_divide
        ))
    ))
    def test_same_binary(self, func, val, val2, unit, unit2, **kwargs):
        if isinstance(func, tuple):
            func, out_unit = func
        else:
            out_unit = unit

        self._test_raise_not_same(func, val, val2, unit, unit2, **kwargs)

        res = func(val, val2)
        self._compare(func(val * unit, val2/2 * (2*unit)),
                      res * out_unit if out_unit is not False else res)
        self._compare(func(val * Unit(), val2),
                      res * Unit() if out_unit is not False else res)

    @mark.parametrize('func', (torch.hardshrink, torch.nan_to_num, torch.clamp_min, torch.clamp_max))
    def test_same_scalar(self, func, val, unit, unit2):
        self.test_same_binary(func, val, torch.tensor(0.5), unit, unit2,
                              only_dispatches_on_first=True)
        with raises(UnitError):
            func(val * unit, 0.5)

    @mark.parametrize('func', (torch.clamp, torch.clip))
    def test_clamp(self, func, val, val2, unit, unit2):
        low, high = val2[:2].sort().values
        res = func(val, low, high)

        self._compare(func(val * unit, low * unit, high * unit), res * unit)
        self._compare(func(val * unit, low/2 * (2*unit), high*2 * (unit/2)), res * unit)
        self._compare(func(val * Unit(), low, high), res * Unit())
        self._compare(func(val, low * Unit(), high), res)

        self._compare(func(val * unit, low * unit), func(val, low) * unit)
        self._compare(func(val * unit, None, low * unit), func(val, None, low) * unit)

        with raises(UnitError):
            func(val * unit, low * unit2, high * unit)
        with raises(UnitError):
            func(val * unit, low * unit, high)


    @mark.parametrize('func', (
        update_wrapper(lambda args: torch.block_diag(*(a.flatten(end_dim=-2) for a in args)), torch.block_diag),
        torch.cat, torch.concat,
        torch.stack, torch.hstack, torch.dstack, torch.vstack, torch.row_stack, torch.column_stack,
        update_wrapper(lambda args: torch.cartesian_prod(*(a.flatten()[:10] for a in args)), torch.cartesian_prod),
    ))
    def test_same_multi(self, func, unit, unit2):
        n = 5
        units = (unit, unit2, *choices((unit, unit2), k=n-2))
        vals = *torch.rand(n, *3*self.shape[-1:]),

        with raises(UnitError):
            func(tuple(v*u for v, u in zip(vals, units)))

        self._compare(func(tuple(v*unit for v in vals)), func(vals) * unit)


class TestProduct(TestQfuncsBase):
    def _test_product(self, func, op, vals: Sequence[_VT], units: Sequence[Unit]):
        res = func(*vals)
        for args in itertools.product(*((v, v*u) for v, u in zip(vals, units))):
            self._compare(func(*args), res * reduce(op, (
                _.unit if isinstance(_, (Unit, GenericQuantity)) else 1 for _ in args)))

    @mark.parametrize('func, op', tuple(map(_tfparam, (
        *zip((
            Tensor.__mul__, Tensor.__rmul__, torch.mul, torch.multiply,
            torch.dot, torch.vdot, torch.inner, torch.outer, torch.ger, torch.kron,
            update_wrapper(lambda x1, x2: torch.cross(x1[:3], x2[:3]), torch.cross),
            postpartial(torch.tensordot, dims=1)
        ), repeat(operator.mul)),
        *zip((torch.div, torch.divide, torch.true_divide), repeat(operator.truediv)),
    ))))
    def test_binary_product(self, func, op, val: _VT, val2: _VT, unit: Unit, unit2: Unit):
        self._test_product(func, op, (val, val2), (unit, unit2))

    @mark.parametrize('func, op', (*zip((
        Tensor.__matmul__, Tensor.__rmatmul__, torch.matmul, torch.mm,
        update_wrapper(lambda *args: torch.bmm(*(a.reshape(-1, *a.shape[-2:]) for a in args)), torch.bmm),
        torch.chain_matmul, update_wrapper(partial(torch.einsum, 'ik, kj -> ij'), torch.einsum),
        torch.trapz, torch.trapezoid, torch.cumulative_trapezoid
    ), repeat(operator.mul)),))
    def test_binary_product_matrix(self, func, op, val_matrix, val2_matrix, unit, unit2):
        self.test_binary_product(func, op, val_matrix, val2_matrix, unit, unit2)

    @mark.parametrize('func', (
        torch.chain_matmul,
        update_wrapper(lambda *args: torch.einsum(
            ','.join(f'i{A}' for A in take(len(args), string.ascii_uppercase))
            + '->' + ''.join(take(len(args), string.ascii_uppercase)),
            *args
        ), torch.einsum)
    ))
    @given(st.integers(min_value=1, max_value=4).flatmap(lambda n: st.lists(units_strategy, min_size=n, max_size=n)))
    def test_multiproduct_matrix(self, func, units: list[Unit]):
        self._test_product(func, operator.mul, [torch.rand(*2*self.shape[-1:]) for _ in units], units)

    @mark.parametrize('func', (torch.mv,))
    def test_mv(self, func, val_matrix, val, unit, unit2):
        self.test_binary_product(func, operator.mul, val_matrix, val, unit, unit2)

    @mark.parametrize('func', (torch.trapz, torch.trapezoid, torch.cumulative_trapezoid))
    def test_trapz_dx(self, func: Callable, q: ConcreteQuantity, unit2: Unit):
        res = func(q.value) * q.unit
        self._compare(func(q), res)
        self._compare(func(q, dx=torch.tensor(1.) * unit2), res * unit2)


class TestPower(TestQfuncsBase):
    @mark.parametrize('func', (Tensor.__pow__, torch.pow, torch.float_power))
    @given(random_tensors, dimful_units_strategy)
    def test_power(self, func, val: _VT, unit: Unit):
        truth = func(val, 2) * unit**2
        q = val * unit
        self._compare(func(q, 2), truth)

        self._compare(func(q, torch.tensor(2.)), truth)
        self._compare(func(q, torch.tensor(1.) * (2*Unit())), truth)

        with raises(UnitError):
            func(q, torch.ones_like(q))
        with raises(UnitError):
            func(q, torch.tensor(1.) * unit)

        # TODO: make functions accept Units
        # self._tree_compare(func(q, 2 * Unit()), truth)
        # with raises(UnitError):
        #     func(q, unit)

    @mark.parametrize('func', (torch.matrix_power,))
    def test_power_matrix(self, func, val_matrix, unit):
        self._compare(func(val_matrix * unit, 2), func(val_matrix, 2) * unit**2)

    @given(random_tensors, dimful_units_strategy)
    def test_rpow(self, val: _VT, unit: Unit):
        with raises(UnitError):
            2 ** (val * unit)
        with raises(UnitError):
            torch.tensor(2.) ** (val * unit)

        self._compare(2**(val/2 * (2*Unit())), 2**val)
