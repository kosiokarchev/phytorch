from functools import update_wrapper
from itertools import combinations, permutations
from math import log, pi

import torch
from hypothesis import assume, given, strategies as st

from phytorch.special.ellipr import elliprc, elliprd, elliprf, elliprg, elliprj


# From Carlson (1995) except where stated otherwise
cases = {
    elliprc: (
        ((0, 1/4), pi), ((9/4, 2), log(2)),
        ((0, 1j), (1-1j) * 1.1107207345396),
        ((-1j, 1j), 1.2260849569072 - 0.34471136988768j),
        ((0.25, -2), log(2) / 3),
        ((1j, -1), 0.77778596920447 + 0.19832484993429j),
    ),
    elliprd: (
        ((0, 2, 1), 1.7972103521034),
        ((2, 3, 4), 0.16510527294261),
        ((1j, -1j, 2), 0.65933854154220),
        ((0, 1j, -1j), 1.2708196271910 + 2.7811120159521j),
        ((0, 1j-1, 1j), -1.8577235439239 - 0.96193450888839j),
        ((-2-1j, -1j, -1+1j), 1.8249027393704 - 1.2218475784827j),
    ),
    elliprf: (
        ((1, 2, 0), 1.3110287771461),
        ((0.5, 1, 0), 1.8540746773014), ((1j, -1j, 0), 1.8540746773014),
        ((1j-1, 1j, 0), 0.79612586584234 - 1.2138566698365j),
        ((2, 3, 4), 0.58408284167715),
        ((1j, -1j, 2), 1.0441445654064),
        ((1j-1, 1j, 1-1j), 0.93912050218619 - 0.53296252018635j),
    ),
    elliprg: (
        ((0, 16, 16), pi),
        ((2, 3, 4), 1.7255030280692),
        ((0, 1j, -1j), 0.42360654239699),
        ((1j-1, 1j, 0), 0.44660591677018 + 0.70768352357515j),
        ((-1j, 1j-1, 1j), 0.36023392184473 + 0.40348623401722j),
        ((0, 0.0796, 4), 1.0284758090288),
        # mpmath:
        ((0, 0, 0), 0), ((0, 0, 16), 2),
        ((1, 4, 0), 1.2110560275684595248036),
        ((1, 1j, -1 + 1j), 0.64139146875812627545 + 0.58085463774808290907j),
    ),
    elliprj: (
        ((0, 1, 2, 3), 0.77688623778582),
        ((2, 3, 4, 5), 0.14297579667157),
        ((2, 3, 4, -1+1j), 0.13613945827771 - 0.38207561624427j),
        ((1j, -1j, 0, 2), 1.6490011662711),
        ((-1+1j, -1-1j, 1, 2), 0.94148358841220),
        ((1j, -1j, 0, 1-1j), 1.8260115229009 + 1.2290661908643j),
        ((-1+1j, -1-1j, 1, -3+1j), -0.61127970812028 - 1.0684038390007j),
        ((-1+1j, -2-1j, -1j, -1+1j), 1.8249027393704 - 1.2218475784827j),
        # computed using mpmath; see comment in C95
        ((2, 3, 4, -0.5), 0.24723819703052 - 0.7509842836890j),
        ((2, 3, 4, -5), -0.12711230042964 - 0.2099064885453j),
    )
}


def with_default_double(func):
    def f(*args, **kwargs):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.double)
        ret = func(*args, **kwargs)
        torch.set_default_dtype(default_dtype)
        return ret
    return update_wrapper(f, func)


CLOSE_KWARGS = dict(atol=1e-6, rtol=1e-6)


@with_default_double
def test_ellipr():
    for func, vals in cases.items():
        ins, outs = map(torch.tensor, zip(*vals))
        res = func(*ins.T)
        isclose = torch.isclose(res, outs, **CLOSE_KWARGS)
        if not isclose.all().item():
            print(func, isclose)
            print(ins[~isclose], res[~isclose], outs[~isclose])
        assert isclose.all()


# from mpmath; Carlson's algorithm is not guaranteed, so mpmath integrates numerically
# for now we can't handle these cases although for all but the first two cases
# the algorith is correct
cases_elliprj_fail = (
    ((-1-0.5j, -10-6j, -10-3j, -5+10j), 0.128470516743927699 + 0.102175950778504625j),  # fails
    ((1.987, 4.463-1.614j, 0, -3.965), -0.341575118513811305 - 0.394703757004268486j),  # fails
    ((0.3068, -4.037+0.632j, 1.654, -0.9609), -1.14735199581485639 - 0.134450158867472264j),
    ((0.3068, -4.037-0.632j, 1.654, -0.9609), 1.758765901861727 - 0.161002343366626892j),
    ((0.3068, -4.037+0.0632j, 1.654, -0.9609), -1.17157627949475577 - 0.069182614173988811j),
    ((0.3068, -4.037+0.00632j, 1.654, -0.9609), -1.17337595670549633 - 0.0623069224526925j),
    ((0.3068, -4.037-0.0632j, 1.654, -0.9609), 1.77940452391261626 + 0.0388711305592447234j),
    ((0.3068, -4.037-0.00632j, 1.654, -0.9609), 1.77806722756403055 + 0.0592749824572262329j)
)


@with_default_double
def test_elliprj_fail():
    assert elliprj(*torch.tensor(next(zip(*cases_elliprj_fail))).T).real.isnan().all()


def allclose_complex_nan(a, b, **kwargs):
    return all(torch.allclose(acc(a), acc(b), **{**CLOSE_KWARGS, 'equal_nan': True, **kwargs})
                for acc in (torch.abs, torch.angle))


def n_tensors_strategy(n, elements: st.SearchStrategy = st.floats(min_value=1e-4, max_value=1e3), max_len=10):
    return st.integers(min_value=1, max_value=max_len).flatmap(lambda m: st.tuples(*(
        st.lists(elements, min_size=m, max_size=m) for i in range(n)
    ))).map(torch.tensor)


def n_complex_tensors_strategy(n, max_len=10, min_magnitude=1e-4, max_magnitude=1e3):
    return n_tensors_strategy(n, st.complex_numbers(min_magnitude=min_magnitude, max_magnitude=max_magnitude), max_len=max_len)


def make_symmetry_tester(func, nsym: int, nadd: int = 0):
    @with_default_double
    @given(args=n_complex_tensors_strategy(nsym + nadd))
    def test_symmetry(args: torch.Tensor):
        assert all(
            allclose_complex_nan(f1, f2)
            for f1, f2 in combinations((func(*arg, *args[nsym:]) for arg in permutations(args[:nsym])), 2)
        )
    return test_symmetry


test_symmetry_elliprf = make_symmetry_tester(elliprf, 3)
test_symmetry_elliprg = make_symmetry_tester(elliprg, 3)
test_symmetry_elliprd = make_symmetry_tester(elliprd, 2, 1)
test_symmetry_elliprj = make_symmetry_tester(elliprj, 3, 1)


@with_default_double
@given(xyz=n_complex_tensors_strategy(3))
def test_definitions(xyz):
    x, y, z = xyz
    assert allclose_complex_nan(elliprc(x, y), elliprf(x, y, y))
    assert allclose_complex_nan(elliprd(x, y, z), elliprj(x, y, z, z))


@with_default_double
@given(xy_lrli=n_tensors_strategy(4))
def test_consistency_3_1(xy_lrli: torch.Tensor):
    x, y, lr, li = xy_lrli
    lbda = torch.complex(lr, li)
    assume(not (lbda == 0).any())
    mu = x * y / lbda
    assert allclose_complex_nan(
        elliprf(x+lbda, y+lbda, lbda) + elliprf(x+mu, y+mu, mu),
        elliprf(x, y, 0)
    )


@with_default_double
@given(x_lrli=n_tensors_strategy(3))
def test_consistency_3_2(x_lrli: torch.Tensor):
    x, lr, li = x_lrli
    lbda = torch.complex(lr, li)
    assume(not (lbda == 0).any())
    mu = x * x / lbda
    assert allclose_complex_nan(
        elliprc(lbda, x+lbda) + elliprc(mu, x+mu),
        elliprc(0, x)
    )


@with_default_double
@given(xyp_lrli=n_tensors_strategy(5))
def test_consistency_3_3(xyp_lrli: torch.Tensor):
    x, y, p, lr, li = xyp_lrli
    lbda = torch.complex(lr, li)
    assume(not (lbda == 0).any())
    mu = x * y / lbda

    assert allclose_complex_nan(
        elliprj(x+lbda, y+lbda, lbda, p+lbda) + elliprj(x+mu, y+mu, mu, p+mu),
        elliprj(x, y, 0, p) - 3*elliprc(p**2 * (lbda + mu + x + y), p * (p+lbda) * (p+mu)),
        rtol=1e-4, atol=1e-4)


@with_default_double
@given(xy_lrli=n_tensors_strategy(4))
def test_consistency_3_5(xy_lrli: torch.Tensor):
    x, y, lr, li = xy_lrli
    lbda = torch.complex(lr, li)
    assume(not (lbda == 0).any())
    mu = x * y / lbda
    assert allclose_complex_nan(
        elliprd(lbda, x+lbda, y+lbda) + elliprd(mu, x+mu, y+mu),
        elliprd(0, x, y) - 3 / (y * (x+y + lbda + mu)**0.5)
    )


@with_default_double
@given(args=n_complex_tensors_strategy(3))
def test_consistency_3_6(args: torch.Tensor):
    # C95 eq. (3.6)
    assert allclose_complex_nan(
        sum(elliprd(*arg) for arg in permutations(args)),
        6 / args.sqrt().prod(0)
    )


# TODO: connection formulae from https://dlmf.nist.gov/19.21
