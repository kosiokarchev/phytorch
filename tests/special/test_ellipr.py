from cmath import phase, log, pi
from itertools import combinations, cycle, permutations
from math import gamma

import torch
from hypothesis import assume, given, strategies as st
from more_itertools import circular_shifts
from torch import isinf

from phytorch.special.ellipr import elliprc, elliprd, elliprf, elliprg, elliprj
from tests.common import (BaseDoubleTest, BaseFloatTest, close_complex_nan, JUST_FINITE, n_complex_tensors_strategy,
                          n_tensors_strategy,
                          nice_and_close, process_cases,
                          with_default_double,
                          CLOSE_KWARGS)


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
        ((0, 2, 1), 3 * gamma(3/4)**2 / (2*pi)**0.5),  # https://dlmf.nist.gov/19.20.E22
        ((2, 3, 4), 0.16510527294261),
        ((1j, -1j, 2), 0.65933854154220),
        ((0, 1j, -1j), 1.2708196271910 + 2.7811120159521j),
        ((0, 1j-1, 1j), -1.8577235439239 - 0.96193450888839j),
        ((-2-1j, -1j, -1+1j), 1.8249027393704 - 1.2218475784827j),
    ),
    elliprf: (
        ((1, 2, 0), gamma(1/4)**2 / 4 / (2*pi)**0.5),  # https://dlmf.nist.gov/19.20.E2
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


@with_default_double
def test_ellipr():
    for func, vals in cases.items():
        ins, outs, res = process_cases(func, vals)
        assert torch.allclose(outs, res, **CLOSE_KWARGS)


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


def make_symmetry_tester(func, nsym: int, nadd: int = 0):
    @with_default_double
    @given(args=n_complex_tensors_strategy(nsym + nadd))
    def test_symmetry(args: torch.Tensor):
        assert all(
            close_complex_nan(f1, f2)
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
    assert close_complex_nan(elliprc(x, y), elliprf(x, y, y))
    assert close_complex_nan(elliprd(x, y, z), elliprj(x, y, z, z))


@with_default_double
@given(xy_lrli=n_tensors_strategy(4))
def test_consistency_3_1(xy_lrli: torch.Tensor):
    x, y, lr, li = xy_lrli
    lbda = torch.complex(lr, li)
    assume(not (lbda == 0).any())
    mu = x * y / lbda
    assert close_complex_nan(
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
    assert close_complex_nan(
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

    assert close_complex_nan(
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
    assert close_complex_nan(
        elliprd(lbda, x+lbda, y+lbda) + elliprd(mu, x+mu, y+mu),
        elliprd(0, x, y) - 3 / (y * (x+y + lbda + mu)**0.5)
    )


@with_default_double
@given(args=n_complex_tensors_strategy(3))
def test_consistency_3_6(args: torch.Tensor):
    # C95 eq. (3.6)
    assert close_complex_nan(
        sum(elliprd(*arg) for arg in permutations(args)),
        6 / args.sqrt().prod(0)
    )


# TODO: sepcial cases formulae from https://dlmf.nist.gov/19.20
class TestElliprSpecialCases(BaseDoubleTest):
    _complex_numbers = (st.complex_numbers(min_magnitude=1e-3, max_magnitude=1e3, **JUST_FINITE),)
    _float = st.floats(1e-3, 1e3)

    @staticmethod
    @given(*2*_complex_numbers, _float)
    def test_elliprc(x, y, l):
        # TODO: global C \ (-inf, 0]
        assume(not any((_.real <= 0 and _.imag == 0) for _ in (x, y)))

        assert nice_and_close(elliprc(x, x), x**(-0.5))
        assert nice_and_close(elliprc(0, x), pi/2 / x**0.5)
        assert nice_and_close(elliprc(l*x, l*y), elliprc(x, y) / l**0.5)

    @staticmethod
    @given(*3*_complex_numbers, _float)
    def test_elliprf(x, y, z, l):
        # TODO: global C \ (-inf, 0]
        assume(not any((_.real <= 0 and _.imag == 0) for _ in (x, y, z)))

        assert nice_and_close(elliprf(x, x, x), x**(-0.5))
        assert nice_and_close(elliprf(0, y, y), pi/2 / y**0.5)
        assert nice_and_close(elliprf(l*x, l*y, l*z), elliprf(x, y, z) / l**0.5)
        assert isinf(elliprf(0, 0, z))

    @staticmethod
    @given(*3*_complex_numbers, _float)
    def test_elliprd(x, y, z, l):
        # TODO: global C \ (-inf, 0]
        assume(not any((_.real <= 0 and _.imag == 0) for _ in (x, y, z)))

        assert nice_and_close(elliprd(x, x, x), x**(-1.5))
        assert nice_and_close(elliprd(l*x, l*y, l*z), elliprd(x, y, z) / l**1.5)
        assert nice_and_close(elliprd(0, y, y), 3*pi/4 / y**1.5)
        assert isinf(elliprd(0, 0, z))

    @staticmethod
    @given(*3*_complex_numbers)
    def test_elliprd1(x, y, z):
        # TODO: global C \ (-inf, 0]
        assume(not any((_.real <= 0 and _.imag == 0) for _ in (x, y, z)))

        assume(x != y and y != 0 and x != z and x*z != 0)
        assert nice_and_close(elliprd(x, y, y), 1.5 * (elliprc(x, y) - x**0.5/y) / (y-x))
        assert nice_and_close(elliprd(x, x, z), 3 * (elliprc(z, x) - z**(-0.5)) / (z-x))

    @staticmethod
    @given(*3*_complex_numbers, _float)
    def test_elliprg(x, y, z, l):
        # TODO: global C \ (-inf, 0]
        assume(not any((_.real <= 0 and _.imag == 0) for _ in (x, y, z)))

        assert nice_and_close(elliprg(x, x, x), x**0.5)
        assert nice_and_close(elliprg(l * x, l * y, l * z), elliprg(x, y, z) * l**0.5)
        assert nice_and_close(elliprg(0, y, y), pi / 4 * y**0.5)
        assert nice_and_close(elliprg(0, 0, z), z**0.5 / 2)
        assert nice_and_close(2 * elliprg(x, y, y), y * elliprc(x, y) + x**0.5)

    @staticmethod
    @given(*4*_complex_numbers, _float)
    def test_elliprj(x, y, z, p, l):
        # TODO: global C \ (-inf, 0]
        assume(not any((_.real <= 0 and _.imag == 0) for _ in (x, y, z)))

        assert nice_and_close(elliprj(x, x, x, x), x**(-1.5))
        assert nice_and_close(elliprj(l*x, l*y, l*z, l*p), elliprj(x, y, z, p) / l**1.5)
        assert nice_and_close(elliprj(x, y, z, z), elliprd(x, y, z))
        assert isinf(elliprj(0, 0, z, p))
        assert nice_and_close(elliprj(x, x, x, p), elliprd(p, p, x))
        assert nice_and_close(elliprj(x, x, x, p).real, 3 * (elliprc(x, p) - x**(-0.5)) / (x-p))
        assert nice_and_close(elliprj(x, y, y, y), elliprd(x, y, y))
        # assert nice_and_close(elliprj(0, y, z, (y*z)**0.5), 1.5/(y*z)**0.5 * elliprf(0, y, z))
        # assert nice_and_close(elliprj(0, y, z, -(y*z)**0.5), -1.5/(y*z)**0.5 * elliprf(0, y, z))

        p = x + ((y-x)*(z-x))**0.5
        assert nice_and_close((p-x)*elliprj(x, y, z, p), 1.5 * (elliprf(x, y, z) - x**0.5*elliprc(y*z, p**2)))

    @staticmethod
    @given(*2*_complex_numbers, _float)
    def test_elliprj1(x, y, p):
        # TODO: global C \ (-inf, 0]
        assume(not any((_.real <= 0 and _.imag == 0) for _ in (x, y)))
        assume(abs(p - y) > 1e-3)

        assert nice_and_close(elliprj(0, y, y, p), 1.5*pi / (y*p**0.5 + p*y**0.5))
        assert nice_and_close(elliprj(x, y, y, p), 3 * (elliprc(x, y) - elliprc(x, p)) / (p-y))

    @staticmethod
    @given(_float, _float)
    def test_elliprj2(y, p):
        assume(y != 0 and y != p)
        assert nice_and_close(elliprj(0, y, y, -p).real, -1.5*pi / y**0.5 / (y+p))


# TODO: connections using float
class TestElliprConnections(BaseDoubleTest):
    _complex_number = st.complex_numbers(max_magnitude=1e10, **JUST_FINITE)
    _complex_numbers = (_complex_number,)

    @staticmethod
    @given(_complex_number)
    def test_legendre_relation(z):
        assume(not (z.real <= 0 and z.imag == 0))

        # https://dlmf.nist.gov/19.21.E1
        assert nice_and_close(
            elliprf(0, z+1, z) * elliprd(0, z+1, 1) + elliprd(0, z+1, z)*elliprf(0, z+1, 1),
            3*pi/2 / z
        )

    @staticmethod
    @given(*2*_complex_numbers)
    def test_complete_1(y, z):
        # TODO: global C \ (-inf, 0]
        assume(not any((_.real <= 0 and _.imag == 0) for _ in (y, z)))

        # https://dlmf.nist.gov/19.21.E2
        assert nice_and_close(3 * elliprf(0, y, z), z*elliprd(0, y, z) + y*elliprd(0, z, y))

        # https://dlmf.nist.gov/19.21.E3
        assert nice_and_close(6*elliprg(0, y, z), y*z*(elliprd(0, y, z) + elliprd(0, z, y)))
        assert nice_and_close(6*elliprg(0, y, z), 3*z*elliprf(0, y, z) + z*(y-z)*elliprd(0, y, z))

    @staticmethod
    @given(_complex_number)
    def test_complete_2(z: complex):
        assume(not z.imag == 0)
        sign = (1 if phase(z) > 0 else -1) * 1j

        # https://dlmf.nist.gov/19.21.E4
        assert nice_and_close(elliprf(0, z-1, z), elliprf(0, 1-z, 1) - sign *elliprf(0, z, 1))

        # https://dlmf.nist.gov/19.21.E5
        assert nice_and_close(
            2*elliprg(0, z-1, z),
            2 * (elliprg(0, 1-z, 1) + sign * elliprg(0, z, 1))
            + (z-1)*elliprf(0, 1-z, 1) - sign * z*elliprf(0, z, 1)
        )

    @staticmethod
    @given(*3*(st.floats(1e-9, exclude_min=True, **JUST_FINITE),))
    def test_complete_3(y, z, p):
        # https://dlmf.nist.gov/19.21.E6
        assume(len({y, z, p}) == 3)
        if z < y < p or p < y < z:
            y, z = z, y

        r = (y - p) / (y - z)
        assume(r > 0)

        assert nice_and_close(
            (r*p)**0.5 / z * elliprj(0, y, z, p),
            (r-1) * elliprf(0, y, z) * elliprd(p, r*z, z) + elliprd(0, y, z) * elliprf(p, r*z, z)
        )

    @staticmethod
    @given(*3*_complex_numbers)
    def test_incomplete(x, y, z):
        # TODO: sign of sqrt(x, y, z)
        assume(x != 0 and y != 0 and z != 0 and len({x, y, z}) == 3)

        # https://dlmf.nist.gov/19.21.E7
        assert nice_and_close(
            abs(3 * elliprf(x, y, z) - ((x-y)*elliprd(y, z, x) + (z-y)*elliprd(x, y, z))),
            abs(3 * (y / x / z)**0.5)
        )

        # https://dlmf.nist.gov/19.21.E8
        assert nice_and_close(
            abs(elliprd(y, z, x) + elliprd(z, x, y) + elliprd(x, y, z)),
            abs(3*(x*y*z)**(-0.5)))

        # https://dlmf.nist.gov/19.21.E9
        assert nice_and_close(x*elliprd(y, z, x) + y*elliprd(z, x, y) + z*elliprd(x, y, z), 3*elliprf(x, y, z))

        # https://dlmf.nist.gov/19.21.E10
        assert nice_and_close(
            abs(2*elliprg(x, y, z) - (z*elliprf(x, y, z) - (x-z)*(y-z)*elliprd(x, y, z) / 3)),
            abs((x*y/z)**0.5)
        )

        # https://dlmf.nist.gov/19.21.E11
        assert nice_and_close(
            6*elliprg(x, y, z),
            3*(x+y+z)*elliprf(x, y, z) - sum(_x**2 * elliprd(_y, _z, _x) for _x, _y, _z in circular_shifts((x, y, z)))
        )
        assert nice_and_close(
            6*elliprg(x, y, z),
            sum(_x * (_y + _z) * elliprd(_y, _z, _x) for _x, _y, _z in circular_shifts((x, y, z)))
        )

    @staticmethod
    @given(*3*(st.floats(1e-9, 1e3, **JUST_FINITE),), st.complex_numbers(min_magnitude=1e-9, max_magnitude=1e10, **JUST_FINITE))
    def test_elliprj(x, y, z, p):
        # TODO: properly analytically continue
        assume(all(abs(a-b)>1e-3 for a, b in combinations((x, y, z, p), 2)))

        # https://dlmf.nist.gov/19.21.E13
        q = (y-x)*(z-x) / (p-x) + x

        # # https://dlmf.nist.gov/19.21.E12
        assert nice_and_close(
            ((p-x)*elliprj(x, y, z, p) + (q-x)*elliprj(x, y, z, q)).real,
            (3 * (elliprf(x, y, z) - elliprc(y*z/x, p*q/x))).real
        )

        # https://dlmf.nist.gov/19.21.E15
        # special case of above with x=0
        q = y*z/p
        assert nice_and_close(
            (p*elliprj(0, y, z, p) + q*elliprj(0, y, z, q)).real,
            3 * elliprf(0, y, z).real
        )
