from cmath import asin, asinh, atan, cos, inf, nan, pi, sin, sinh, tan

import numpy as np
import scipy.special as sp
import torch
from hypothesis import assume, given, strategies as st
from mpmath import mp
from pytest import mark
from torch import isclose, isnan, tensor

from phytorch.special.ellip import ellipd, ellipdinc, ellipe, ellipeinc, ellipk, ellipkinc, ellippi, ellippiinc
from phytorch.special.ellipr import elliprc
from tests.common import (BaseDoubleTest, BaseDtypeTest, BaseFloatTest, close_complex_nan, JUST_FINITE,
                          make_dtype_tests,
                          nice_and_close, with_default_double)


# @with_default_double
# def test_grid():
#     # TODO: separate functions
#     m, phi = torch.linspace(0, 1, 101), torch.linspace(0, pi/2, 101).unsqueeze(-1)
#
#     for ours, theirs, args in (
#         (ellipk, sp.ellipk, (m,)),
#         (ellipe, sp.ellipe, (m,)),
#         (ellipkinc, sp.ellipkinc, (phi, m)),
#         (ellipeinc, sp.ellipeinc, (phi, m)),
#         (ellippi, (sp_ellippi := np.vectorize(lambda *a, **kw: float(mp.ellippi(*a, *kw)))), (m[:-1].unsqueeze(-2), m[:-1])),  # avoid edges
#         (ellippiinc, sp_ellippi, (m[:-1].unsqueeze(-2).unsqueeze(-3), phi, m[:-1])),  # avoid edges
#     ):
#         assert torch.allclose(ours(*args).real, torch.as_tensor(theirs(*args)), **CLOSE_KWARGS, equal_nan=True)


# From the scipy.special test suite, unless otherwise stated:

cases = {
    ellipk: (
        ((0.2,), 1.659623598610528),
        ((-10,), 0.7908718902387385)
    ),
    ellipe: (
        ((0.2,), 1.4890350580958529),
        ((-10,), 3.6391380384177689),
        ((0.0,), pi / 2),
        ((1.0,), 1.0),
        ((2,), complex(mp.ellipe(2))),  # scipy does not handle complex
        ((nan,), nan),
        # TODO: ellipe infinities
        # ((-inf,), inf),
        # ((inf,), inf*1j)
    ),
    ellipkinc: (
        ((45*pi/180, sin(20*pi/180)**2), 0.79398143),
        ((0.38974112035318718, 1), 0.4),
        ((1.5707, -10), 0.79084284661724946),
        ((pi / 2, 0.0), pi / 2),
        ((pi / 2, 1.0), inf),
        ((pi / 2, -inf), 0.0),
        ((pi / 2, nan), nan),
        ((pi / 2, 2), complex(mp.ellipf(pi / 2, 2))),  # scipy does not handle complex
        ((0, 0.5), 0.0),
        ((inf, inf), nan),
        ((inf, -inf), nan),
        ((-inf, -inf), nan),
        ((-inf, inf), nan),
        ((nan, 0.5), nan),
        ((nan, nan), nan),
        # in our convention, ellipf is periodic in phi, so those don't hold
        # ((inf, 0.5), inf),
        # ((-inf, 0.5), -inf),
    ),
    ellipeinc: (
        ((1.5707, -10), 3.6388185585822876),
        ((35*pi/180, sin(52*pi/180)**2), 0.58823065),
        ((0, 0.5), 0.0),
        ((pi / 2, 0.0), pi / 2),
        ((pi / 2, 1.0), 1.0),
        ((pi / 2, 2), complex(mp.ellipe(pi / 2, 2))),  # scipy does not handle complex
        ((pi / 2, np.nan), np.nan),
        # TODO: ellipeinc infinities
        # ((pi / 2, -np.inf), np.inf),
        # ((np.inf, 0.5), np.inf),
        # ((-np.inf, 0.5), -np.inf),
        # ((np.inf, -np.inf), np.inf),
        # ((-np.inf, -np.inf), -np.inf),
        # ((np.inf, np.inf), np.nan),
        # ((-np.inf, np.inf), np.nan),
        # ((np.nan, 0.5), np.nan),
        # ((np.nan, np.nan), np.nan),
    )
}


class EllipCasesTest(BaseDtypeTest):
    @with_default_double
    @mark.parametrize('func, args, truth', tuple(
        (func, args, truth)
        for func, vals in cases.items() for args, truth in vals
    ))
    def test_case(self, func, args, truth):
        print(func, args, truth)
        assert close_complex_nan(res := func(*args), tensor(truth, dtype=res.dtype))


globals().update(make_dtype_tests((EllipCasesTest,), 'EllipCases'))


class TestEllipPi2:
    @staticmethod
    @given(st.floats(0, 1))
    def test(m):
        assert isclose(ellipk(m), ellipkinc(pi/2, m))
        assert isclose(ellipe(m), ellipeinc(pi/2, m))
        assert isclose(ellipd(m), ellipdinc(pi/2, m))

    @staticmethod
    @given(st.floats(0, 1), st.floats(0, 1))
    def test_ellippi(n, m):
        assert isclose(ellippi(n, m), ellippiinc(n, pi/2, m))


globals().update(make_dtype_tests((TestEllipPi2,), 'EllipPi2'))


# TODO: connections using float
class TestEllipConnections(BaseDoubleTest):
    @staticmethod
    @given(st.floats(0, 1, exclude_max=True))
    def test_ellippi_m_equals_n(m):
        # https://dlmf.nist.gov/19.6.E1, line 4
        assert nice_and_close(ellippi(m, m), ellipe(m) / (1-m))

        # https://dlmf.nist.gov/19.6.E2
        assert nice_and_close(ellippi(-m**0.5, m), pi/4 / (1+m**0.5) + ellipk(m) / 2)

    @staticmethod
    @given(st.floats(-inf, 1, exclude_min=True, exclude_max=True))
    def test_ellippi_every_n(n):
        # https://dlmf.nist.gov/19.6.E3
        assert nice_and_close(ellippi(n, 0), pi / 2 / (1-n)**0.5)

    @staticmethod
    @given(st.floats(0, pi/2))
    def test_every_phi(phi):
        # https://dlmf.nist.gov/19.6.E8
        assert nice_and_close(ellipkinc(phi, 1), asinh(tan(phi)))

        # https://dlmf.nist.gov/19.6.E9, line 4
        assert nice_and_close(ellipeinc(phi, 1), sin(phi))
        #
        # https://dlmf.nist.gov/19.6.E11, line 3
        assert nice_and_close(ellippiinc(1, phi, 0), tan(phi))

    @staticmethod
    @given(st.floats(0, pi/2))
    def test_every_phi_but_zero(phi):
        assume(sin(phi) != 0 and not isnan(ellippiinc(1, phi, 1)))
        c = 1 / sin(phi)**2

        # https://dlmf.nist.gov/19.6.E12
        assert nice_and_close(ellippiinc(1, phi, 1), (elliprc(c, c-1) + c**0.5 / (c-1)) / 2)

    @staticmethod
    @given(st.floats(0, pi/2), st.floats(0, 1))
    def test_ellippiinc_every_phi_m(phi, m):
        assume(sin(phi) != 0)
        c = 1 / sin(phi)**2
        D = (1-m/c)**0.5

        # https://dlmf.nist.gov/19.6.E12
        assert nice_and_close(ellippiinc(m, phi, 0), elliprc(c-1, c-m))
        assert nice_and_close(ellippiinc(m, phi, 1), (elliprc(c, c-1) - m * elliprc(c, c-m)) / (1-m))
        assert nice_and_close(ellippiinc(0, phi, m), ellipkinc(phi, m))
        # https://dlmf.nist.gov/19.6.E13
        assert nice_and_close(ellippiinc(m, phi, m), (ellipeinc(phi, m) - m/D * sin(phi)*cos(phi)) / (1-m))
        assert nice_and_close(ellippiinc(1, phi, m), ellipkinc(phi, m) - (ellipeinc(phi, m) - D*tan(phi)) / (1-m))

    @staticmethod
    @given(st.complex_numbers(min_magnitude=1e-9, max_magnitude=1e6))
    def test_legendre_relation(m: complex):
        k = m**0.5
        mm = 1-m
        kp = mm**0.5

        for left, right in (
            # https://dlmf.nist.gov/19.7.E2
            (ellipk((+1j * k / kp)**2), kp * ellipk(m)),
            (ellipk((-1j * kp / k)**2), k * ellipk(mm)),
            (ellipe((+1j * k / kp)**2), ellipe(m) / kp),
            (ellipe((-1j * kp / k)**2), ellipe(mm) / k),
            # https://dlmf.nist.gov/19.7.E3
            (ellipk(1/m), k * (ellipk(m) + (-1 if m.imag >= 0 else 1) * 1j*ellipk(mm))),
            (ellipk(1/mm), kp * (ellipk(mm) + (-1 if m.imag <= 0 else 1) * 1j*ellipk(m))),
            (ellipe(1/m), (ellipe(m) + (1 if m.imag >= 0 else -1) * 1j*ellipe(mm) - (mm*ellipk(m) + (1 if m.imag >= 0 else -1) * 1j*m*ellipk(mm))) / k),
            (ellipe(1/mm), (ellipe(mm) + (-1 if m.imag >= 0 else 1) * 1j*ellipe(m) - (m*ellipk(mm) + (-1 if m.imag >= 0 else 1) * 1j*mm*ellipk(m))) / kp)
        ):
            assert nice_and_close(left, right)

    @staticmethod
    @given(st.floats(**JUST_FINITE), st.floats(1e-9, 1), st.floats(1e-9, 1))
    def test_modulus_transformations(phi, n, m):
        assume(m != 0 and sin(phi)**2 != 0)

        # https://dlmf.nist.gov/19.7.E6
        kappap = 1 / (1+m)**0.5
        mu = m * kappap**2
        ctheta = (1 + m*sin(phi)**2) * kappap**2 / sin(phi)**2
        n1 = (n+m) * kappap**2

        # TODO: properly analytically continue
        # https://dlmf.nist.gov/19.7.E4
        cbeta = m / sin(phi)**2
        assert nice_and_close(ellipkinc(phi, 1/m), m**0.5 * ellipkinc(None, m, cbeta))
        assert nice_and_close(ellipeinc(phi, 1/m), (ellipeinc(None, m, cbeta) - (1-m)*ellipkinc(None, m, cbeta)) / m**0.5)
        assert nice_and_close(ellippiinc(n, phi, 1/m), m**0.5 * ellippiinc(m*n, None, m, cbeta))


        # https://dlmf.nist.gov/19.7.E5
        assert nice_and_close(ellipkinc(phi, -m), kappap * ellipkinc(None, mu, ctheta))
        assert nice_and_close(ellipeinc(phi, -m), (ellipeinc(None, mu, ctheta) - mu * ((ctheta-1) / (ctheta-mu) / ctheta)**0.5) / kappap)
        assert nice_and_close(ellippiinc(n, phi, -m), (kappap / n1) * (mu * ellipkinc(None, mu, ctheta) + kappap**2*n*ellippiinc(n1, None, mu, ctheta)))

    @staticmethod
    @given(st.floats(-100, 100, **JUST_FINITE), st.floats(1e-9, 1), st.floats(1e-9, 1))
    def test_argument_transformations(phi, n, m):
        # https://dlmf.nist.gov/19.7.E7
        assume(sinh(phi) != 0)
        cpsi = 1 + sinh(phi)**(-2)
        for left, right in (
            (ellipkinc(1j * phi, m), ellipkinc(None, 1-m, cpsi)),
            (ellipeinc(1j*phi, m), ellipkinc(None, 1-m, cpsi) - ellipeinc(None, 1-m, cpsi) + (1 if phi>=0 else -1) * sinh(phi) * (1-(1-m)/cpsi)**0.5),
            (ellippiinc(n, 1j*phi, m), (ellipkinc(None, 1-m, cpsi) - n*ellippiinc(1-n, None, 1-m, cpsi)) / (1-n))
        ):
            # TODO: properly analytically continue
            assert nice_and_close(abs(left.imag), abs(right.real))


# TODO: test ellipd


# class TestEllip:
#     def test_ellipkinc_2(self):
#         # Regression test for gh-3550
#         # ellipkinc(phi, mbad) was NaN and mvals[2:6] were twice the correct value
#         mbad = 0.68359375000000011
#         phi = 0.9272952180016123
#         m = np.nextafter(mbad, 0)
#         mvals = []
#         for j in range(10):
#             mvals.append(m)
#             m = np.nextafter(m, 1)
#         f = sp.ellipkinc(phi, mvals)
#         assert_array_almost_equal_nulp(f, np.full_like(f, 1.0259330100195334), 1)
#         # this bug also appears at phi + n * pi for at least small n
#         f1 = sp.ellipkinc(phi + pi, mvals)
#         assert_array_almost_equal_nulp(f1, np.full_like(f1, 5.1296650500976675), 2)
#
#     def test_ellipeinc_2(self):
#         # Regression test for gh-3550
#         # ellipeinc(phi, mbad) was NaN and mvals[2:6] were twice the correct value
#         mbad = 0.68359375000000011
#         phi = 0.9272952180016123
#         m = np.nextafter(mbad, 0)
#         mvals = []
#         for j in range(10):
#             mvals.append(m)
#             m = np.nextafter(m, 1)
#         f = sp.ellipeinc(phi, mvals)
#         assert_array_almost_equal_nulp(f, np.full_like(f, 0.84442884574781019), 2)
#         # this bug also appears at phi + n * pi for at least small n
#         f1 = sp.ellipeinc(phi + pi, mvals)
#         assert_array_almost_equal_nulp(f1, np.full_like(f1, 3.3471442287390509), 4)
