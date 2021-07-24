from math import inf, nan, pi, sin

import numpy as np
import scipy.special as sp
import torch
from mpmath import mp

from phytorch.special.ellip import ellipe, ellipeinc, ellipk, ellipkinc, ellippi, ellippiinc
from tests.common import close_complex_nan, CLOSE_KWARGS, process_cases, with_default_double


@with_default_double
def test_grid():
    # TODO: separate functions
    m, phi = torch.linspace(0, 1, 101), torch.linspace(0, pi/2, 101).unsqueeze(-1)

    for ours, theirs, args in (
        (ellipk, sp.ellipk, (m,)),
        (ellipe, sp.ellipe, (m,)),
        (ellipkinc, sp.ellipkinc, (phi, m)),
        (ellipeinc, sp.ellipeinc, (phi, m)),
        (ellippi, (sp_ellippi := np.vectorize(lambda *a, **kw: float(mp.ellippi(*a, *kw)))), (m[:-1].unsqueeze(-2), m[:-1])),  # avoid edges
        (ellippiinc, sp_ellippi, (m[:-1].unsqueeze(-2).unsqueeze(-3), phi, m[:-1])),  # avoid edges
    ):
        assert torch.allclose(ours(*args).real, torch.as_tensor(theirs(*args)), **CLOSE_KWARGS, equal_nan=True)


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


@with_default_double
def test_cases():
    for func, vals in cases.items():
        ins, outs, res = process_cases(func, vals)
        assertion = close_complex_nan(outs.to(res), res, close_func=torch.isclose)
        if not assertion.all():
            print(func)
            print(ins[~assertion])
            print(outs[~assertion])
            print(res[~assertion])
            print(80*'=')
        assert assertion.all()


# TODO: test ellipd
# TODO: test incomplete(pi/2) == complete


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
#     def test_ellipkinc_singular(self):
#         # ellipkinc(phi, 1) has closed form and is finite only for phi in (-pi/2, pi/2)
#         xlog = np.logspace(-300, -17, 25)
#         xlin = np.linspace(1e-17, 0.1, 25)
#         xlin2 = np.linspace(0.1, pi/2, 25, endpoint=False)
#
#         assert_allclose(sp.ellipkinc(xlog, 1), np.arcsinh(np.tan(xlog)), rtol=1e14)
#         assert_allclose(sp.ellipkinc(xlin, 1), np.arcsinh(np.tan(xlin)), rtol=1e14)
#         assert_allclose(sp.ellipkinc(xlin2, 1), np.arcsinh(np.tan(xlin2)), rtol=1e14)
#         assert_equal(sp.ellipkinc(np.pi / 2, 1), np.inf)
#         assert_allclose(sp.ellipkinc(-xlog, 1), np.arcsinh(np.tan(-xlog)), rtol=1e14)
#         assert_allclose(sp.ellipkinc(-xlin, 1), np.arcsinh(np.tan(-xlin)), rtol=1e14)
#         assert_allclose(sp.ellipkinc(-xlin2, 1), np.arcsinh(np.tan(-xlin2)), rtol=1e14)
#         assert_equal(sp.ellipkinc(-np.pi / 2, 1), np.inf)
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
