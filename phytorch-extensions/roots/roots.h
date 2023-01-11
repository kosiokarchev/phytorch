#pragma once

#include "../common/templating.h"

#define MACRO(z, n, text) BOOST_PP_COMMA_IF(n) text
#define ROOTS_RETTYPE(N) std::tuple<BOOST_PP_REPEAT(N, MACRO, T)>
#define ROOTS_TEMPLATE(N) template <typename scalar_t, typename T=complex<scalar_t>> PHYTORCH_DEVICE ROOTS_RETTYPE(N)
#define DEF_ROOTS(N) ROOTS_TEMPLATE(N) roots##N


DEF_ROOTS(2)(T b, T c) {
    if (not (isfinite(b) and isfinite(c)))
        return {TNAN, TNAN};
    if (c == ltrl(0)) return {ltrl(0), -b};
    auto q = sqrt(b*b - 4*c);
    q = -(b.real()>=0 ? (b+q) : (b-q)) / 2;
    return {q, c/q};
}


DEF_ROOTS(3)(T b, T c, T d) {
    if (not (isfinite(b) and isfinite(c) and isfinite(d)))
        return {TNAN, TNAN, TNAN};

    auto Q = (b*b - 3*c) / 9, R = (2*b*b*b - 9*b*c + 27*d) / 54;
    auto A = sqrt((R*R - Q*Q*Q));
    A = - pow((R.real() >= 0 ? R+A : R-A), ltrl(1./3.));
    auto B = A == ltrl(0) ? 0 : Q/A;

    auto AB = A+B, Re = -AB/2 - b/3, Im = ltrl(sqrt(3) / 2) * (A-B);
    return {AB - b/3, Re + TIMAG*Im, Re - TIMAG*Im};
}


ROOTS_TEMPLATE(4) roots4_depressed(T p, T q, T r) {
    if (q == ltrl(0)) {
        auto [r1, r2] = roots2<scalar_t>(p, r);
        auto s1 = sqrt(r1), s2 = sqrt(r2);
        return {s1, -s1, s2, -s2};
    }

    auto [r1, r2, r3] = roots3<scalar_t>(2*p, p*p-4*r, -q*q);
    auto s1 = sqrt(r2), s2 = sqrt(r3), s3 = -q / (s1*s2);
    return {(s1+s2+s3) / 2, (s1-s2-s3) / 2, (s2-s1-s3) / 2, (s3-s1-s2) / 2};
}

// TODO: prettier sort4(a1, r1, a2, r2, a3, r3, a4, r4)
template <typename Tin, typename T>
PHYTORCH_DEVICE inline ROOTS_RETTYPE(4) sort4(
    const Tin& v1, const T& r1,
    const Tin& v2, const T& r2,
    const Tin& v3, const T& r3,
    const Tin& v4, const T& r4
) {
    if (v1 > v2)
        if (v2 > v3)
            if (v4 > v2)
                if (v4 > v1)
                    return {r4, r1, r2, r3};
                else
                    return {r1, r4, r2, r3};
            else
                if (v4 > v3)
                    return {r1, r2, r4, r3};
                else
                    return {r1, r2, r3, r4};
        else
            if (v1 > v3)
                if (v4 > v3)
                    if (v4 > v1)
                        return {r4, r1, r3, r2};
                    else
                        return {r1, r4, r3, r2};
                else
                    if (v4 > v2)
                        return {r1, r3, r4, r2};
                    else
                        return {r1, r3, r2, r4};
            else
                if (v4 > v1)
                    if (v4 > v3)
                        return {r4, r3, r1, r2};
                    else
                        return {r3, r4, r1, r2};
                else
                    if (v4 > v2)
                        return {r3, r1, r4, r2};
                    else
                        return {r3, r1, r2, r4};
    else
        if (v1 > v3)
            if (v4 > v1)
                if (v4 > v2)
                    return {r4, r2, r1, r3};
                else
                    return {r2, r4, r1, r3};
            else
                if (v4 > v3)
                    return {r2, r1, r4, r3};
                else
                    return {r2, r1, r3, r4};
        else
            if (v2 > v3)
                if (v4 > v3)
                    if (v4 > v2)
                        return {r4, r2, r3, r1};
                    else
                        return {r2, r4, r3, r1};
                else
                    if (v4 > v1)
                        return {r2, r3, r4, r1};
                    else
                        return {r2, r3, r1, r4};
            else
                if (v4 > v2)
                    if (v4 > v3)
                        return {r4, r3, r2, r1};
                    else
                        return {r3, r4, r2, r1};
                else
                    if (v4 > v1)
                        return {r3, r2, r4, r1};
                    else
                        return {r3, r2, r1, r4};
}

DEF_ROOTS(4)(T b, T c, T d, T e) {
    if (not (isfinite(b) and isfinite(c) and isfinite(d) and isfinite(e)))
        return {TNAN, TNAN, TNAN, TNAN};

    auto [r1, r2, r3, r4] = roots4_depressed<scalar_t>(
        (8*c - 3*b*b) / 8,
        (b*b*b - 4*b*c + 8*d) / 8,
        (-3*b*b*b*b + 256*e - 64*b*d + 16*b*b*c) / 256
    );
    r1 -= b/4; r2 -= b/4; r3 -= b/4; r4 -= b/4;

    auto a1 = abs(r1), a2 = abs(r2), a3 = abs(r3), a4 = abs(r4);
    return sort4(a1, r1, a2, r2, a3, r3, a4, r4);
}
