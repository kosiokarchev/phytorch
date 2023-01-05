#pragma once

#include "elliptic.h"


DEFINE_COMPLEX_FUNCTION(elliprj, (x, y, z, p)) {
    if (isnan(x) or isnan(y) or isnan(z) or isnan(p)) return T(x * y * z);
    if (isinf(x) or isinf(y) or isinf(z) or isinf(p)) return T();
    if (not p or (not x + not y + not z > 1)) return numeric_limits<T>::infinity();

    // see Carlson (1995) for the following condition
    if (not (
        (x.real() >= 0 and y.real() >= 0 and z.real() >= 0 and p.real() > 0)
        or (p and (
            (is_real_nonnegative(x) and is_real_nonnegative(y) and is_real_nonnegative(z))
            or (is_real_nonnegative(x) and are_conjugate(y, z) and y)
            or (is_real_nonnegative(y) and are_conjugate(z, x) and z)
            or (is_real_nonnegative(z) and are_conjugate(x, y) and x)
        ))
        or (x == p or y == p or z == p)  // last paragraph of algorithm
    )) {
// #define CHICKEN
        printf("Carlson's elliprj algorithm not guaranteed."
#ifdef CHICKEN
               " Chickening out!"
#else
               " But nobody calls me chicken!"
#endif
               "\n");
#ifdef CHICKEN
        return cnan<T>();
#endif
    }

    // The following is the implementation of the above in mpmath:
    // if (not (x.real() >= 0 and y.real() >= 0 and z.real() >= 0 and p.real() > 0)
    //     and not (x == p or y == p or z == p)
    //     and not (
    //         (p.imag() != 0 or p.real() != 0) and (
    //             (x.imag() == 0 and x.real() >= 0 and conj(y) == z)
    //             or (y.imag() == 0 and y.real() >= 0 and conj(x) == z)
    //             or (z.imag() == 0 and z.real() >= 0 and conj(x) == y)
    //         )
    // )) return cnan<T>();

    auto Am = (x + y + z + p + p) / 5;

    T xm = x, ym = y, zm = z, pm = p, A0 = Am,
    delta = (p-x)*(p-y)*(p-z);
    auto Q = pow(0.25 * numeric_limits<scalar_t>::epsilon() * pow(2, 10), ltrl(-1./6.))
             * max(max(max(abs(A0-x), abs(A0-y)), abs(A0-z)), abs(A0-p));
    scalar_t pow4 = 1;

    T s = 0;
    int m = 0;
    while (pow4 * Q >= abs(Am)) {
        auto sx = sqrt(xm), sy = sqrt(ym), sz = sqrt(zm), sp = sqrt(pm),
             lm = sx*sy + sx*sz + sy*sz;
        xm = (xm + lm) / 4, ym = (ym + lm) / 4, zm = (zm + lm) / 4, pm = (pm + lm) / 4, Am = (Am + lm) / 4;
        auto dm = (sp + sx) * (sp + sy) * (sp + sz),
             em = delta * pow(ltrl(4), ltrl(-3*m)) / (dm*dm);
        // s += elliprc<scalar_t>(T(1.), 1+em) * pow4 / dm;  // Replaced according to Johansson, 2018 (1806.06725)
        s += atan(sqrt(em)) / sqrt(em) * pow4 / dm;
        pow4 /= 4; m += 1;
    }

    auto t = pow4 / Am, X = (A0 - x) * t, Y = (A0 - y) * t, Z = (A0 - z) * t, P = (-X-Y-Z) / 2,
            E2 = X*Y + X*Z + Y*Z - 3*P*P,
            E3 = X*Y*Z + 2*E2*P + 4*P*P*P,
            E4 = (2*X*Y*Z + E2*P + 3*P*P*P) * P,
            E5 = X*Y*Z*P*P;

    return 6*s + pow4 / Am / sqrt(Am) * (
        1
        - E2 * ltrl(3./14.)
        + E3 / ltrl(6)
        + E2*E2 * ltrl(9./88.)
        - E4 * ltrl(3./22.)
        - E2*E3 * ltrl(9./52.)
        + E5 * ltrl(3./26.)
        // Extra terms
        // - E2*E2*E2 / ltrl(16)
        // + E3*E3 * ltrl(3./40.)
        // + E2*E4 * ltrl(3./20.)
        // + E2*E2*E3 * ltrl(45./272.)
        // - (E3*E4 + E2*E5) * ltrl(9./68.)
    );
}
