#pragma once

#include "elliptic.h"

DEFINE_COMPLEX_FUNCTION(elliprc, (x, y)) {
    if (isinf(x) or isinf(y)) return ltrl(1) / (x*y);
    if (not y) return numeric_limits<T>::infinity();
    if (not x) return ltrl(M_PI_2) / sqrt(y);

    // principal value
    if (not y.imag() and y.real() < 0)
        return sqrt(x / (x-y)) * elliprc<scalar_t>(x-y, -y);

    auto Am = (x + y + y) / ltrl(3), xm = x, ym = y, A0 = Am;
    auto Q = pow(3*EPS, -1./6.) * abs(A0 - x);
    scalar_t pow4 = 1.;

    while (pow4 * Q > abs(Am)) {
        auto lm = 2*sqrt(xm)*sqrt(ym) + ym;
        xm = (xm + lm) / ltrl(4); ym = (ym + lm) / ltrl(4);
        Am = (Am + lm) / ltrl(4); pow4 /= 4.;
    }

    auto s = (y - A0) * pow4 / Am;
    return (
       1
       + s*s * ltrl(3./10.)
       + s*s*s / ltrl(7)
       + s*s*s*s * ltrl(3./8.)
       + s*s*s*s*s * ltrl(9./22.)
       + s*s*s*s*s*s * ltrl(159./208.)
       + s*s*s*s*s*s*s * ltrl(9./8.)
   ) / sqrt(Am);
}
