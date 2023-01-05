#pragma once

#include "gammahead.h"

COMPLEX_TEMPLATE inline T stirling(T z) {
    auto w = ltrl(1) / z;
    return ltrl(SQRT2PI) * pow(z, z - ltrl(0.5)) * exp(-z) * (
        ltrl(1) +
        ((((((ltrl(-5.92166437353693882865e-4) * w
            + ltrl(6.97281375836585777429e-5)) * w
            + ltrl(7.84039221720066627474e-4)) * w
            + ltrl(-2.29472093621399176955e-4)) * w
            + ltrl(-2.68132716049382716049e-3)) * w
            + ltrl(3.47222222222222222222e-3)) * w
            + ltrl(8.33333333333333333333E-2)) * w);
}


DEFINE_COMPLEX_FUNCTION(gamma, (z)) {
    if (is_int(z) and is_real_nonpositive(z)) return cnan<T>();
    if (abs(z.real()) >= 18) {
        if (z.real() < 0)
            return ltrl(M_PI) / sin(ltrl(M_PI) * z) / gamma<scalar_t>(1 - z);
        else return stirling<scalar_t>(z);
    }

    T c = 1;
    while (z.real() < 18) {
        if (abs(z) < 1e-9)
            return ltrl(1) / (((1 + ltrl(0.5772156649015329) * z) * z) * c);
        c *= z;
        z += 1;
    }
    return stirling<scalar_t>(z) / c;
}

