#pragma once

#include "../common/templating.h"
#include "../common/moremath.h"
#include "../common/complex.h" // NOLINT(modernize-deprecated-headers)

DEFINE_COMPLEX_FUNCTION_H(gamma, (z))
DEFINE_COMPLEX_FUNCTION_H(loggamma, (z))
DEFINE_COMPLEX_FUNCTION_H(digamma, (z))
COMPLEX_TEMPLATE_H T polygamma(unsigned long n, T z);

DEFINE_REAL_FUNCTION_H(gammainc, (a, x))
DEFINE_REAL_FUNCTION_H(gammaincc, (a, x))
DEFINE_REAL_FUNCTION_H(gammaincinv, (a, p))
DEFINE_REAL_FUNCTION_H(gammainccinv, (a, q))

// DEFINE_COMPLEX_FUNCTION_H(hyp2f1, (a, b, c, z))

DEFINE_COMPLEX_FUNCTION_H(deta1, (z))
DEFINE_COMPLEX_FUNCTION_H(zeta, (z))
