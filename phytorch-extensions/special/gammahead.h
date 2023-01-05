#pragma once

#include "../common/templating.h"

// // from https://www.mathworks.com/matlabcentral/fileexchange/978-special-functions-math-library
#define polygamma_g (607./128.)
#define polygamma_h 0.5
#define lanczos_g 6.024680040776729583740234375

DEFINE_COMPLEX_FUNCTION_H(gamma, (z))
DEFINE_COMPLEX_FUNCTION_H(loggamma, (z))
DEFINE_COMPLEX_FUNCTION_H(digamma, (z))

DEFINE_REAL_FUNCTION_H(igam_fac, (a, x))

DEFINE_REAL_FUNCTION_H(gammainc, (a, x))
DEFINE_REAL_FUNCTION_H(gammaincc, (a, x))
DEFINE_REAL_FUNCTION_H(gammaincinv, (a, p))
DEFINE_REAL_FUNCTION_H(gammainccinv, (a, q))
