#pragma once

#include <utility>
#include <limits>

#include "../common/templating.h"
#include "../common/complex.h" // NOLINT(modernize-deprecated-headers)


#define EPS numeric_limits<scalar_t>::epsilon() * 1e-6

#define ELLIPR_CHECK_xyz \
if (isnan(x) or isnan(y) or isnan(z)) return T(x * y * z); \
if (isinf(x) or isinf(y) or isinf(z)) return T();


#define ELLIPR_HEADER(rfactor) \
T xm = x, ym = y, zm = z, A0 = Am;     \
auto Q = pow(EPS * (rfactor), -1./6.)  \
         * max(max(abs(A0 - x), abs(A0 - y)), abs(A0 - z));  \
scalar_t pow4 = 1.;


#define ELLIPR_STEP \
auto xs = sqrt(xm), ys = sqrt(ym), zs = sqrt(zm), \
     lm = xs*ys + xs*zs + ys*zs,                  \
     Am1 = (Am + lm) / ltrl(4);


#define ELLIPR_UPDATE(zm1) \
xm = (xm + lm) / ltrl(4); ym = (ym + lm) / ltrl(4); zm = (zm1) / ltrl(4); \
Am = Am1; pow4 /= 4.;


#define ELLIPR_XY auto t = pow4 / Am, X = (A0 - x) * t, Y = (A0 - y) * t;


DEFINE_COMPLEX_FUNCTION_H(elliprc, (x, y))
DEFINE_COMPLEX_FUNCTION_H(elliprd, (x, y, z))
DEFINE_COMPLEX_FUNCTION_H(elliprf, (x, y, z))
DEFINE_COMPLEX_FUNCTION_H(elliprg, (x, y, z))
DEFINE_COMPLEX_FUNCTION_H(elliprj, (x, y, z, p))

DEFINE_COMPLEX_FUNCTION_H(ellipk, (m))
DEFINE_COMPLEX_FUNCTION_H(ellipe, (m))
DEFINE_COMPLEX_FUNCTION_H(ellipd, (m))
DEFINE_COMPLEX_FUNCTION_H(ellippi, (n, m))

DEFINE_COMPLEX_FUNCTION_H(csc2, (phi))

DEFINE_COMPLEX_FUNCTION_H(ellipkinc_, (c, m))
DEFINE_COMPLEX_FUNCTION_H(ellipeinc_, (c, m))
DEFINE_COMPLEX_FUNCTION_H(ellipdinc_, (c, m))
DEFINE_COMPLEX_FUNCTION_H(ellippiinc_, (n, c, m))

DEFINE_COMPLEX_FUNCTION_H(ellipkinc, (phi, m))
DEFINE_COMPLEX_FUNCTION_H(ellipeinc, (phi, m))
DEFINE_COMPLEX_FUNCTION_H(ellipdinc, (phi, m))
DEFINE_COMPLEX_FUNCTION_H(ellippiinc, (n, phi, m))
