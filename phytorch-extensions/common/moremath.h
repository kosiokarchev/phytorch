#pragma once

#include <cmath>
#include <limits>

#define CEPHES_MAXLOG 7.09782712893383996843e2

#define LOGPI 1.14472988584940017414
#define SQRT2PI 2.50662827463100050242
#define R2SQRTPI 0.15915494309189535  // 1 / (2 * sqrt(pi))
#define LOGSQRT2PI 0.91893853320467274178
#define EULER 0.577215664901532860606512090082402431


#define ltrl(a) ((scalar_t) (a))

using std::numeric_limits;
using std::min; using std::max; using std::round; using std::floor;
using std::isfinite; using std::isnan; using std::isinf;
using std::abs;
using std::pow; using std::log2; using std::log;
using std::sin; using std::cos; using std::tan;

#define TINF numeric_limits<T>::infinity()
#define TNAN numeric_limits<T>::quiet_NaN()
#define TEPS numeric_limits<scalar_t>::epsilon()
#define TPREC numeric_limits<scalar_t>::digits
