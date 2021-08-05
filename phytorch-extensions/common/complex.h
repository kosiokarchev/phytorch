#pragma once

#include "moremath.h"

#define ltrl(a) ((scalar_t) (a))

#include "c10/util/complex.h" // NOLINT(modernize-deprecated-headers)
template <typename scalar_t> using complex = c10::complex<scalar_t>;

using std::numeric_limits;
using std::min; using std::max; using std::round; using std::floor;
using std::isfinite; using std::isnan; using std::isinf;
using std::conj; using std::abs; using std::arg;
using std::pow; using std::log2; using std::log; using std::sin; using std::cos; using std::tan;

#define TINF numeric_limits<T>::infinity()
#define TNAN numeric_limits<T>::quite_NaN()

template <typename T> __host__ __device__ T cnan() {
    static auto ret = numeric_limits<T>::quiet_NaN();
    return ret;
}

#define is_real(a) ((a).imag() == 0)
#define is_nonpositive(a) ((a) <= 0)
#define is_nonnegative(a) ((a) >= 0)
#define is_real_nonpositive(a) (is_real(a) and is_nonpositive((a).real()))
#define is_real_nonnegative(a) (is_real(a) and is_nonnegative((a).real()))
#define is_real_negative(a) (is_real(a) and not is_nonnegative((a).real()))
#define is_fint(a) (abs(round(a) - (a)) < 100*numeric_limits<decltype(a)>::epsilon())  // TODO: is_int
#define is_int(a) (is_real(a) and is_fint((a).real()))
#define is_nonpositive_int(a) (is_int(a) and is_nonpositive((a).real()))
#define are_conjugate(a, b) (conj(a) == (b))

#define DEF_COMPLEX_CHECK template <typename T> __host__ __device__ bool

DEF_COMPLEX_CHECK isfinite(complex<T> a) { return isfinite(a.real()) and isfinite(a.imag()); }

DEF_COMPLEX_CHECK isnan(complex<T> a) { return isnan(a.real()) or isnan(a.imag()); }

DEF_COMPLEX_CHECK isinf(complex<T> a) { return isinf(a.real()) or isinf(a.imag()); }
