#pragma once

#include "device.h"
#include "moremath.h"

#include "c10/util/complex.h" // NOLINT(modernize-deprecated-headers)
template <typename scalar_t> using complex = c10::complex<scalar_t>;

using std::conj; using std::arg; using std::abs;

#define TIMAG T(0, 1)

template <typename T> PHYTORCH_DEVICE T cnan() {
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

#define DEF_COMPLEX_CHECK template <typename T> PHYTORCH_DEVICE bool

DEF_COMPLEX_CHECK isfinite(complex<T> a) { return isfinite(a.real()) and isfinite(a.imag()); }

DEF_COMPLEX_CHECK isnan(complex<T> a) { return isnan(a.real()) or isnan(a.imag()); }

DEF_COMPLEX_CHECK isinf(complex<T> a) { return isinf(a.real()) or isinf(a.imag()); }
