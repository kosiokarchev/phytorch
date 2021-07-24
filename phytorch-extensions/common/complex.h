#pragma once

#define ltrl(a) ((scalar_t) (a))

#include "c10/util/complex.h" // NOLINT(modernize-deprecated-headers)
template <typename scalar_t> using complex = c10::complex<scalar_t>;

#define is_nonnegative(a) ((a) >= 0)
#define is_real(a) ((a).imag() == 0)
#define is_real_nonnegative(a) (is_real(a) and is_nonnegative((a).real()))
#define is_real_negative(a) (is_real(a) and not is_nonnegative((a).real()))
#define are_conjugate(a, b) (std::conj(a) == (b))

#define DEF_COMPLEX_CHECK template <typename T> bool

DEF_COMPLEX_CHECK isfinite(complex<T> a) { return std::isfinite(a.real()) and std::isfinite(a.imag()); }

DEF_COMPLEX_CHECK isnan(complex<T> a) { return std::isnan(a.real()) or std::isnan(a.imag()); }

DEF_COMPLEX_CHECK isinf(complex<T> a) { return std::isinf(a.real()) or std::isinf(a.imag()); }
