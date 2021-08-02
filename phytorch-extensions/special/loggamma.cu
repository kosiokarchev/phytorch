#include "special.cuh"

DEFINE_COMPLEX_FUNCTION(loggamma, (z)) {
    if (is_int(z) and is_real_nonpositive(z)) return numeric_limits<T>::infinity();

    T res;
    bool doconj = z.imag() < 0;
    if (doconj) z = conj(z);

    if (z.real() < -14 or z.imag() < -14) {
        res = ltrl(LOGPI) - loggamma<scalar_t>(1-z) - (
                (z.imag() > 36.7 or z.imag() < -36.7) ?
                ltrl(M_PI) * z.imag() - ltrl(0.6931471805599453094) + T(0, M_PI * (0.5 - z.real())) :
                log(sin(z-floor(z.real()))) - T(0, M_PI * floor(z.real()))
        );
    } else {
        T c = 1;
        scalar_t a = 0;

        if (z.real() < 14)
            while (z.real() < 14) {
                c *= z;
                a += arg(z);
                z += 1;
            }

        res = (z - ltrl(0.5)) * log(z) - z + ltrl(LOGSQRT2PI) - T(log(abs(c)), a);

        if (abs(z) < 1e8) {
            auto w = ltrl(1) / (z*z);
            // TODO: macro-ify
            res += (((((ltrl(-1.9175269175269175269175269175269175269175e-3) * w
                      + ltrl(8.4175084175084175084175084175084175084175E-4)) * w
                      + ltrl(-5.9523809523809523809523809523809523809524e-4)) * w
                      + ltrl(7.9365079365079365079365079365079365079365e-4)) * w
                      + ltrl(-2.7777777777777777777777777777777777777778e-3)) * w
                      + ltrl(8.3333333333333333333333333333333333333333e-2)) / z;
        }
    }

    return doconj ? conj(res) : res;
}
