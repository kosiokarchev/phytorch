#include "elliptic.cuh"


DEFINE_COMPLEX_FUNCTION(elliprc, (x, y)) {
    if (isinf(x) or isinf(y)) return ltrl(1) / (x*y);
    if (not y) return numeric_limits<T>::infinity();
    if (not x) return ltrl(M_PI_2) / sqrt(y);

    if (not y.imag() and y.real() < 0) return sqrt(x / (x-y)) * elliprc<scalar_t>(x-y, -y);
    // TODO: handle x=y in elliprc better
    if (abs(sqrt(1-x/y)) < max((scalar_t) 1e-3, 100*sqrt(numeric_limits<scalar_t>::epsilon())))
        return (ltrl(7./6.) - (x/y) / ltrl(6)) / sqrt(y);

    return acos(sqrt(x) / sqrt(y)) / (sqrt(1-x/y) * sqrt(y));
}
