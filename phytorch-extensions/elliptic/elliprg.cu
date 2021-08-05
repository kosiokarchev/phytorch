#include "elliptic.cuh"


#define are_equal_negative(a, b) (is_real_negative(a) and is_real_negative(b) and (a).real() == (b).real())

DEFINE_COMPLEX_FUNCTION(elliprg, (x, y, z)) {
    switch (not x + not y + not z) {
        case 3:
            return (x+y+z) * ltrl(0.);
        case 2:
            if (x) return sqrt(x) / ltrl(2.);
            if (y) return sqrt(y) / ltrl(2.);
            return sqrt(z) / ltrl(2.);
        case 1:
            if (not z) std::swap(x, z);
    }
    if (are_equal_negative(x, y) or are_equal_negative(x, z) or are_equal_negative(y, z))
        return cnan<T>();

    return (z * elliprf<scalar_t>(x, y, z)
            - (x-z)*(y-z) * elliprd<scalar_t>(x, y, z) / ltrl(3.)
            + sqrt(x) * sqrt(y) / sqrt(z)) / ltrl(2.);
}