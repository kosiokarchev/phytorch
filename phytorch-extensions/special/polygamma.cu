#include "special.cuh"
#include "gammahead.cuh"

#define POLYGAMMA_COEFFS ( \
    -13.000011048992318, -11.999911510243422, -11.000331481556389, -9.999215716230553, -9.00134490373618,               \
    -7.998118637023387, -7.0024851819328395, -5.99579520534724, -5.020526188298227, -4.091435542300593,                 \
    T(-4.385193550253947, 0.19149326909941256), T(-4.385193550253947, -0.19149326909941256),\
    T(-4.161470979872063, 0.1457810712519625), T(-4.161470979872063, -0.1457810712519625))
#define POLYGAMMA_TERM(r, data, i, elem) BOOST_PP_IF(i, +, ) pow(z-T(elem), -ltrl(n)-1) - pow(z+(i), -ltrl(n)-1)

COMPLEX_TEMPLATE T polygamma_(unsigned long n, T z) {
    return
        pow(ltrl(-1), ltrl(n) + 1) * (
            gamma<scalar_t>(T(n)) * pow(z+ltrl(polygamma_g - polygamma_h), -ltrl(n))
            + gamma<scalar_t>(T(n+1)) * pow(z+ltrl(polygamma_g - polygamma_h), -ltrl(n) - 1) * ltrl(polygamma_g))
        + pow(ltrl(-1), ltrl(n)) * gamma<scalar_t>(T(n+1)) * (
            BOOST_PP_SEQ_FOR_EACH_I(POLYGAMMA_TERM, _, BOOST_PP_TUPLE_TO_SEQ(POLYGAMMA_COEFFS)));
}


#define POLYGAMMA_POLY_COEFFS ( \
    (1, 0),                     \
    (0, -1),                    \
    (2, 0),                     \
    (-4, 0, -2),                \
    (8, 0, 16, 0),              \
    (-16, 0, -88, 0, -16),      \
    (32, 0, 416, 0, 272, 0),    \
    (-64, 0, -1824, 0, -2880, 0, -272),  \
    (128, 0, 7680, 0, 24576, 0, 7936, 0),  \
    (-256, 0, -31616, 0, -185856, 0, -137216, 0, -7936),  \
    (512, 0, 128512, 0, 1304832, 0, 1841152, 0, 353792, 0),  \
    (-1024, 0, -518656, 0, -8728576, 0, -21253376, 0, -9061376, 0, -353792),  \
    (2048, 0, 2084864, 0, 56520704, 0, 222398464, 0, 175627264, 0, 22368256, 0),  \
    (-4096, 0, -8361984, 0, -357888000, 0, -2174832640, 0, -2868264960, 0, -795300864, 0, -22368256),  \
    (8192, 0, 33497088, 0, 2230947840, 0, 20261765120, 0, 41731645440, 0, 21016670208, 0, 1903757312, 0),  \
    (-16384, 0, -134094848, 0, -13754155008, 0, -182172651520, 0, -559148810240, 0, -460858269696, 0, -89702612992, 0, -1903757312),  \
    (32768, 0, 536608768, 0, 84134068224, 0, 1594922762240, 0, 7048869314560, 0, 8885192097792, 0, 3099269660672, 0, 209865342976, 0))

COMPLEX_TEMPLATE T polygamma(unsigned long n, T z) {
    if (is_int(z) and is_real_nonpositive(z)) return cnan<T>();
    if (n==0) return digamma<scalar_t>(z);
    if (z.real() < 0) {
        T cospiz = cos(ltrl(M_PI) * z), poly;
#define a_case(r, data, i, elem) case (i): poly = HORNER(cospiz, elem); break;
        switch (n) {
            BOOST_PP_SEQ_FOR_EACH_I(a_case, _, BOOST_PP_TUPLE_TO_SEQ(POLYGAMMA_POLY_COEFFS))
            default:
                return cnan<T>();
        }
        return pow(ltrl(-1), ltrl(n)) * polygamma_<scalar_t>(n, 1-z) - poly * pow(sin(ltrl(M_PI) * z) / ltrl(M_PI), -ltrl(n)-1);
    }
    else return polygamma_<scalar_t>(n, z);
}
template __host__ __device__ complex<float> polygamma<float, complex<float>>(unsigned long, complex<float>);
template __host__ __device__ complex<double> polygamma<double, complex<double>>(unsigned long, complex<double>);
