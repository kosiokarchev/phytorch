#include "elliptic.cuh"

DEFINE_COMPLEX_FUNCTION(csc2, (phi)) {
    auto s = sin(phi);
    return (s == ltrl(0)) ? std::numeric_limits<T>::infinity() : pow(s, -2);
}

DEFINE_COMPLEX_FUNCTION(ellipk, (m)) {
    return elliprf<scalar_t, T>(0, 1-m, 1);
}

DEFINE_COMPLEX_FUNCTION(ellipe, (m)) {
    if (m == T(TINF, 0)) return T(0, TINF);
    if (m == T(-TINF, 0)) return T(TINF, 0);
    return 2 * elliprg<scalar_t, T>(0, 1-m, 1);
}

DEFINE_COMPLEX_FUNCTION(ellipd, (m)) {
    return elliprd<scalar_t, T>(0, 1-m, 1) / 3;
}

DEFINE_COMPLEX_FUNCTION(ellippi, (n, m)) {
    if (n == ltrl(0)) return ellipk<scalar_t>(m);
    return ellipk<scalar_t>(m) + n / 3 * elliprj<scalar_t, T>(0, 1-m, 1, 1-n);
}

DEFINE_COMPLEX_FUNCTION(ellipkinc_, (c, m)) {
    return elliprf<scalar_t>(c-1, c-m, c);
}

DEFINE_COMPLEX_FUNCTION(ellipeinc_, (c, m)) {
    if (c == ltrl(1) and m == ltrl(1)) return 1;
    return elliprf<scalar_t>(c-1, c-m, c) - m * ellipdinc_<scalar_t>(c, m);
}

DEFINE_COMPLEX_FUNCTION(ellipdinc_, (c, m)) {
    return elliprd<scalar_t>(c-1, c-m, c) / 3;
}

DEFINE_COMPLEX_FUNCTION(ellippiinc_, (n, c, m)) {
    if (n == ltrl(0)) return ellipkinc_<scalar_t>(c, m);
    return ellipkinc_<scalar_t>(c, m) + n / 3 * elliprj<scalar_t>(c-1, c-m, c, c-n);
}

DEFINE_COMPLEX_FUNCTION(ellipkinc, (phi, m)) {
    return ellipkinc_<scalar_t>(csc2<scalar_t>(phi), m);
}

DEFINE_COMPLEX_FUNCTION(ellipeinc, (phi, m)) {
    return ellipeinc_<scalar_t>(csc2<scalar_t>(phi), m);
}

DEFINE_COMPLEX_FUNCTION(ellipdinc, (phi, m)) {
    return ellipdinc_<scalar_t>(csc2<scalar_t>(phi), m);
}

DEFINE_COMPLEX_FUNCTION(ellippiinc, (n, phi, m)) {
    return ellippiinc_<scalar_t>(n, csc2<scalar_t>(phi), m);
}