#include "special.cuh"
#include "gammahead.cuh"

/*
 * (C) Copyright John Maddock 2006.
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 *  LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)
 */

// Computation of the Incomplete Gamma Function Ratios and their Inverse
// ARMIDO R. DIDONATO and ALFRED H. MORRIS, JR.
// ACM Transactions on Mathematical Software, Vol. 12, No. 4,
// December 1986, Pages 377-393.


REAL_TEMPLATE T find_inverse_s(const T& p, const T& q) {
    // eq. (32)
    auto t = (p < 0.5) ? sqrt(-2 * log(p)) : sqrt(-2 * log(q));
    auto s = t - (
        HORNER(t, (0.213623493715853, 4.28342155967104, 11.6616720288968, 3.31125922108741))
        / HORNER(t, (0.3611708101884203e-1, 1.27364489782223, 6.40691597760039, 6.61053765625462, 1))
    );
    return (p < 0.5) ? -s : s;
}

REAL_TEMPLATE T didonato_SN(const T& a, const T& x, unsigned N, const T& tolerance) {
    // eq. (34)
    T sum = 1;

    if (N >= 1) {
        unsigned i;
        auto partial = x / (a + 1);

        sum += partial;
        for(i = 2; i <= N; ++i) {
            partial *= x / (a + i);
            sum += partial;
            if(partial < tolerance) break;
        }
    }
    return sum;
}


REAL_TEMPLATE T find_inverse_gamma(const T& a, const T& p, const T& q) {
    if (a == 1)
        return (q > 0.9) ? -log1p(-p) : -log(q);
    else if (a < 1) {
        auto g = tgamma(a);
        auto b = q * g;

        if ((b > 0.6) || ((b >= 0.45) && (a >= 0.3))) {
            /* DiDonato & Morris Eq 21:
             *
             * There is a slight variation from DiDonato and Morris here:
             * the first form given here is unstable when p is close to 1,
             * making it impossible to compute the inverse of Q(a,x) for small
             * q. Fortunately the second form works perfectly well in this case.
             */
            auto u = ((b * q > 1e-8) && (q > 1e-5)) ?
                pow(p * g * a, 1 / a) : exp((-q / a) - EULER);
            return  u / (1 - (u / (a + 1)));
        } else if ((a < 0.3) && (b >= 0.35)) {
            /* DiDonato & Morris Eq 22: */
            auto t = exp(-EULER - b);
            return t * exp(t * exp(t));
        } else if ((b > 0.15) || (a >= 0.3)) {
            /* DiDonato & Morris Eq 23: */
            auto y = -log(b);
            auto u = y - (1 - a) * log(y);
            return y - (1 - a) * log(u) - log(1 + (1 - a) / (1 + u));
        } else if (b > 0.1) {
            /* DiDonato & Morris Eq 24: */
            auto y = -log(b);
            auto u = y - (1 - a) * log(y);
            return y - (1 - a) * log(u) - log((u * u + 2 * (3 - a) * u + (2 - a) * (3 - a)) / (u * u + (5 - a) * u + 2));
        } else {
            /* DiDonato & Morris Eq 25: */
            auto y = -log(b);
            auto c1 = (a - 1) * log(y);
            auto c1_2 = c1 * c1;
            auto c1_3 = c1_2 * c1;
            auto c1_4 = c1_2 * c1_2;
            auto a_2 = a * a;
            auto a_3 = a_2 * a;

            auto c2 = (a - 1) * (1 + c1);
            auto c3 = (a - 1) * (-(c1_2 / 2)
                                 + (a - 2) * c1
                                 + (3 * a - 5) / 2);
            auto c4 = (a - 1) * ((c1_3 / 3) - (3 * a - 5) * c1_2 / 2
                                 + (a_2 - 6 * a + 7) * c1
                                 + (11 * a_2 - 46 * a + 47) / 6);
            auto c5 = (a - 1) * (-(c1_4 / 4)
                                 + (11 * a - 17) * c1_3 / 6
                                 + (-3 * a_2 + 13 * a -13) * c1_2
                                 + (2 * a_3 - 25 * a_2 + 72 * a - 61) * c1 / 2
                                 + (25 * a_3 - 195 * a_2 + 477 * a - 379) / 12);

            auto y_2 = y * y;
            auto y_3 = y_2 * y;
            auto y_4 = y_2 * y_2;
            return y + c1 + (c2 / y) + (c3 / y_2) + (c4 / y_3) + (c5 / y_4);
        }
    } else {
        /* DiDonato and Morris Eq 31: */
        auto s = find_inverse_s<scalar_t>(p, q);

        auto s_2 = s * s;
        auto s_3 = s_2 * s;
        auto s_4 = s_2 * s_2;
        auto s_5 = s_4 * s;
        auto ra = sqrt(a);

        auto w = a + s * ra + (s_2 - 1) / 3;
        w += (s_3 - 7 * s) / (36 * ra);
        w -= (3 * s_4 + 7 * s_2 - 16) / (810 * a);
        w += (9 * s_5 + 256 * s_3 - 433 * s) / (38880 * a * ra);

        if ((a >= 500) && (fabs(1 - w / a) < 1e-6)) {
            return w;
        } else if (p > 0.5) {
            if (w < 3 * a)
                return w;
            else {
                auto D = fmax(2, a * (a - 1));
                auto lg = lgamma(a);
                auto lb = log(q) + lg;
                if (lb < -D * 2.3) {
                    /* DiDonato and Morris Eq 25: */
                    auto y = -lb;
                    auto c1 = (a - 1) * log(y);
                    auto c1_2 = c1 * c1;
                    auto c1_3 = c1_2 * c1;
                    auto c1_4 = c1_2 * c1_2;
                    auto a_2 = a * a;
                    auto a_3 = a_2 * a;

                    auto c2 = (a - 1) * (1 + c1);
                    auto c3 = (a - 1) * (-(c1_2 / 2)
                                         + (a - 2) * c1
                                         + (3 * a - 5) / 2);
                    auto c4 = (a - 1) * ((c1_3 / 3)
                                         - (3 * a - 5) * c1_2 / 2
                                         + (a_2 - 6 * a + 7) * c1
                                         + (11 * a_2 - 46 * a + 47) / 6);
                    auto c5 = (a - 1) * (-(c1_4 / 4)
                                         + (11 * a - 17) * c1_3 / 6
                                         + (-3 * a_2 + 13 * a -13) * c1_2
                                         + (2 * a_3 - 25 * a_2 + 72 * a - 61) * c1 / 2
                                         + (25 * a_3 - 195 * a_2 + 477 * a - 379) / 12);

                    auto y_2 = y * y;
                    auto y_3 = y_2 * y;
                    auto y_4 = y_2 * y_2;
                    return y + c1 + (c2 / y) + (c3 / y_2) + (c4 / y_3) + (c5 / y_4);
                }
                else {
                    /* DiDonato and Morris Eq 33: */
                    auto u = -lb + (a - 1) * log(w) - log(1 + (1 - a) / (1 + w));
                    return -lb + (a - 1) * log(u) - log(1 + (1 - a) / (1 + u));
                }
            }
        } else {
            auto z = w;
            auto ap1 = a + 1;
            auto ap2 = a + 2;
            if (w < 0.15 * ap1) {
                /* DiDonato and Morris Eq 35: */
                auto v = log(p) + lgamma(ap1);
                z = exp((v + w) / a);
                s = log1p(z / ap1 * (1 + z / ap2));
                z = exp((v + z - s) / a);
                s = log1p(z / ap1 * (1 + z / ap2));
                z = exp((v + z - s) / a);
                s = log1p(z / ap1 * (1 + z / ap2 * (1 + z / (a + 3))));
                z = exp((v + z - s) / a);
            }

            if ((z <= 0.01 * ap1) || (z > 0.7 * ap1))
                return z;
            else {
                /* DiDonato and Morris Eq 36: */
                auto ls = log(didonato_SN<scalar_t>(a, z, 100, ltrl(1e-4)));
                auto v = log(p) + lgamma(ap1);
                z = exp((v + z - ls) / a);
                return z * (1 - (a * log(z) - z - v + ls) / (a - z));
            }
        }
    }
}


DEFINE_REAL_FUNCTION(gammaincinv, (a, p)) {
    if (isnan(a) || isnan(p) || (a < 0) || (p < 0) || (p > 1))
        return TNAN;
    else if (p == 0) return 0;
    else if (p == 1) return TINF;
    else if (p > 0.9) return gammainccinv<scalar_t>(a, 1-p);


    auto x = find_inverse_gamma<scalar_t>(a, p, 1 - p);

    /* Halley's method */
    for (auto i = 0; i < 3; i++) {
        auto fac = igam_fac<scalar_t>(a, x);
        if (fac == 0) return x;

        auto f_fp = (gammainc<scalar_t>(a, x) - p) * x / fac;
        /* The ratio of the first and second derivatives simplifies */
        auto fpp_fp = (a - 1) / x - 1;

        x = x - ((isfinite(fpp_fp)) ?
            f_fp : f_fp / (1 - f_fp * fpp_fp / 2)
        );
    }
    return x;
}


DEFINE_REAL_FUNCTION(gammainccinv, (a, q)) {
    if (isnan(a) || isnan(q) || (a < 0) || (q < 0) || (q > 1)) return TNAN;
    else if (q == 0) return TINF;
    else if (q == 1) return 0;
    else if (q > 0.9) return gammaincinv<scalar_t>(a, 1-q);

    auto x = find_inverse_gamma<scalar_t>(a, 1 - q, q);

    for (auto i = 0; i < 3; i++) {
        auto fac = igam_fac<scalar_t>(a, x);
        if (fac == 0) return x;

        auto f_fp = (gammaincc<scalar_t>(a, x) - q) * x / (-fac);
        auto fpp_fp = -1 + (a - 1) / x;

        x = x - ((isfinite(fpp_fp)) ?
            f_fp : f_fp / (1 - f_fp * fpp_fp / 2)
        );
    }
    return x;
}
