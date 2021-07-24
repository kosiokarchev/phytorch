#include <utility>
#include <limits>

#include "../common/complex.h" // NOLINT(modernize-deprecated-headers)


#define EPS std::numeric_limits<scalar_t>::epsilon() * 1e-6


#define DEF_ELLIPR_KERNEL(f) \
template <typename scalar_t, typename T=complex<scalar_t>> __host__ __device__ \
T ellipr##f##_kernel


#define ELLIPR_CHECK_xyz \
if (isnan(x) or isnan(y) or isnan(z)) return T(x * y * z); \
if (isinf(x) or isinf(y) or isinf(z)) return T();


#define ELLIPR_HEADER(rfactor) \
T xm = x, ym = y, zm = z, A0 = Am;     \
auto Q = pow(EPS * (rfactor), -1./6.)  \
         * std::max(std::max(std::abs(A0 - x), std::abs(A0 - y)), std::abs(A0 - z));  \
scalar_t pow4 = 1.;


#define ELLIPR_STEP \
auto xs = sqrt(xm), ys = sqrt(ym), zs = sqrt(zm), \
     lm = xs*ys + xs*zs + ys*zs,                  \
     Am1 = (Am + lm) / ltrl(4);


#define ELLIPR_UPDATE(zm1) \
xm = (xm + lm) / ltrl(4); ym = (ym + lm) / ltrl(4); zm = (zm1) / ltrl(4); \
Am = Am1; pow4 /= 4.;


#define ELLIPR_XY auto t = pow4 / Am, X = (A0 - x) * t, Y = (A0 - y) * t;


DEF_ELLIPR_KERNEL(c)(T x, T y) {
    if (isinf(x) or isinf(y)) return ltrl(1) / (x*y);
    if (not y) return std::numeric_limits<T>::infinity();
    if (not x) return ltrl(M_PI_2) / sqrt(y);

    if (not y.imag() and y.real() < 0) return sqrt(x / (x-y)) * elliprc_kernel<scalar_t>(x-y, -y);
    if (std::abs(x-y) < 1e-6) return 1 / sqrt(x);

    return acos(sqrt(x) / sqrt(y)) / (sqrt(1-x/y) * sqrt(y));
}


DEF_ELLIPR_KERNEL(d)(T x, T y, T z) {
    ELLIPR_CHECK_xyz
    auto Am = (x + y + ltrl(3.)*z) / ltrl(5.);
    ELLIPR_HEADER(1./4.)

    T s = 0.;
    while (pow4 * Q > std::abs(Am)) {
        ELLIPR_STEP
        auto zm1 = zm + lm;
        s += ltrl(pow4) / sqrt(zm) / zm1;
        ELLIPR_UPDATE(zm1)
    }

    ELLIPR_XY
    auto Z = - (X + Y) / ltrl(3.),
            XY = X*Y, Z2 = Z*Z, XYZ = XY*Z,
            E2 = XY - ltrl(6.)*Z2,
            E3 = ltrl(3.)*XYZ - ltrl(8.)*Z2*Z,
            E4 = ltrl(3.) * (XY - Z2) * Z2,
            E5 = XYZ * Z2;
    return ltrl(3.)*s + ltrl(pow4) / Am / sqrt(Am) * (
            ltrl(1.)
            - ltrl(3./14.) * E2
            + E3 / ltrl(6.)
            + ltrl(9./88.) * E2*E2
            - ltrl(3./22.) * E4
            - ltrl(9./52.) * E2*E3
            + ltrl(3./26.) * E5);
}


DEF_ELLIPR_KERNEL(f)(T x, T y, T z) {
    if (y==z) return elliprc_kernel<scalar_t>(x, y);
    if (x==z) return elliprc_kernel<scalar_t>(y, x);
    if (x==y) return elliprc_kernel<scalar_t>(z, x);

    ELLIPR_CHECK_xyz

    auto Am = (x + y + z) / ltrl(3.);
    ELLIPR_HEADER(3.)

    while (pow4 * Q > std::abs(Am)) {
        ELLIPR_STEP
        ELLIPR_UPDATE(zm + lm)
    }

    ELLIPR_XY
    auto Z = - (X + Y),
            E2 = X*Y - Z*Z,
            E3 = X*Y*Z;

    return (ltrl(1.) - E2 / ltrl(10.) + E2 * E2 / ltrl(14.) + E3 / ltrl(24.) - ltrl(3. / 44.) * E2 * E3) / sqrt(Am);
}


#define are_equal_negative(a, b) (is_real_negative(a) and is_real_negative(b) and a.real() == b.real())

DEF_ELLIPR_KERNEL(g)(T x, T y, T z) {
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
        return std::numeric_limits<T>::quiet_NaN();

    return (z * elliprf_kernel<scalar_t>(x, y, z)
            - (x-z)*(y-z) * elliprd_kernel<scalar_t>(x, y, z) / ltrl(3.)
            + sqrt(x) * sqrt(y) / sqrt(z)) / ltrl(2.);
}


DEF_ELLIPR_KERNEL(j)(T x, T y, T z, T p) {
    if (isnan(x) or isnan(y) or isnan(z) or isnan(p)) return T(x * y * z);
    if (isinf(x) or isinf(y) or isinf(z) or isinf(p)) return T();
    if (not p or (not x + not y + not z > 1)) return std::numeric_limits<T>::infinity();

    // see Carlson (1995) for the following condition
    if (not (
        (x.real() >= 0 and y.real() >= 0 and z.real() >= 0 and p.real() > 0)
        or (p and (
            (is_real_nonnegative(x) and is_real_nonnegative(y) and is_real_nonnegative(z))
            or (is_real_nonnegative(x) and are_conjugate(y, z) and y)
            or (is_real_nonnegative(y) and are_conjugate(z, x) and z)
            or (is_real_nonnegative(z) and are_conjugate(x, y) and x)
        ))
        or (x == p or y == p or z == p)  // last paragraph of algorithm
    )) return std::numeric_limits<T>::quiet_NaN();

    // The following is the implementation of the above in mpmath:
    // if (not (x.real() >= 0 and y.real() >= 0 and z.real() >= 0 and p.real() > 0)
    //     and not (x == p or y == p or z == p)
    //     and not (
    //         (p.imag() != 0 or p.real() != 0) and (
    //             (x.imag() == 0 and x.real() >= 0 and std::conj(y) == z)
    //             or (y.imag() == 0 and y.real() >= 0 and std::conj(x) == z)
    //             or (z.imag() == 0 and z.real() >= 0 and std::conj(x) == y)
    //         )
    // )) return std::numeric_limits<T>::quiet_NaN();

    auto Am = (x + y + z + p + p) / ltrl(5);

    T xm = x, ym = y, zm = z, pm = p, A0 = Am,
      delta = (p-x)*(p-y)*(p-z);
    auto Q = pow(EPS / ltrl(4), -1./6.)
             * std::max(std::max(std::max(std::abs(A0 - x), std::abs(A0 - y)), std::abs(A0 - z)), std::abs(A0 - p));
    scalar_t pow4 = 1.;

    T s = 0.;
    int m = 0;
    while (pow4 * Q >= std::abs(Am)) {
        auto xs = sqrt(xm), ys = sqrt(ym), zs = sqrt(zm), ps = sqrt(pm),
             lm = xs*ys + xs*zs + ys*zs;
        xm = (xm + lm) / ltrl(4); ym = (ym + lm) / ltrl(4); zm = (zm + lm) / ltrl(4); pm = (pm + lm) / ltrl(4), Am = (Am + lm) / ltrl(4);
        auto dm = (ps+xs) * (ps+ys) * (ps+zs),
             em = delta * T(pow(4., -3*m)) / (dm*dm);
        s += elliprc_kernel<scalar_t>(T(1.), ltrl(1)+em) * pow4 / dm;
        pow4 /= 4.; m += 1;
    }

    auto t = pow4 / Am, X = (A0 - x) * t, Y = (A0 - y) * t, Z = (A0 - z)*t, P = (-X-Y-Z) / ltrl(2),
         E2 = X*Y + X*Z + Y*Z - 3*P*P,
         E3 = X*Y*Z + 2*E2*P + 4*P*P*P,
         E4 = (2*X*Y*Z + E2*P + 3*P*P*P) * P,
         E5 = X*Y*Z*P*P;

    return ltrl(6)*s + pow4 / Am / sqrt(Am) * (
        1 - ltrl(3./14.) * E2 + E3 / ltrl(6) + ltrl(9./88.) * E2*E2
        - ltrl(3./22.) * E4 - ltrl(9./52.) * E2*E3 + ltrl(3./26.) * E5
    );
}


#include "../common/implement.h"
IMPLEMENT(elliprc, (T x, T y), (x, y), T)
IMPLEMENT(elliprd, (T x, T y, T z), (x, y, z), T)
IMPLEMENT(elliprf, (T x, T y, T z), (x, y, z), T)
IMPLEMENT(elliprg, (T x, T y, T z), (x, y, z), T)
IMPLEMENT(elliprj, (T x, T y, T z, T p), (x, y, z, p), T)

