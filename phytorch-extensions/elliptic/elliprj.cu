#include "elliptic.cuh"


DEFINE_COMPLEX_FUNCTION(elliprj, (x, y, z, p)) {
    if (isnan(x) or isnan(y) or isnan(z) or isnan(p)) return T(x * y * z);
    if (isinf(x) or isinf(y) or isinf(z) or isinf(p)) return T();
    if (not p or (not x + not y + not z > 1)) return numeric_limits<T>::infinity();

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
    )) return cnan<T>();

    // The following is the implementation of the above in mpmath:
    // if (not (x.real() >= 0 and y.real() >= 0 and z.real() >= 0 and p.real() > 0)
    //     and not (x == p or y == p or z == p)
    //     and not (
    //         (p.imag() != 0 or p.real() != 0) and (
    //             (x.imag() == 0 and x.real() >= 0 and conj(y) == z)
    //             or (y.imag() == 0 and y.real() >= 0 and conj(x) == z)
    //             or (z.imag() == 0 and z.real() >= 0 and conj(x) == y)
    //         )
    // )) return cnan<T>();

    auto Am = (x + y + z + p + p) / ltrl(5);

    T xm = x, ym = y, zm = z, pm = p, A0 = Am,
            delta = (p-x)*(p-y)*(p-z);
    auto Q = pow(EPS / ltrl(4), -1./6.)
             * max(max(max(abs(A0 - x), abs(A0 - y)), abs(A0 - z)), abs(A0 - p));
    scalar_t pow4 = 1.;

    T s = 0.;
    int m = 0;
    while (pow4 * Q >= abs(Am)) {
        auto xs = sqrt(xm), ys = sqrt(ym), zs = sqrt(zm), ps = sqrt(pm),
                lm = xs*ys + xs*zs + ys*zs;
        xm = (xm + lm) / ltrl(4); ym = (ym + lm) / ltrl(4); zm = (zm + lm) / ltrl(4); pm = (pm + lm) / ltrl(4), Am = (Am + lm) / ltrl(4);
        auto dm = (ps+xs) * (ps+ys) * (ps+zs),
                em = delta * T(pow(4., -3*m)) / (dm*dm);
        s += elliprc<scalar_t>(T(1.), ltrl(1)+em) * pow4 / dm;
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
