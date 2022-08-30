#include "special.cuh"

#define GAMMA(a) gamma<scalar_t>((a))
#define LGAMMA(a) lgamma((a).real())


template <typename scalar_t> __host__ __device__ scalar_t mag2(scalar_t a) {return log2(abs(a));}
template <typename scalar_t> __host__ __device__ scalar_t mag2(complex<scalar_t> a) {return log2(abs(a));}


COMPLEX_TEMPLATE T hyp2f1_gosper(const T& a, const T& b, const T& c, const T& z) {
    T f1;

    printf("hyp2f1_gosper!\n");

    auto abz = a * b * z,
         nz = ltrl(1) - z,
         g = z / nz,
         abg = a * b * g,
         z2 = z - ltrl(2),
         ch = c / ltrl(2),
         c1h = (c + ltrl(1)) / ltrl(2),
         cba = c - b - a;
    auto tol = mag2(numeric_limits<T>::epsilon()) - 10;

    scalar_t maxprec=1000, extra = 10;
    while (true) {
        auto maxmag = -numeric_limits<scalar_t>::infinity();
        T d = ltrl(0), e = ltrl(1), f = ltrl(0);
        for (scalar_t k = 0; true; ++k) { // NOLINT(cert-flp30-c)
            auto kch = k + ch;
            auto kakbz = (k + a) * (k + b) * z / (ltrl(4) * (k + ltrl(1)) * kch * (k + c1h)),
                    d1 = kakbz * (e - (k + cba) * d * g),
                    e1 = kakbz * (d * abg + (k + c) * e),
                    ft = d * (k * (cba * z + k * z2 - c) - abz) / (2 * kch * nz);
            f1 = f + e - ft;
            maxmag = max(maxmag, mag2(f1));
            if (mag2(f1 - f) < tol) break;

            d = d1, e = e1, f = f1;
        }
        auto cancellation = maxmag - mag2(f1);
        if (cancellation < extra) break;
        else {
            extra += cancellation;
            if (extra > maxprec) return cnan<scalar_t>();
        }
    }

    return f1;
}


template <int p, int q, typename scalar_t, typename T=complex<scalar_t>> __host__ __device__
T hypsum(const std::array<T, p+q>& coeffs, const T& z) {
    T s = 1., t = 1.;
    unsigned int k = 0;
    while (true) {
        for (auto i=0; i<p; ++i) t *= coeffs[i] + k;
        for (auto i=p; i<p+q; ++i) t /= coeffs[i] + k;
        k += 1; t /= k; t *= z; s += t;
        if (abs(t) < numeric_limits<scalar_t>::epsilon())
            return s;
        if (k > 6000) {
            printf("did not converge in 6000 terms...\n");
            return cnan<scalar_t>();
        }
    }
}

#define PERTURBATION (min(pow(numeric_limits<T>::epsilon(), 0.5), 1e-3))
#define hyp2f1_zerodiv(a, b, c) ( \
    is_nonpositive_int(c) and not ( \
        (is_nonpositive_int(a) and (c).real() <= (a).real()) or \
        (is_nonpositive_int(b) and (c).real() <= (b).real())))

DEFINE_COMPLEX_FUNCTION(hyp2f1, (a, b, c, z)) {
    if (z == ltrl(1)) {
        return (((c-a-b).real() > 0 or is_nonpositive_int(a) or is_nonpositive_int(b)) and not hyp2f1_zerodiv(a, b, c))
        // TODO: gammaprod
        ? gamma<scalar_t>(c) * gamma<scalar_t>(c-a-b) / gamma<scalar_t>(c-a) / gamma<scalar_t>(c-b)
        : numeric_limits<T>::infinity();
    }
    if (z == ltrl(0)) return (c != ltrl(0) || a == ltrl(0) || b == ltrl(0)) ? ltrl(1)+z : cnan<T>();

    if (hyp2f1_zerodiv(a, b, c))
        return numeric_limits<scalar_t>::infinity();

    if (abs(z) <= 0.8
        or (is_real_nonpositive(a) and is_int(a) and -1000 <= a.real() <= 0)
        or (is_real_nonpositive(b) and is_int(b) and -1000 <= b.real() <= 0))
        return hypsum<2, 1, scalar_t>(std::array<T, 3>{{a, b, c}}, z);
    if (abs(z) >= 1.3) {  // https://dlmf.nist.gov/15.8.E2
        for (auto i=0; i<10; ++i) {
            auto res = ltrl(M_PI) / sin(ltrl(M_PI) * (b-a)) * (
                // exp(LGAMMA(c) - LGAMMA(b) - LGAMMA(c-a) - LGAMMA(a-b+1))
                GAMMA(c) / GAMMA(b) / GAMMA(c-a) / GAMMA(a-b+1)
                    * pow(-z, -a) * hyp2f1<scalar_t>(a, a-c+1, a-b+1, ltrl(1)/z)
                // - exp(LGAMMA(c) - LGAMMA(a) - LGAMMA(c-b) - LGAMMA(b-a+1))
                - GAMMA(c) / GAMMA(a) / GAMMA(c-b) / GAMMA(b-a+1)
                    * pow(-z, -b) * hyp2f1<scalar_t>(b, b-c+1, b-a+1, ltrl(1)/z)
            );
            if (isfinite(res)) return res;
            a += pow(10, i) * PERTURBATION;
            b += 1.2 * pow(10, i) * PERTURBATION;
        }
        printf("possible gamma overflow...\n");
        return cnan<T>();
    }
    if (abs(ltrl(1)-z) <= 0.75) { // https://dlmf.nist.gov/15.8.E4
        for (auto i=0; i<10; ++i) {
            auto res = ltrl(M_PI) / sin(ltrl(M_PI) * (c-a-b)) * (
                // exp(LGAMMA(c) - LGAMMA(c-a) - LGAMMA(c-b) - LGAMMA(a+b-c+1))
                GAMMA(c) / GAMMA(c-a) / GAMMA(c-b) / GAMMA(a+b-c+1)
                    * hyp2f1<scalar_t>(a, b, a+b-c+1, 1-z)
                // - exp(LGAMMA(c) - LGAMMA(a) - LGAMMA(b) - LGAMMA(c-a-b+1))
                - GAMMA(c) / GAMMA(a) / GAMMA(b) / GAMMA(c-a-b+1)
                    * pow(1-z, c-a-b) * hyp2f1<scalar_t>(c-a, c-b, c-a-b+1, 1-z)
            );
            if (isfinite(res)) return res;
            a += pow(10, i) * PERTURBATION;
            b += 1.2 * pow(10, i) * PERTURBATION;
        }
        printf("possible gamma overflow...\n");
        return cnan<T>();
    }
    if (abs(z/(z-ltrl(1))) <= 0.75) {  // https://dlmf.nist.gov/15.8.E1
        return hyp2f1<scalar_t>(a, c-b, c, z/(z-ltrl(1))) / pow(ltrl(1)-z, a);
    }
    return hyp2f1_gosper<scalar_t>(a, b, c, z);
}
