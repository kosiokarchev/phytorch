#include "thrust/tuple.h"

#include "ATen/Dispatch.h"
#include "ATen/native/cuda/Loops.cuh"
#include "../common/Loops.h"

#include "../common/complex.h" // NOLINT(modernize-deprecated-headers)


#define ROOTS_TEMPLATE(N) template <typename scalar_t, typename T=complex<scalar_t>> __host__ __device__ std::array<T, N>
#define DEF_ROOTS(N) ROOTS_TEMPLATE(N) roots##N


DEF_ROOTS(2)(T b, T c) {
    if (not (isfinite(b) and isfinite(c)))
        return {numeric_limits<T>::quiet_NaN(),
                numeric_limits<T>::quiet_NaN()};
    if (c == ltrl(0)) return {ltrl(0), -b};
    auto q = sqrt(b*b - 4*c);
    q = -(b.real()>=0 ? (b+q) : (b-q)) / 2;
    return {q, c/q};
}


DEF_ROOTS(3)(T b, T c, T d) {
    if (not (isfinite(b) and isfinite(c) and isfinite(d)))
        return {numeric_limits<T>::quiet_NaN(),
                numeric_limits<T>::quiet_NaN(),
                numeric_limits<T>::quiet_NaN()};

    auto Q = (b*b - 3*c) / 9, R = (2*b*b*b - 9*b*c + 27*d) / 54;
    auto A = sqrt((R*R - Q*Q*Q));
    A = - pow((R.real() >= 0 ? R+A : R-A), ltrl(1./3.));
    auto B = A == ltrl(0) ? 0 : Q/A;

    auto AB = A+B, Re = -AB/2 - b/3, Im = ltrl(sqrt(3) / 2) * (A-B);
    return {AB - b/3, Re + T(0, 1)*Im, Re - T(0, 1)*Im};

    // auto D0 = b*b - 3*c,
    //      D1 = ltrl(2)*b*b*b - ltrl(9)*b*c + ltrl(27)*d;
    // if (D0 == ltrl(0) and D1 == ltrl(0))
    //     return {-b/ltrl(3), -b/ltrl(3), -b/ltrl(3)};
    //
    // auto D2 = sqrt(D1*D1 - ltrl(4)*D0*D0*D0),
    //      C = D1 + D2;
    // if (C == ltrl(0)) C = D1 - D2;
    // C = pow(C / ltrl(2), ltrl((1./3.)));
    // auto cr1 = T(-0.5, -sqrt(3)/2);
    // return {-(b + C + D0/C) / ltrl(3),
    //         -(b + C*cr1 + D0/(C*cr1)) / ltrl(3),
    //         -(b + C/cr1 + D0/(C/cr1)) / ltrl(3)};
}


ROOTS_TEMPLATE(4) roots4_depressed(T p, T q, T r) {
    if (q == ltrl(0)) {
        auto r2 = roots2<scalar_t>(p, r);
        auto s1 = sqrt(r2[0]), s2 = sqrt(r2[1]);
        return {s1, -s1, s2, -s2};
    }

    auto r3 = roots3<scalar_t>(2*p, p*p-4*r, -q*q);
    auto s1 = sqrt(r3[1]), s2 = sqrt(r3[2]), s3 = -q / (s1*s2);
    return {(s1+s2+s3) / 2, (s1-s2-s3) / 2, (s2-s1-s3) / 2, (s3-s1-s2) / 2};
}


DEF_ROOTS(4)(T b, T c, T d, T e) {
    if (not (isfinite(b) and isfinite(c) and isfinite(d) and isfinite(e)))
        return {numeric_limits<T>::quiet_NaN(),
                numeric_limits<T>::quiet_NaN(),
                numeric_limits<T>::quiet_NaN(),
                numeric_limits<T>::quiet_NaN()};

    auto r4d = roots4_depressed<scalar_t>(
        (8*c - 3*b*b) / 8,
        (b*b*b - 4*b*c + 8*d) / 8,
        (-3*b*b*b*b + 256*e - 64*b*d + 16*b*b*c) / 256
    );

    std::array<T, 4> ret = {r4d[0] - b/4, r4d[1] - b/4, r4d[2] - b/4, r4d[3] - b/4};
    std::sort(ret.begin(), ret.end(), [] (const T& _a, const T& _b) {return abs(_a) < abs(_b);});

    return ret;

    // auto twop = (ltrl(8)*c - ltrl(3)*b*b) / ltrl(4),
    //      q = (b*b*b - ltrl(4)*b*c + ltrl(8)*d) / ltrl(8),
    //      D0 = c*c - ltrl(3)*b*d + ltrl(12)*e,
    //      D1 = ltrl(2)*c*c*c - ltrl(9)*b*c*d + ltrl(27)*b*b*e + ltrl(27)*d*d - ltrl(72)*c*e;
    // T Q;
    // if (D0 == ltrl(0)) {
    //     if (D1 == ltrl(0)) {
    //         if (q == ltrl(0)) return {ltrl(0), ltrl(0), ltrl(0), ltrl(0)};
    //         auto x0 = (-ltrl(72)*e + ltrl(10)*c*c - ltrl(3)*b*b*c) / ltrl(9) / q,
    //              x1 = -b - ltrl(3)*x0;
    //         return {x0, x0, x0, x1};
    //     } else Q = pow(D1, ltrl(1./3.));
    // } else Q = pow(D1 + sqrt(D1*D1 - ltrl(4) * D0*D0*D0), ltrl(1./3.)) / cbrt(ltrl(2));
    // auto _s2 = ((Q + D0 / Q) - twop) / ltrl(12);
    // T s2;
    // if (_s2 == ltrl(0)) {
    //     Q *= T(-0.5, -sqrt(3)/2);
    //     s2 = ((Q + D0 / Q) - twop) / ltrl(12);
    // } else s2 = _s2;
    // auto S = sqrt(s2),
    //      pmp = sqrt(ltrl(-4)*s2 - twop + q/S) / ltrl(2),
    //      pmm = sqrt(ltrl(-4)*s2 - twop - q/S) / ltrl(2),
    //      mb4apS = -b / ltrl(4) + S,
    //      mb4amS = -b / ltrl(4) - S;
    // return {mb4amS - pmp, mb4amS + pmp, mb4apS - pmm, mb4apS + pmm};
}


#define ROOTS_IMPL(N, VARSPEC, VARNAMES, ...) \
    if (iter.device(0).is_cpu())              \
        cpu_kernel_multiple_outputs(iter, roots##N<scalar_t>); \
    else at::native::gpu_kernel_multiple_outputs(                     \
        iter, []GPU_LAMBDA VARSPEC -> thrust::tuple<__VA_ARGS__> {      \
            return c10::guts::apply(thrust::make_tuple<__VA_ARGS__>, roots##N<scalar_t>VARNAMES);});


template <int n> void roots_impl(at::TensorIteratorBase& iter) {
    TORCH_CHECK(iter.device(0).is_cpu() or iter.device(0).is_cuda(),
                "\"roots4_kernel_cuda\" only implemented on CPU and cuda.")

    AT_DISPATCH_FLOATING_TYPES(toRealValueType(iter.common_dtype()), "roots", [&] {
        using T = complex<scalar_t>;

        // TODO: Why doesn't if constexpr work with more than one ROOTS_IMPl?!...
        switch (n) {
            case 2: ROOTS_IMPL(2, (T b, T c), (b, c), T, T); break;
            case 3: ROOTS_IMPL(3, (T b, T c, T d), (b, c, d), T, T, T); break;
            case 4: ROOTS_IMPL(4, (T b, T c, T d, T e), (b, c, d, e), T, T, T, T); break;
            default: TORCH_CHECK(false, "\"roots\" only implemented for orders 2, 3, 4.")
        }
    });
}

template void roots_impl<2>(at::TensorIteratorBase& iter);
template void roots_impl<3>(at::TensorIteratorBase& iter);
template void roots_impl<4>(at::TensorIteratorBase& iter);
