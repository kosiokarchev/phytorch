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
}


#define ROOTS_IMPL(N, VARSPEC, VARNAMES, ...) \
    if (iter.device(0).is_cpu())              \
        cpu_kernel_multiple_outputs(iter, roots##N<scalar_t>); \
    else at::native::gpu_kernel_multiple_outputs(                     \
        iter, []GPU_LAMBDA VARSPEC -> thrust::tuple<__VA_ARGS__> {      \
            return c10::guts::apply(thrust::make_tuple<__VA_ARGS__>, roots##N<scalar_t>VARNAMES);});


// TODO: roots on CUDA?!
template <int n> void roots_impl(at::TensorIteratorBase& iter) {
    TORCH_CHECK(iter.device(0).is_cpu(), "\"roots\" only implemented on CPU.")

    AT_DISPATCH_FLOATING_TYPES(toRealValueType(iter.common_dtype()), "roots", [&] {
        using T = complex<scalar_t>;

        if (n == 2) cpu_kernel_multiple_outputs(iter, roots2<scalar_t>); else
        if (n == 3) cpu_kernel_multiple_outputs(iter, roots3<scalar_t>); else
        if (n == 4) cpu_kernel_multiple_outputs(iter, roots4<scalar_t>);
        else TORCH_CHECK(false, "\"roots\" only implemented for orders 2, 3, 4.")
    });
}

template void roots_impl<2>(at::TensorIteratorBase& iter);
template void roots_impl<3>(at::TensorIteratorBase& iter);
template void roots_impl<4>(at::TensorIteratorBase& iter);
