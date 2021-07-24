#include <thrust/tuple.h>

#include <ATen/Dispatch.h>
#include "../common/Loops.h"
#include <ATen/native/cuda/Loops.cuh>

#include "../common/complex.h" // NOLINT(modernize-deprecated-headers)


#define EPS 1e-8

#define DEF_ROOTS(N) template <typename scalar_t, typename T=complex<scalar_t>> __host__ __device__ std::array<T, N> roots##N##_kernel


DEF_ROOTS(2)(T b, T c) {
    return {(-b - sqrt(b*b - ltrl(4)*c)) / ltrl(2),
            (-b + sqrt(b*b - ltrl(4)*c)) / ltrl(2)};
}


DEF_ROOTS(3)(T b, T c, T d) {
    auto D0 = b*b - 3*c,
         D1 = ltrl(2)*b*b*b - ltrl(9)*b*c + ltrl(27)*d,
         D2 = sqrt(D1*D1 - ltrl(4)*D0*D0*D0),
         C = D1 + D2;
    if (std::abs(C) < EPS) C = D1 + D2;
    C = pow(C / ltrl(2), ltrl((1./3.)));
    auto cr1 = T(-0.5, -sqrt(3)/2);
    return {-(b + C + D0/C) / ltrl(3),
            -(b + C*cr1 + D0/(C*cr1)) / ltrl(3),
            -(b + C/cr1 + D0/(C/cr1)) / ltrl(3)};
}


DEF_ROOTS(4)(T b, T c, T d, T e) {
    auto twop = (ltrl(8)*c - ltrl(3)*b*b) / ltrl(4),
         q = (b*b*b - ltrl(4)*b*c + ltrl(8)*d) / ltrl(8),
         D0 = c*c - ltrl(3)*b*d + ltrl(12)*e,
         D1 = ltrl(2)*c*c*c - ltrl(9)*b*c*d + ltrl(27)*b*b*e + ltrl(27)*d*d - ltrl(72)*c*e,
         Q = pow(D1 + sqrt(D1*D1 - ltrl(4) * D0*D0*D0), ltrl((1./3.))) / cbrt(ltrl(2)),
         _s2 = ((Q + D0 / Q) - twop) / ltrl(12);
    T s2;
    if (std::abs(_s2) < EPS) {
        Q *= T(-0.5, -sqrt(3)/2);
        s2 = ((Q + D0 / Q) - twop) / ltrl(12);
    } else s2 = _s2;
    auto S = sqrt(s2),
         pmp = sqrt(ltrl(-4)*s2 - twop + q/S) / ltrl(2),
         pmm = sqrt(ltrl(-4)*s2 - twop - q/S) / ltrl(2),
         mb4apS = -b / ltrl(4) + S,
         mb4amS = -b / ltrl(4) - S;
    return {mb4amS - pmp, mb4amS + pmp, mb4apS - pmm, mb4apS + pmm};
}


#define ROOTS_IMPL(N, VARSPEC, VARNAMES, ...) \
    if (iter.device(0).is_cpu())              \
        cpu_kernel_multiple_outputs(iter, roots##N##_kernel<scalar_t>); \
    else at::native::gpu_kernel_multiple_outputs(                     \
        iter, []GPU_LAMBDA VARSPEC -> thrust::tuple<__VA_ARGS__> {      \
            return c10::guts::apply(thrust::make_tuple<__VA_ARGS__>, roots##N##_kernel<scalar_t>VARNAMES);});


template <int n> void roots_impl(at::TensorIteratorBase& iter) {
    TORCH_CHECK(iter.device(0).is_cpu() or iter.device(0).is_cuda(),
                "\"roots4_kernel_cuda\" only implemented on CPU and cuda.")

    AT_DISPATCH_FLOATING_TYPES(toValueType(iter.common_dtype()), "roots", [&] {
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
