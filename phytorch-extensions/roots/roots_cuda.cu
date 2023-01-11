#include "../common/implement_cuda.cuh"

#include "roots.h"


void roots_impl_cuda(at::TensorIteratorBase& iter, size_t nout) {
    AT_DISPATCH_FLOATING_TYPES(toRealValueType(iter.common_dtype()), "roots", [&] {
        using T = complex<scalar_t>;

        switch (nout) {
            case 2: at::native::gpu_kernel_multiple_outputs(iter, []GPU_LAMBDA (T b, T c) {
                auto ret = roots2<scalar_t>(b, c);
                return thrust::tuple<T, T>(std::get<0>(ret), std::get<1>(ret));
            }); break;
            case 3: at::native::gpu_kernel_multiple_outputs(iter, []GPU_LAMBDA (T b, T c, T d) {
                auto ret = roots3<scalar_t>(b, c, d);
                return thrust::tuple<T, T, T>(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret));
            }); break;
            case 4: at::native::gpu_kernel_multiple_outputs(iter, []GPU_LAMBDA (T b, T c, T d, T e) {
                auto ret = roots4<scalar_t>(b, c, d, e);
                return thrust::tuple<T, T, T, T>(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret));
            }); break;
            default: TORCH_CHECK(false, "\"roots\" only implemented for orders 2, 3, 4.")
        }
    });
}