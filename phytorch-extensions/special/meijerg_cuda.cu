#include "../common/implement_cuda.cuh"
#include "meijerg.h"

void Tgamma_impl_cuda(at::TensorIteratorBase& iter, size_t m) {
    AT_DISPATCH_FLOATING_TYPES(toRealValueType(iter.common_dtype()), "Tgamma", [&] {
        using T = complex<scalar_t>;
        at::native::gpu_kernel(iter, [=]GPU_LAMBDA (T a, T z) -> T {return Tgamma<scalar_t>(m, a, z);});});
}
