#include "../common/implement.cuh"
#include "special.cuh"


IMPLEMENT(gamma, (T z), (z), T)
IMPLEMENT(loggamma, (T z), (z), T)
IMPLEMENT(digamma, (T z), (z), T)

// TODO: generalise fancy implementations
void polygamma_impl(at::TensorIteratorBase& iter, const unsigned long& n) {
    TORCH_CHECK(iter.device(0).is_cpu() || iter.device(0).is_cuda(),
                stringify(#NAME) " only implemented on CPU and cuda.")
    AT_DISPATCH_FLOATING_TYPES(toValueType(iter.common_dtype()), "polygamma", [&] {
        using T = complex<scalar_t>;
        if (iter.device_type(0) == c10::DeviceType::CPU)
            at::native::cpu_kernel(iter, [n](T z) -> T {return polygamma<scalar_t, T>(n, z);});
        else at::native::gpu_kernel(iter, [n]GPU_LAMBDA (T z) -> T {return polygamma<scalar_t>(n, z);});});
}

IMPLEMENT(hyp2f1, (T a, T b, T c, T z), (a, b, c, z), T)

IMPLEMENT(deta1, (T z), (z), T)
IMPLEMENT(zeta, (T z), (z), T)
