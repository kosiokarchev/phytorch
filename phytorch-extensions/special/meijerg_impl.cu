#include "../common/implement.cuh"
#include "meijerg.cuh"


void Tgamma_impl(const size_t& m, at::TensorIteratorBase& iter) {
    TORCH_CHECK(iter.device(0).is_cpu() || iter.device(0).is_cuda(),
                "Tgamma only implemented on CPU and cuda.")
    AT_DISPATCH_FLOATING_TYPES(toRealValueType(iter.common_dtype()), "Tgamma", [&] {
        using T = complex<scalar_t>;
        if (iter.device_type(0) == c10::DeviceType::CPU)
            at::native::cpu_kernel(iter, [&](T a, T z) -> T {return Tgamma<scalar_t>(m, a, z);});
        else at::native::gpu_kernel(iter, [=]GPU_LAMBDA (T a, T z) -> T {return Tgamma<scalar_t>(m, a, z);});});
}