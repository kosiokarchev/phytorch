#include "../common/implement.cuh"
#include "hyper.cuh"


void hyp0f1_impl(at::TensorIteratorBase& iter, HYP_KWARGS) {
    TORCH_CHECK(iter.device(0).is_cpu() || iter.device(0).is_cuda(),
                "hyp0f1 only implemented on CPU and cuda.")
    AT_DISPATCH_FLOATING_TYPES(toRealValueType(iter.common_dtype()), "hyp0f1", [&] {
        using T = complex<scalar_t>;
        if (iter.device_type(0) == c10::DeviceType::CPU)
            at::native::cpu_kernel(iter, [&](T b, T z) -> T {return hyp0f1<scalar_t, T>(b, z, HYP_KWARGS_NAMES);});
        else at::native::gpu_kernel(iter, [=]GPU_LAMBDA (T b, T z) -> T {return hyp0f1<scalar_t>(b, z, HYP_KWARGS_NAMES);});});
}