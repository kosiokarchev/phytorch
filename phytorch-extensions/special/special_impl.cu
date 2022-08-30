#include "../common/implement.cuh"
#include "special.cuh"


IMPLEMENT_COMPLEX(gamma, (z))
IMPLEMENT_COMPLEX(loggamma, (z))
IMPLEMENT_COMPLEX(digamma, (z))

// TODO: generalise fancy implementations
void polygamma_impl(at::TensorIteratorBase& iter, const unsigned long& n) {
    TORCH_CHECK(iter.device(0).is_cpu() || iter.device(0).is_cuda(),
                "polygamma only implemented on CPU and cuda.")
    AT_DISPATCH_FLOATING_TYPES(toRealValueType(iter.common_dtype()), "polygamma", [&] {
        using T = complex<scalar_t>;
        if (iter.device_type(0) == c10::DeviceType::CPU)
            at::native::cpu_kernel(iter, [&](T z) -> T {return polygamma<scalar_t, T>(n, z);});
        else at::native::gpu_kernel(iter, [=]GPU_LAMBDA (T z) -> T {return polygamma<scalar_t>(n, z);});});
}

IMPLEMENT_REAL(gammainc, (a, x))
IMPLEMENT_REAL(gammaincc, (a, x))
IMPLEMENT_REAL(gammaincinv, (a, p))
IMPLEMENT_REAL(gammainccinv, (a, q))

// IMPLEMENT_COMPLEX(hyp2f1, (a, b, c, z))

IMPLEMENT_COMPLEX(deta1, (z))
IMPLEMENT_COMPLEX(zeta, (z))
