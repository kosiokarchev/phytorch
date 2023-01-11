#include "../common/implement_cpu.h"

#include "roots.h"


void roots_impl_cpu(at::TensorIteratorBase& iter, size_t nout) {
    AT_DISPATCH_FLOATING_TYPES(toRealValueType(iter.common_dtype()), "roots", [&] {
        using T = complex<scalar_t>;

        switch (nout) {
            case 2: at::native::cpu_kernel_multiple_outputs(iter, roots2<scalar_t>); break;
            case 3: at::native::cpu_kernel_multiple_outputs(iter, roots3<scalar_t>); break;
            case 4: at::native::cpu_kernel_multiple_outputs(iter, roots4<scalar_t>); break;
            default: TORCH_CHECK(false, "\"roots\" only implemented for orders 2, 3, 4.")
        }
    });
}
