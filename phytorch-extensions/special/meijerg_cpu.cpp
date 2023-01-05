#include "../common/implement_cpu.h"
#include "meijerg.h"

void Tgamma_impl_cpu(at::TensorIteratorBase& iter, size_t m) {
    AT_DISPATCH_FLOATING_TYPES(toRealValueType(iter.common_dtype()), "Tgamma", [&] {
        using T = complex<scalar_t>;
        at::native::cpu_kernel(iter, [&](T a, T z) {return Tgamma<scalar_t>(m, a, z);});}
    );
}
