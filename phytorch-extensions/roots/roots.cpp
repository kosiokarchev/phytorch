#include "../common/implement.h"


TORCH_IMPLEMENTATION(roots, size_t)

#define roots2_impls roots_impls
TORCH_IMPLEMENT_MULTIPLE_OUTPUTS(roots2, (b, c), 2)
#define roots3_impls roots_impls
TORCH_IMPLEMENT_MULTIPLE_OUTPUTS(roots3, (b, c, d), 3)
#define roots4_impls roots_impls
TORCH_IMPLEMENT_MULTIPLE_OUTPUTS(roots4, (b, c, d, e), 4)


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    MDEF(roots2) MDEF(roots3) MDEF(roots4)
}
