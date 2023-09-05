#include "../common/implement.h"

// Dependencies:
//   gammaincc?inv -> gammaincc?, igam_fac
//   Tgamma -> meijerg -> hypercomb -> gamma
// Clashes with torch namespace:
//   lanczos_sum_expg_scaled


TORCH_IMPLEMENT(gamma, (z))
TORCH_IMPLEMENT(loggamma, (z))
TORCH_IMPLEMENT(digamma, (z))

// // TODO: polygamma

TORCH_IMPLEMENT(gammainc, (a, x))
TORCH_IMPLEMENT(gammaincc, (a, x))
TORCH_IMPLEMENT(gammaincinv, (a, p))
TORCH_IMPLEMENT(gammainccinv, (a, q))


TORCH_IMPLEMENTATION(Tgamma, size_t)
TORCH_DEFINE_IMPLEMENTATION(Tgamma, (size_t m, TENSORS_TO_SIGNATURE((a, z)))) {
    return implement<2, size_t>(Tgamma_impls, {a, z}, m);
}


TORCH_IMPLEMENT(hyp0f1, (b, z))
TORCH_IMPLEMENT(hyp1f0, (a, z))
TORCH_IMPLEMENT(hyp1f1, (a, b, z))


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    MDEF(gamma) MDEF(loggamma) MDEF(digamma)
    MDEF(gammainc) MDEF(gammaincc)
    MDEF(gammaincinv) MDEF(gammainccinv)

    MDEF(Tgamma)

    MDEF(hyp0f1) MDEF(hyp1f0)  MDEF(hyp1f1)
}
