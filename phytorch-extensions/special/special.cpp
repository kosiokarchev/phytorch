#include "../common/implement.h"


TORCH_IMPLEMENT(gamma, (z))
TORCH_IMPLEMENT(loggamma, (z))
TORCH_IMPLEMENT(digamma, (z))

void polygamma_impl(at::TensorIteratorBase& iter, const unsigned long& n);
auto torch_polygamma(const unsigned long& n, const torch::Tensor& z) {
    return implement<1>([=](at::TensorIteratorBase& iter) {return polygamma_impl(iter, n);}, {z});
}

TORCH_IMPLEMENT(hyp2f1, (a, b, c, z))

TORCH_IMPLEMENT(deta1, (z))
TORCH_IMPLEMENT(zeta, (z))


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    MDEF(gamma) MDEF(loggamma) MDEF(digamma) MDEF(polygamma)
    MDEF(hyp2f1)
    MDEF(deta1) MDEF(zeta)
}
