#include "../common/implement.h"
#include "hyper.h"


TORCH_IMPLEMENT(gamma, (z))
TORCH_IMPLEMENT(loggamma, (z))
TORCH_IMPLEMENT(digamma, (z))

void polygamma_impl(at::TensorIteratorBase& iter, const unsigned long& n);
auto torch_polygamma(const unsigned long& n, const torch::Tensor& z) {
    return implement<1>([&](at::TensorIteratorBase& iter) {return polygamma_impl(iter, n);}, {z});
}

TORCH_IMPLEMENT(gammainc, (a, x))
TORCH_IMPLEMENT(gammaincc, (a, x))
TORCH_IMPLEMENT(gammaincinv, (a, p))
TORCH_IMPLEMENT(gammainccinv, (a, q))

// TORCH_IMPLEMENT(hyp2f1, (a, b, c, z))
void hyp0f1_impl(at::TensorIteratorBase&, HYP_KWARGS_TYPES);
auto torch_hyp0f1(const torch::Tensor& b, const torch::Tensor& z, HYP_KWARGS_H) {
    return implement<2>([&](at::TensorIteratorBase& iter) {return hyp0f1_impl(iter, HYP_KWARGS_NAMES);}, {b, z});
}

void Tgamma_impl(const size_t&, at::TensorIteratorBase&);
auto torch_Tgamma(size_t m, const torch::Tensor& a, const torch::Tensor& z) {
    return implement<2>([&](at::TensorIteratorBase& iter) {return Tgamma_impl(m, iter);}, {a, z});
}


TORCH_IMPLEMENT(deta1, (z))
TORCH_IMPLEMENT(zeta, (z))


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    MDEF(gamma) MDEF(loggamma) MDEF(digamma) MDEF(polygamma)
    MDEF(gammainc) MDEF(gammaincc) MDEF(gammaincinv) MDEF(gammainccinv)
    m.def("hyp0f1", &torch_hyp0f1,
          py::arg("b"), py::arg("z"),
          py::arg(STRINGIFY(HYP_KWARGS_FORCE_NAME)) = HYP_KWARGS_FORCE_VALUE,
          py::arg(STRINGIFY(HYP_KWARGS_MAXTERMS_NAME)) = HYP_KWARGS_MAXTERMS_VALUE
    );
    MDEF(Tgamma)
    // MDEF(hyp2f1)
    MDEF(deta1) MDEF(zeta)
}
