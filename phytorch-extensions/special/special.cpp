#include "../common/implement.h"


#define TORCH_IMPLEMENT_1(f)                 \
void f##_impl(at::TensorIteratorBase& iter); \
auto torch_##f(const torch::Tensor& z) {     \
    return implement<1>(f##_impl, {z});}


#define MDEF(f) m.def(#f, &torch_##f, #f);


TORCH_IMPLEMENT_1(gamma)
TORCH_IMPLEMENT_1(loggamma)
TORCH_IMPLEMENT_1(digamma)

void polygamma_impl(at::TensorIteratorBase& iter, const unsigned long& n);
auto torch_polygamma(const unsigned long& n, const torch::Tensor& z) {
    return implement<1>([=](at::TensorIteratorBase& iter) {return polygamma_impl(iter, n);}, {z});
}

// TODO: macro-ify
void hyp2f1_impl(at::TensorIteratorBase& iter);
auto torch_hyp2f1(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& c, const torch::Tensor& z) {
    return implement<4>(hyp2f1_impl, {a, b, c, z});
}

TORCH_IMPLEMENT_1(deta1)
TORCH_IMPLEMENT_1(zeta)


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    MDEF(gamma) MDEF(loggamma) MDEF(digamma) MDEF(polygamma)
    MDEF(hyp2f1)
    MDEF(deta1) MDEF(zeta)
}
