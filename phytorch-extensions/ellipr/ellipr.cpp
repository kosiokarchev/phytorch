#include "torch/extension.h"
#include <ATen/TensorIterator.h>


template <int n>
auto implement(void (*func)(at::TensorIteratorBase&), std::array<torch::Tensor, n> inputs) {
    at::TensorIteratorConfig iterconfig;
    iterconfig.check_all_same_device(true);
    iterconfig.promote_inputs_to_common_dtype(true).promote_integer_inputs_to_float(true);
    iterconfig.add_output(torch::Tensor());
    for (auto i=0; i<n; ++i) iterconfig.add_input(inputs[i]);

    auto iter = iterconfig.build();
    func(iter);
    return iter.output(0);
}


void elliprf_impl(at::TensorIteratorBase& iter);
auto elliprf(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z)  {
    return implement<3>(elliprf_impl, {x, y, z});
}

void elliprc_impl(at::TensorIteratorBase& iter);
auto elliprc(const torch::Tensor& x, const torch::Tensor& y)  {
    return implement<2>(elliprc_impl, {x, y});
}

void elliprj_impl(at::TensorIteratorBase& iter);
auto elliprj(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z, const torch::Tensor& p)  {
    return implement<4>(elliprj_impl, {x, y, z, p});
}

void elliprd_impl(at::TensorIteratorBase& iter);
auto elliprd(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z)  {
    return implement<3>(elliprd_impl, {x, y, z});
}

void elliprg_impl(at::TensorIteratorBase& iter);
auto elliprg(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z)  {
    return implement<3>(elliprg_impl, {x, y, z});
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elliprf", &elliprf, "elliprf");
    m.def("elliprc", &elliprc, "elliprc");
    m.def("elliprj", &elliprj, "elliprj");
    m.def("elliprd", &elliprd, "elliprd");
    m.def("elliprg", &elliprg, "elliprg");
}

