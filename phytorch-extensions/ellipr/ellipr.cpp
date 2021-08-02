#include "../common/implement.h"


void elliprf_impl(at::TensorIteratorBase& iter);
auto torch_elliprf(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z)  {
    return implement<3>(elliprf_impl, {x, y, z});
}

void elliprc_impl(at::TensorIteratorBase& iter);
auto torch_elliprc(const torch::Tensor& x, const torch::Tensor& y)  {
    return implement<2>(elliprc_impl, {x, y});
}

void elliprj_impl(at::TensorIteratorBase& iter);
auto torch_elliprj(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z, const torch::Tensor& p)  {
    return implement<4>(elliprj_impl, {x, y, z, p});
}

void elliprd_impl(at::TensorIteratorBase& iter);
auto torch_elliprd(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z)  {
    return implement<3>(elliprd_impl, {x, y, z});
}

void elliprg_impl(at::TensorIteratorBase& iter);
auto torch_elliprg(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z)  {
    return implement<3>(elliprg_impl, {x, y, z});
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elliprf", &torch_elliprf, "elliprf");
    m.def("elliprc", &torch_elliprc, "elliprc");
    m.def("elliprj", &torch_elliprj, "elliprj");
    m.def("elliprd", &torch_elliprd, "elliprd");
    m.def("elliprg", &torch_elliprg, "elliprg");
}
