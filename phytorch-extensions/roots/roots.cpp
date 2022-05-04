#include "torch/extension.h"
#include "ATen/TensorIterator.h"

#include "../common/utils.h"


template <int n> void roots_impl(at::TensorIteratorBase& iter);

template <int n>
auto torch_roots(const std::array<torch::Tensor, n>& inputs) {
    at::TensorIteratorConfig iterconfig;
    iterconfig.check_all_same_device(true);
    iterconfig.promote_inputs_to_common_dtype(true).promote_integer_inputs_to_float(true);
    #pragma unroll
    for (auto i=0; i<n; ++i) iterconfig.add_owned_output(torch::Tensor());
    for (auto i=0; i<n; ++i) iterconfig.add_input(inputs[i]);

    auto iter = iterconfig.build();
    roots_impl<n>(iter);

    std::array<torch::Tensor, n> ret;
    #pragma unroll
    for (auto i=0; i<n; ++i) ret[i] = iter.output(i);
    return array_to_tuple(ret, std::make_index_sequence<n>{});
}

typedef const torch::Tensor& cT;

auto torch_roots2(cT b, cT c) {return torch_roots<2>({b, c});}
auto torch_roots3(cT b, cT c, cT d) {return torch_roots<3>({b, c, d});}
auto torch_roots4(cT b, cT c, cT d, cT e) {return torch_roots<4>({b, c, d, e});}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("roots2", &torch_roots2, "roots2");
    m.def("roots3", &torch_roots3, "roots3");
    m.def("roots4", &torch_roots4, "roots4");
}
