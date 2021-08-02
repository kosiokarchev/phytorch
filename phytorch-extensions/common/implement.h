#pragma once

#include "torch/extension.h"
#include <ATen/TensorIterator.h>


template <int n>
// auto implement(void (*func)(at::TensorIteratorBase&), std::array<torch::Tensor, n> inputs) {
auto implement(std::function<void(at::TensorIteratorBase&)> func, std::array<torch::Tensor, n> inputs) {
    at::TensorIteratorConfig iterconfig;
    iterconfig.check_all_same_device(true);
    iterconfig.promote_inputs_to_common_dtype(true).promote_integer_inputs_to_float(true);
    iterconfig.add_output(torch::Tensor());
    #pragma unroll
    for (auto i=0; i<n; ++i) iterconfig.add_input(inputs[i]);

    auto iter = iterconfig.build();
    func(iter);
    return iter.output(0);
}
