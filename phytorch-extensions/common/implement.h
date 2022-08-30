#pragma once

#include "torch/extension.h"
#include "ATen/TensorIterator.h"

#include "preprocessor.h"


template <int n>
auto implement(const std::function<void(at::TensorIteratorBase&)>& func, std::array<torch::Tensor, n> inputs) {
    at::TensorIteratorConfig iterconfig;
    iterconfig.check_all_same_device(true);
    iterconfig.promote_inputs_to_common_dtype(true).promote_integer_inputs_to_float(true);
    iterconfig.add_owned_output(torch::Tensor());

    for (auto i=0; i<n; ++i) iterconfig.add_input(inputs[i]);

    auto iter = iterconfig.build();
    func(iter);
    return iter.output(0);
}

#define TORCH_IMPLEMENT(name, vars) \
void name##_impl(at::TensorIteratorBase& iter); \
auto torch_##name(VARS_TO_SIGNATURE((BOOST_PP_SEQ_FOR_EACH_I(PREPEND_data, const torch::Tensor&, BOOST_PP_TUPLE_TO_SEQ(vars)))))  { \
    return implement<BOOST_PP_TUPLE_SIZE(vars)>(name##_impl, {BOOST_PP_TUPLE_ENUM(vars)});}

#define MDEF(name) m.def(#name, &torch_##name, #name);
