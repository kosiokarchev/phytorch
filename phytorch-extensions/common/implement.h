#pragma once

#ifdef PHYTORCH_CUDA
#define DEVICE_CHECK iter.device(0).is_cpu() || iter.device(0).is_cuda()
#define AVAILABLE_DEVICES "CPU and cuda"
#define IMPLMAP(name) {{c10::DeviceType::CPU, name##_impl_cpu}, {c10::DeviceType::CUDA, name##_impl_cuda}}
#else
#define DEVICE_CHECK iter.device(0).is_cpu()
#define AVAILABLE_DEVICES "CPU"
#define IMPLMAP(name) {{c10::DeviceType::CPU, name##_impl_cpu}}
#endif


#include <map>

#include "torch/extension.h"
#include "ATen/TensorIterator.h"

#include "preprocessor.h"


template <typename... argtypes>
using impl = std::function<void(at::TensorIteratorBase&, argtypes...)>;

template <typename... argtypes>
using implmap = std::map<c10::DeviceType, impl<argtypes...>>;

template <int n, typename... argtypes>
auto implement(const implmap<argtypes...>& impls, std::array<torch::Tensor, n> inputs, argtypes... args) {
    at::TensorIteratorConfig iterconfig;
    iterconfig.check_all_same_device(true);
    iterconfig.promote_inputs_to_common_dtype(true).promote_integer_inputs_to_float(true);
    iterconfig.add_owned_output(torch::Tensor());

    for (auto i=0; i<n; ++i) iterconfig.add_input(inputs[i]);

    auto iter = iterconfig.build();
    TORCH_CHECK(DEVICE_CHECK, stringify(elliprc) " only implemented on " AVAILABLE_DEVICES ".")
    impls.at(iter.device_type(0))(iter, args...);
    return iter.output(0);
}

#define TORCH_IMPLEMENTATION(name, ...) \
void name##_impl_cpu(at::TensorIteratorBase& iter __VA_OPT__(,) __VA_ARGS__);  \
void name##_impl_cuda(at::TensorIteratorBase& iter __VA_OPT__(,) __VA_ARGS__); \
static implmap<__VA_ARGS__> name##_impls = IMPLMAP(name);

#define TENSORS_TO_SIGNATURE(vars) VARS_TO_SIGNATURE((BOOST_PP_SEQ_FOR_EACH_I(PREPEND_data, const torch::Tensor&, BOOST_PP_TUPLE_TO_SEQ(vars))))
#define TORCH_DEFINE_IMPLEMENTATION(name, signature) auto torch_##name signature

#define TORCH_IMPLEMENT(name, vars) \
TORCH_IMPLEMENTATION(name)          \
TORCH_DEFINE_IMPLEMENTATION(name, (TENSORS_TO_SIGNATURE(vars))) { \
    return implement<BOOST_PP_TUPLE_SIZE(vars)>(name##_impls, {BOOST_PP_TUPLE_ENUM(vars)});}

#define MDEF(name) m.def(#name, &torch_##name, #name);
