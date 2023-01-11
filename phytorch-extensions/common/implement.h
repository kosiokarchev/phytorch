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

template <size_t nin, size_t nout, typename... argtypes>
auto implement_(const implmap<argtypes...>& impls, const std::array<torch::Tensor, nin>& inputs, argtypes... args) {
    at::TensorIteratorConfig iterconfig;
    iterconfig.check_all_same_device(true);
    iterconfig.promote_inputs_to_common_dtype(true).promote_integer_inputs_to_float(true);

    // TODO: transform loops to index_sequence expansions
    for (size_t i=0; i<nout; ++i) iterconfig.add_owned_output(torch::Tensor());
    for (size_t i=0; i<nin; ++i) iterconfig.add_input(inputs[i]);

    auto iter = iterconfig.build();
    // TODO: give name of function in device check
    TORCH_CHECK(DEVICE_CHECK, "function only implemented on " AVAILABLE_DEVICES ".")
    impls.at(iter.device_type(0))(iter, args...);

    return iter;
}

template <size_t nin, typename... argtypes>
auto implement(const implmap<argtypes...>& impls, const std::array<torch::Tensor, nin>& inputs, argtypes... args) {
    return implement_<nin, 1, argtypes...>(impls, inputs, args...).output(0);
}

template <size_t nin, size_t nout, typename... argtypes, size_t... iout>
auto implement_multiple_outputs_impl(
        const implmap<argtypes...>& impls,
        const std::array<torch::Tensor, nin>& inputs,
        std::index_sequence<iout...>,
        argtypes... args) {
    auto iter = implement_<nin, nout, argtypes...>(impls, inputs, args...);
    return std::make_tuple(iter.output(iout)...);
}

template <size_t nin, size_t nout, typename... argtypes>
auto implement_multiple_outputs(const implmap<argtypes...>& impls, const std::array<torch::Tensor, nin>& inputs, argtypes... args) {
    return implement_multiple_outputs_impl<nin, nout, argtypes...>(impls, inputs, std::make_index_sequence<nout>{}, args...);
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

#define TORCH_IMPLEMENT_MULTIPLE_OUTPUTS(name, vars, nout) \
TORCH_DEFINE_IMPLEMENTATION(name, (TENSORS_TO_SIGNATURE(vars))) { \
    return implement_multiple_outputs<BOOST_PP_TUPLE_SIZE(vars), nout, size_t>(name##_impls, {BOOST_PP_TUPLE_ENUM(vars)}, nout);}

#define MDEF(name) m.def(#name, &torch_##name, #name);
