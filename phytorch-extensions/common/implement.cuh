#pragma once


#include <ATen/TensorIterator.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cpu/Loops.h>

#include "preprocessor.h"


#define stringify(a) #a

#define IMPLEMENT_VERBOSE(name, signature, varnames, return_type)                  \
void name##_impl(at::TensorIteratorBase& iter) {                                   \
    TORCH_CHECK(iter.device(0).is_cpu() || iter.device(0).is_cuda(),               \
                stringify(#name) " only implemented on CPU and cuda.")             \
    AT_DISPATCH_FLOATING_TYPES(toValueType(iter.common_dtype()), #name, [&] {      \
        using T = complex<scalar_t>;                                               \
        if (iter.device_type(0) == c10::DeviceType::CPU)                           \
            at::native::cpu_kernel(iter, name<scalar_t, T>);                       \
        else at::native::gpu_kernel(iter, []GPU_LAMBDA signature -> return_type {  \
            return name<scalar_t> varnames;});});}

#define IMPLEMENT(name, vars, return_type) IMPLEMENT_VERBOSE(name, (VARS_TO_SIGNATURE(vars)), (VARS_TO_CALL(vars)), return_type)

#define IMPLEMENT_COMPLEX(name, vars) IMPLEMENT(name, (BOOST_PP_SEQ_FOR_EACH_I(PREPEND_data, T, BOOST_PP_TUPLE_TO_SEQ(vars))), T)
