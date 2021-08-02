#pragma once


#include <ATen/TensorIterator.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cpu/Loops.h>


#define stringify(a) #a

#define IMPLEMENT(NAME, SIGNATURE, VARNAMES, RETURN_TYPE)                          \
void NAME##_impl(at::TensorIteratorBase& iter) {                                   \
    TORCH_CHECK(iter.device(0).is_cpu() || iter.device(0).is_cuda(),               \
                stringify(#NAME) " only implemented on CPU and cuda.")             \
    AT_DISPATCH_FLOATING_TYPES(toValueType(iter.common_dtype()), #NAME, [&] {      \
        using T = complex<scalar_t>;                                               \
        if (iter.device_type(0) == c10::DeviceType::CPU)                           \
            at::native::cpu_kernel(iter, NAME<scalar_t, T>);                       \
        else at::native::gpu_kernel(iter, []GPU_LAMBDA SIGNATURE -> RETURN_TYPE {  \
            return NAME<scalar_t> VARNAMES;});});}
