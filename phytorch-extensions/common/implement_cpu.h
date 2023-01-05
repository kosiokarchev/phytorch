#pragma once

#include "ATen/Dispatch.h"
#include "ATen/TensorIterator.h"
#include "ATen/native/cpu/Loops.h"


#define IMPLEMENT_CPU(name) \
void name##_impl_cpu(at::TensorIteratorBase& iter) { \
    AT_DISPATCH_FLOATING_TYPES(toRealValueType(iter.common_dtype()), #name, [&] { \
        using T = complex<scalar_t>;                                              \
        at::native::cpu_kernel(iter, name<scalar_t>);});}
