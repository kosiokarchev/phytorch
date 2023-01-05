#pragma once

#include "ATen/Dispatch.h"
#include "ATen/TensorIterator.h"
#include "ATen/native/cuda/Loops.cuh"


#define IMPLEMENT_CUDA_VERBOSE(name, signature, varnames, return_type, _T) \
void name##_impl_cuda(at::TensorIteratorBase& iter) { \
    AT_DISPATCH_FLOATING_TYPES(toRealValueType(iter.common_dtype()), #name, [&] { \
        using T = _T; \
        at::native::gpu_kernel(iter, []GPU_LAMBDA signature -> return_type {  \
            return name<scalar_t> varnames;});});}

#define IMPLEMENT_CUDA(name, vars, return_type, _T) IMPLEMENT_CUDA_VERBOSE(name, (VARS_TO_SIGNATURE(vars)), (VARS_TO_CALL(vars)), return_type, _T)

#define IMPLEMENT_CUDA_FUNCTION(name, vars, _T) IMPLEMENT_CUDA(name, (BOOST_PP_SEQ_FOR_EACH_I(PREPEND_data, T, BOOST_PP_TUPLE_TO_SEQ(vars))), T, _T)

#define IMPLEMENT_CUDA_REAL(name, vars) IMPLEMENT_CUDA_FUNCTION(name, vars, scalar_t)
#define IMPLEMENT_CUDA_COMPLEX(name, vars) IMPLEMENT_CUDA_FUNCTION(name, vars, complex<scalar_t>)
