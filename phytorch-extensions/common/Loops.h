#pragma once

#include "ATen/native/cpu/Loops.h"

template <typename func_t, int n, typename scalar_t>
inline void cpu_kernel_multiple_outputs_impl(at::TensorIteratorBase& iter, func_t&& op) {
    using traits = function_traits<func_t>;
    iter.for_each([&] (char * C10_RESTRICT data[], const int64_t * strides, int64_t size) {
        for (auto k=0; k<size; ++k) {
            auto output = c10::guts::apply(
                    std::forward<func_t>(op),
                    at::native::dereference<traits>(&data[n], &strides[n], k));

            #pragma unroll
            for (auto i=0; i<n; ++i)
                *reinterpret_cast<scalar_t *>(data[i] + strides[i]*k) = output[i];
        }
    });
}

template <typename func_t> void cpu_kernel_multiple_outputs(at::TensorIteratorBase& iter, func_t&& op) {
    using traits = function_traits<func_t>;
    cpu_kernel_multiple_outputs_impl<
            func_t, std::tuple_size<typename traits::result_type>::value, typename traits::result_type::value_type
    >(iter, std::forward<func_t>(op));
}
