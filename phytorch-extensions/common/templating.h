#pragma once

#include "preprocessor.h"

#define COMPLEX_TYPE_LIST(z, n, dtype) BOOST_PP_COMMA_IF(n) complex<dtype>
#define INSTANTIATE_COMPLEX_TEMPLATE_NOARGS(f, dtype) template __host__ __device__ complex<dtype> f<dtype, complex<dtype>>
#define INSTANTIATE_COMPLEX_TEMPLATE_ONE(f, n, dtype) INSTANTIATE_COMPLEX_TEMPLATE_NOARGS(f, dtype)(BOOST_PP_REPEAT(n, COMPLEX_TYPE_LIST, dtype));
#define INSTANTIATE_COMPLEX_TEMPLATE(f, n) INSTANTIATE_COMPLEX_TEMPLATE_ONE(f, n, float) INSTANTIATE_COMPLEX_TEMPLATE_ONE(f, n, double)


#define COMPLEX_TEMPLATE_H template <typename scalar_t, typename T> __host__ __device__
#define COMPLEX_TEMPLATE template <typename scalar_t, typename T=complex<scalar_t>> __host__ __device__

#define DCF_GET_ARG(n, i, names) BOOST_PP_COMMA_IF(i) T BOOST_PP_TUPLE_ELEM(i, names)
#define DEFINE_COMPLEX_FUNCTION(f, vars) \
INSTANTIATE_COMPLEX_TEMPLATE(f, BOOST_PP_TUPLE_SIZE(vars)) \
COMPLEX_TEMPLATE T f(BOOST_PP_REPEAT(BOOST_PP_TUPLE_SIZE(vars), DCF_GET_ARG, vars))

#define DEFINE_COMPLEX_FUNCTION_H(f, vars) \
COMPLEX_TEMPLATE_H T f(BOOST_PP_REPEAT(BOOST_PP_TUPLE_SIZE(vars), DCF_GET_ARG, vars));
