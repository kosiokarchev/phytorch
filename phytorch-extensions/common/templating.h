#pragma once

#include "preprocessor.h"
#include "complex.h" // NOLINT(modernize-deprecated-headers)

#define DEFINE_FUNCTION_GET_ARG(n, i, names) BOOST_PP_COMMA_IF(i) T BOOST_PP_TUPLE_ELEM(i, names)

// REAL:
#define REAL_TYPE_LIST(z, n, dtype) BOOST_PP_COMMA_IF(n) dtype
#define INSTANTIATE_REAL_TEMPLATE_NOARGS(f, dtype) template PHYTORCH_DEVICE dtype f<dtype>
#define INSTANTIATE_REAL_TEMPLATE_ONE(f, n, dtype) INSTANTIATE_REAL_TEMPLATE_NOARGS(f, dtype)(BOOST_PP_REPEAT(n, REAL_TYPE_LIST, dtype));
#define INSTANTIATE_REAL_TEMPLATE(f, n) INSTANTIATE_REAL_TEMPLATE_ONE(f, n, float) INSTANTIATE_REAL_TEMPLATE_ONE(f, n, double)

#define REAL_TEMPLATE_H template <typename scalar_t, typename T> PHYTORCH_DEVICE
#define REAL_TEMPLATE template <typename scalar_t, typename T=scalar_t> PHYTORCH_DEVICE

#define DEFINE_REAL_FUNCTION(f, vars) \
INSTANTIATE_REAL_TEMPLATE(f, BOOST_PP_TUPLE_SIZE(vars)) \
REAL_TEMPLATE T f(BOOST_PP_REPEAT(BOOST_PP_TUPLE_SIZE(vars), DEFINE_FUNCTION_GET_ARG, vars))

#define DEFINE_REAL_FUNCTION_H(f, vars) \
REAL_TEMPLATE_H T f(BOOST_PP_REPEAT(BOOST_PP_TUPLE_SIZE(vars), DEFINE_FUNCTION_GET_ARG, vars));

// COMPLEX:
#define COMPLEX_TYPE_LIST(z, n, dtype) BOOST_PP_COMMA_IF(n) complex<dtype>
#define INSTANTIATE_COMPLEX_TEMPLATE_NOARGS(f, dtype) template PHYTORCH_DEVICE complex<dtype> f<dtype, complex<dtype>>
#define INSTANTIATE_COMPLEX_TEMPLATE_ONE(f, n, dtype) INSTANTIATE_COMPLEX_TEMPLATE_NOARGS(f, dtype)(BOOST_PP_REPEAT(n, COMPLEX_TYPE_LIST, dtype));
#define INSTANTIATE_COMPLEX_TEMPLATE(f, n) INSTANTIATE_COMPLEX_TEMPLATE_ONE(f, n, float) INSTANTIATE_COMPLEX_TEMPLATE_ONE(f, n, double)

#define COMPLEX_TEMPLATE_H template <typename scalar_t, typename T> PHYTORCH_DEVICE
#define COMPLEX_TEMPLATE template <typename scalar_t, typename T=complex<scalar_t>> PHYTORCH_DEVICE

#define DEFINE_COMPLEX_FUNCTION(f, vars) \
INSTANTIATE_COMPLEX_TEMPLATE(f, BOOST_PP_TUPLE_SIZE(vars)) \
COMPLEX_TEMPLATE T f(BOOST_PP_REPEAT(BOOST_PP_TUPLE_SIZE(vars), DEFINE_FUNCTION_GET_ARG, vars))

#define DEFINE_COMPLEX_FUNCTION_H(f, vars) \
COMPLEX_TEMPLATE_H T f(BOOST_PP_REPEAT(BOOST_PP_TUPLE_SIZE(vars), DEFINE_FUNCTION_GET_ARG, vars));
