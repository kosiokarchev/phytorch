#pragma once

#define BOOST_PP_LIMIT_SEQ 256
#define BOOST_PP_LIMIT_TUPLE 256

#include <boost/preprocessor.hpp>

#define PREPEND_data(r, data, i, elem) BOOST_PP_COMMA_IF(i) (data, elem)

#define SIGNATURE_VARDECL(r, data, i, elem) BOOST_PP_COMMA_IF(i) BOOST_PP_TUPLE_ELEM(0, elem) BOOST_PP_TUPLE_ELEM(1, elem)
#define VARS_TO_SIGNATURE(vars) BOOST_PP_SEQ_FOR_EACH_I(SIGNATURE_VARDECL, _, BOOST_PP_TUPLE_TO_SEQ(vars))
#define SIGNATURE_VARCALL(r, data, i, elem) BOOST_PP_COMMA_IF(i) BOOST_PP_TUPLE_ELEM(1, elem)
#define VARS_TO_CALL(vars) BOOST_PP_SEQ_FOR_EACH_I(SIGNATURE_VARCALL, _, BOOST_PP_TUPLE_TO_SEQ(vars))

#define HORNER_VAR x
#define HORNER_POLYOP(d, state, elem) (elem) + (HORNER_VAR) * (state)
#define HORNER(coeffs) BOOST_PP_LIST_FOLD_LEFT(HORNER_POLYOP, BOOST_PP_LIST_FIRST(BOOST_PP_TUPLE_TO_LIST(coeffs)), BOOST_PP_LIST_REST(BOOST_PP_TUPLE_TO_LIST(coeffs)))
