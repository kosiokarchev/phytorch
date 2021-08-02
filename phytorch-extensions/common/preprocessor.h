#pragma once

#define BOOST_PP_LIMIT_SEQ 256
#define BOOST_PP_LIMIT_TUPLE 256

#include <boost/preprocessor.hpp>


#define HORNER_VAR x
#define HORNER_POLYOP(d, state, elem) (elem) + (HORNER_VAR) * (state)
#define HORNER(coeffs) BOOST_PP_LIST_FOLD_LEFT(HORNER_POLYOP, BOOST_PP_LIST_FIRST(BOOST_PP_TUPLE_TO_LIST(coeffs)), BOOST_PP_LIST_REST(BOOST_PP_TUPLE_TO_LIST(coeffs)))
