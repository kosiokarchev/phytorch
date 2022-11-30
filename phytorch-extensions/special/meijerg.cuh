#pragma once

#include "hyper.cuh"


COMPLEX_TEMPLATE_H T meijerg(size_t m, size_t n, size_t p, size_t q, const vector<T>& args, T z, T r, HYP_KWARGS);
COMPLEX_TEMPLATE_H T Tgamma(size_t m, T a, T z);
