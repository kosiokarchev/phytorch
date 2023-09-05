#include "../common/implement_cpu.h"
#include "hyper.h"


COMPLEX_TEMPLATE T hyp0f1(T b, T z) { return hyp0f1_<scalar_t>(b, z); }
COMPLEX_TEMPLATE T hyp1f0(T a, T z) { return hyp1f0_<scalar_t>(a, z); }
COMPLEX_TEMPLATE T hyp1f1(T a, T b, T z) { return hyp1f1_<scalar_t>(a, b, z); }

IMPLEMENT_CPU(hyp0f1, complex<scalar_t>)
IMPLEMENT_CPU(hyp1f0, complex<scalar_t>)
IMPLEMENT_CPU(hyp1f1, complex<scalar_t>)
