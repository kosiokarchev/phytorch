#include "../common/implement_cuda.cuh"
#include "gammainc.h"

IMPLEMENT_CUDA_REAL(gammainc, (a, x))
IMPLEMENT_CUDA_REAL(gammaincc, (a, x))