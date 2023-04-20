#include "../common/implement_cuda.cuh"
#include "gammaincinv.h"

IMPLEMENT_CUDA_REAL(gammaincinv, (a, p))
IMPLEMENT_CUDA_REAL(gammainccinv, (a, q))