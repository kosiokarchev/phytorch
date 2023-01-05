#pragma once

#ifdef __CUDACC__
#define PHYTORCH_HOST_DEVICE __host__ __device__
#define PHYTORCH_DEVICE __host__ __device__
#else
#define PHYTORCH_HOST_DEVICE
#define PHYTORCH_DEVICE
#endif
