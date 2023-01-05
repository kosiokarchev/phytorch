#include "../common/implement_cuda.cuh"
#include "ellip.h"

IMPLEMENT_CUDA_COMPLEX(ellipk, (m))
IMPLEMENT_CUDA_COMPLEX(ellipe, (m))
IMPLEMENT_CUDA_COMPLEX(ellipd, (m))
IMPLEMENT_CUDA_COMPLEX(ellippi, (n, m))

IMPLEMENT_CUDA_COMPLEX(csc2, (phi))

IMPLEMENT_CUDA_COMPLEX(ellipkinc_, (c, m))
IMPLEMENT_CUDA_COMPLEX(ellipeinc_, (c, m))
IMPLEMENT_CUDA_COMPLEX(ellipdinc_, (c, m))
IMPLEMENT_CUDA_COMPLEX(ellippiinc_, (n, c, m))

IMPLEMENT_CUDA_COMPLEX(ellipkinc, (phi, m))
IMPLEMENT_CUDA_COMPLEX(ellipeinc, (phi, m))
IMPLEMENT_CUDA_COMPLEX(ellipdinc, (phi, m))
IMPLEMENT_CUDA_COMPLEX(ellippiinc, (n, phi, m))