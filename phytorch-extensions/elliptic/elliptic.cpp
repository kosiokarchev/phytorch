#include "../common/implement.h"

TORCH_IMPLEMENT(elliprc, (x, y))
TORCH_IMPLEMENT(elliprd, (x, y, z))
TORCH_IMPLEMENT(elliprf, (x, y, z))
TORCH_IMPLEMENT(elliprg, (x, y, z))
TORCH_IMPLEMENT(elliprj, (x, y, z, p))

TORCH_IMPLEMENT(ellipk, (m))
TORCH_IMPLEMENT(ellipe, (m))
TORCH_IMPLEMENT(ellipd, (m))
TORCH_IMPLEMENT(ellippi, (n, m))

TORCH_IMPLEMENT(csc2, (phi))

TORCH_IMPLEMENT(ellipkinc_, (c, m))
TORCH_IMPLEMENT(ellipeinc_, (c, m))
TORCH_IMPLEMENT(ellipdinc_, (c, m))
TORCH_IMPLEMENT(ellippiinc_, (n, c, m))

TORCH_IMPLEMENT(ellipkinc, (phi, m))
TORCH_IMPLEMENT(ellipeinc, (phi, m))
TORCH_IMPLEMENT(ellipdinc, (phi, m))
TORCH_IMPLEMENT(ellippiinc, (n, phi, m))

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    MDEF(elliprc) MDEF(elliprd) MDEF(elliprf) MDEF(elliprg) MDEF(elliprj)

    MDEF(ellipk) MDEF(ellipe) MDEF(ellipd) MDEF(ellippi)
    MDEF(csc2)
    MDEF(ellipkinc_) MDEF(ellipeinc_) MDEF(ellipdinc_) MDEF(ellippiinc_)
    MDEF(ellipkinc) MDEF(ellipeinc) MDEF(ellipdinc) MDEF(ellippiinc)
}