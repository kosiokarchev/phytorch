#include "../common/implement.cuh"
#include "elliptic.cuh"


IMPLEMENT_COMPLEX(elliprc, (x, y))
IMPLEMENT_COMPLEX(elliprd, (x, y, z))
IMPLEMENT_COMPLEX(elliprf, (x, y, z))
IMPLEMENT_COMPLEX(elliprg, (x, y, z))
IMPLEMENT_COMPLEX(elliprj, (x, y, z, p))

IMPLEMENT_COMPLEX(ellipk, (m))
IMPLEMENT_COMPLEX(ellipe, (m))
IMPLEMENT_COMPLEX(ellipd, (m))
IMPLEMENT_COMPLEX(ellippi, (n, m))

IMPLEMENT_COMPLEX(csc2, (phi))

IMPLEMENT_COMPLEX(ellipkinc_, (c, m))
IMPLEMENT_COMPLEX(ellipeinc_, (c, m))
IMPLEMENT_COMPLEX(ellipdinc_, (c, m))
IMPLEMENT_COMPLEX(ellippiinc_, (n, c, m))

IMPLEMENT_COMPLEX(ellipkinc, (phi, m))
IMPLEMENT_COMPLEX(ellipeinc, (phi, m))
IMPLEMENT_COMPLEX(ellipdinc, (phi, m))
IMPLEMENT_COMPLEX(ellippiinc, (n, phi, m))
