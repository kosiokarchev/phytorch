#include "../common/implement_cpu.h"
#include "ellip.h"

IMPLEMENT_CPU(ellipk, complex<scalar_t>)
IMPLEMENT_CPU(ellipe, complex<scalar_t>)
IMPLEMENT_CPU(ellipd, complex<scalar_t>)
IMPLEMENT_CPU(ellippi, complex<scalar_t>)

IMPLEMENT_CPU(csc2, complex<scalar_t>)

IMPLEMENT_CPU(ellipkinc_, complex<scalar_t>)
IMPLEMENT_CPU(ellipeinc_, complex<scalar_t>)
IMPLEMENT_CPU(ellipdinc_, complex<scalar_t>)
IMPLEMENT_CPU(ellippiinc_, complex<scalar_t>)

IMPLEMENT_CPU(ellipkinc, complex<scalar_t>)
IMPLEMENT_CPU(ellipeinc, complex<scalar_t>)
IMPLEMENT_CPU(ellipdinc, complex<scalar_t>)
IMPLEMENT_CPU(ellippiinc, complex<scalar_t>)
