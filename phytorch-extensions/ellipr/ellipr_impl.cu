#include "../common/implement.cuh"
#include "ellipr.cuh"


IMPLEMENT(elliprc, (T x, T y), (x, y), T)
IMPLEMENT(elliprd, (T x, T y, T z), (x, y, z), T)
IMPLEMENT(elliprf, (T x, T y, T z), (x, y, z), T)
IMPLEMENT(elliprg, (T x, T y, T z), (x, y, z), T)
IMPLEMENT(elliprj, (T x, T y, T z, T p), (x, y, z, p), T)
