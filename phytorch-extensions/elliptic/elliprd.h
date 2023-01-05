#pragma once

#include "elliptic.h"


DEFINE_COMPLEX_FUNCTION(elliprd, (x, y, z)) {
    ELLIPR_CHECK_xyz
    if (not z or (not x + not y + not z > 1)) return numeric_limits<T>::infinity();

    auto Am = (x + y + ltrl(3.)*z) / ltrl(5.);
    ELLIPR_HEADER(1./4.)

    T s = 0.;
    while (pow4 * Q > abs(Am)) {
        ELLIPR_STEP
        auto zm1 = zm + lm;
        s += ltrl(pow4) / sqrt(zm) / zm1;
        ELLIPR_UPDATE(zm1)
    }

    ELLIPR_XY
    auto Z = - (X + Y) / ltrl(3.),
         XY = X*Y, Z2 = Z*Z, XYZ = XY*Z,
         E2 = XY - ltrl(6.)*Z2,
         E3 = ltrl(3.)*XYZ - ltrl(8.)*Z2*Z,
         E4 = ltrl(3.) * (XY - Z2) * Z2,
         E5 = XYZ * Z2;
    return ltrl(3.)*s + ltrl(pow4) / Am / sqrt(Am) * (
        ltrl(1.)
        - ltrl(3./14.) * E2
        + E3 / ltrl(6.)
        + ltrl(9./88.) * E2*E2
        - ltrl(3./22.) * E4
        - ltrl(9./52.) * E2*E3
        + ltrl(3./26.) * E5);
}
