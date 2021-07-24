import mpmath

from .core import EllipticReduction


class MPEllipticReduction(EllipticReduction):
    elliprc = staticmethod(mpmath.elliprc)
    elliprd = staticmethod(mpmath.elliprd)
    elliprf = staticmethod(mpmath.elliprf)
    elliprj = staticmethod(mpmath.elliprj)