try:
    from astropy import cosmology, units
    from astropy.cosmology import Cosmology, LambdaCDM, wCDM, w0waCDM, wpwaCDM, w0wzCDM
    from astropy.units import PhysicalType, UnitBase

except ImportError:
    from typing import Type
    from warnings import warn


    class NothingImplemented:
        def __getattribute__(self, item):
            return self


    warn('AstroPy unavailable; interop won\'t work.', ImportWarning)

    cosmology = units = NothingImplemented()
    Cosmology = LambdaCDM = wCDM = w0waCDM = wpwaCDM = w0wzCDM = Type
    PhysicalType = UnitBase = Type
