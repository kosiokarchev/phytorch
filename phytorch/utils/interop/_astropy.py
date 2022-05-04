from warnings import warn


class NothingImplemented:
    def __getattribute__(self, item):
        return self


try:
    from astropy import cosmology, units
except ImportError:
    warn(ImportWarning('Astropy unavailable; interop won\'t work.'))

    cosmology = units = NothingImplemented()
