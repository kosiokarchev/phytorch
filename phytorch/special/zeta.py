from ..extensions import special as _special
from ..utils.complex import with_complex_args
from ..utils.function_context import CargsMixin


# noinspection PyMethodOverriding
class Zeta(CargsMixin):
    @staticmethod
    def _forward(ctx, z, *args):
        return _special.zeta(z)


zeta = with_complex_args(Zeta.apply)
