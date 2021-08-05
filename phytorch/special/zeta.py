from ..extensions import special as _special
from ..utils.complex import with_complex_args
from ..utils.function_context import ComplexTorchFunction


# noinspection PyMethodOverriding
class Zeta(ComplexTorchFunction):
    @staticmethod
    def _forward(ctx, z, *args):
        return _special.zeta(z)


zeta = with_complex_args(Zeta.apply)
