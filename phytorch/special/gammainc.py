from torch import exp, log, Tensor
from torch.special import digamma, gammainc as _gammainc, gammaincc as _gammaincc, gammaln

from ..extensions import special as _special
from ..utils.complex import as_complex_tensors
from ..utils.function_context import CimplMixin, InverseMixin, TensorArgsMixin


class TGamma(CimplMixin):
    ninputs = 3

    # noinspection PyMethodOverriding
    @staticmethod
    def _forward(ctx, a, z, m, *args):
        # Switcheroo because _forward is called with (*saved_tensors, *orig_args)
        # and saved_tensors = a, z; orig_args = m, a, z
        # TODO: make C Tgamma(a, z, m)?
        return _special.Tgamma(m, a, z)

    # noinspection PyMethodOverriding
    @staticmethod
    def saved_tensors(ctx, m, a, z):
        ctx.m = m
        return a, z

    @staticmethod
    def grad_a(ctx, a, z, T):
        return log(z) * T + (ctx.m-1) * Tgamma(ctx.m+1, a, z)

    @staticmethod
    def grad_z(ctx, a, z, T):
        return - (Tgamma(ctx.m-1, a, z) + T) / z

    gradfuncs = None, grad_a, grad_z


def Tgamma(m: int, a: Tensor, z: Tensor) -> Tensor:
    return TGamma.apply(m, *as_complex_tensors(a, z))


class Gammainc(CimplMixin, TensorArgsMixin):
    _impl_func = _gammainc
    ninputs = 2

    @staticmethod
    def grad_a(ctx, a, x, p, q=None):
        if q is None:
            q = 1 - p
        return q * (digamma(a) - log(x)) - x*Tgamma(3, a, x).real / gammaln(a).exp()

    @staticmethod
    def grad_z(ctx, a, x, p):
        return x**(a-1) * exp(-x - gammaln(a))


class Gammaincc(CimplMixin, TensorArgsMixin):
    _impl_func = _gammaincc
    ninputs = 2

    @staticmethod
    def grad_a(ctx, a, x, q):
        return - Gammainc.grad_a(ctx, a, x, None, q)

    @staticmethod
    def grad_z(ctx, a, x, q):
        return - Gammainc.grad_z(ctx, a, x, None)


class Gammaincinv(CimplMixin, TensorArgsMixin, InverseMixin):
    _impl_func = _special.gammaincinv
    _forward_cls = Gammainc
    _forward_cls_grad_inv_var = Gammainc.grad_z


class Gammainccinv(CimplMixin, TensorArgsMixin, InverseMixin):
    _impl_func = _special.gammainccinv
    _forward_cls = Gammaincc
    _forward_cls_grad_inv_var = Gammaincc.grad_z


gammainc = Gammainc.apply
gammaincc = Gammaincc.apply
gammaincinv = Gammaincinv.apply
gammainccinv = Gammainccinv.apply


# __all__ = 'gammainc', 'gammaincc', 'gammaincinv', 'gammainccinv', 'Tgamma'
