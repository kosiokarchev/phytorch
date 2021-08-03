from math import cos, gamma, log, pi, sin, atan

from pytest import mark
from torch import isclose, tensor

from phytorch.special.hyper import hyp2f1
from tests.common import with_default_double


cases = {
    # https://github.com/scipy/scipy/blob/3c35f0a3abdd07c2d2b8f2f1528709d6a393498e/scipy/special/tests/test_basic.py#L2140
    hyp2f1: (
        (0.5, 1, 1.5, 0.2**2, 0.5 / 0.2 * log((1 + 0.2) / (1 - 0.2))),
        (0.5, 1, 1.5, -0.2**2, 1. / 0.2 * atan(0.2)),
        (1, 1, 2, 0.2, -1 / 0.2 * log(1 - 0.2)),
        (3, 3.5, 1.5, 0.2**2, 0.5 / 0.2 / (-5) * ((1 + 0.2)**(-5) - (1 - 0.2)**(-5))),
        (-3, 3, 0.5, sin(0.2)**2, cos(2 * 3 * 0.2)),
        # z==1
        (3, 4, 8, 1, gamma(8) * gamma(8 - 4 - 3) / gamma(8 - 3) / gamma(8 - 4)),
        (3, 2, 2, -1, 1. / 2**3 * pi**0.5 * gamma(1 + 3 - 2) / gamma(1 + 0.5 * 3 - 2) / gamma(0.5 + 0.5 * 3)),
        (5, 2, 4, -1, 1. / 2**5 * pi**0.5 * gamma(1 + 5 - 2) / gamma(1 + 0.5 * 5 - 2) / gamma(0.5 + 0.5 * 5)),
        (4, 4.5, -6.5, -1. / 3, (8. / 9)**(-2 * 4) * gamma(4. / 3) * gamma(1.5 - 2 * 4) / gamma(3. / 2) / gamma(4. / 3 - 2 * 4)),
        (1.5, -0.5, 1.0, -10.0, 4.1300097765277476484),
        # # negative integer a or b, with c-a-b integer and x > 0.9
        (-2, 3, 1, 0.95, 0.715),
        (2, -3, 1, 0.95, -0.007),
        (-6, 3, 1, 0.95, 0.0000810625),
        (2, -5, 1, 0.95, -0.000029375),
        # # huge negative integers
        # TODO: gamma(complex) without overflowing...
        # (10, -900, 10.5, 0.99, 1.91853705796607664803709475658e-24),
        # (10, -900, -10.5, 0.99, 3.54279200040355710199058559155e-18),
    )
}


@mark.parametrize('a, b, c, z, target', cases[hyp2f1])
@with_default_double
def test_hyp2f1(a, b, c, z, target):
    assert isclose(res := hyp2f1(a, b, c, z), tensor(target, dtype=res.dtype))
