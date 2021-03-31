import contextlib
import operator
import typing as tp
from enum import auto, Enum
from fractions import Fraction

from .delegator import Delegating, PowerDelegator, ProductDelegator, QuantityDelegator
from ..units.angular import rad
from ..units.Unit import Unit


_T = tp.TypeVar('_T')


class GenericQuantity(Delegating, tp.Generic[_T]):
    unit: Unit = None
    _generic_class_index = 0

    @classmethod
    @property
    def _T(cls):
        return tp.get_args(cls.__orig_bases__[cls._generic_class_index])[0]

    def __init__(self, *args, unit: Unit, **kwargs):
        super().__init__()
        self.unit = unit

    def _fill_quantity(self, a, unit=None):
        if unit is not False and isinstance(a, self._T):
            if not issubclass(type(a), type(self)):
                a.__class__ = type(self)
            a.unit = unit(self.unit) if callable(unit) else unit if isinstance(unit, Unit) else self.unit
        return a

    def __repr__(self):
        with self.unitless_context() as uself:
            trep = uself.__repr__()
        if '\n' in trep:
            trep = '\n' + trep
        return f'Quantity({trep} {self.unit!s})'

    def __setitem__(self, key, value):
        if isinstance(value, self._T):
            if not isinstance(value, GenericQuantity):
                value = type(self)(value, unit=Unit())
            if value.unit.dimension != self.unit.dimension:
                raise ValueError(f'Cannot assign units {value.unit} to {self.unit}')
            value = value.to(self.unit)
        elif self.unit:
            raise TypeError(f'Can only assign a {self._T} to unitful {type(self)}.')

        with self.delegator_context:
            return super().__setitem__(key, value)

    __add__, __radd__, __iadd__, __sub__, __rsub__, __isub__ = (QuantityDelegator() for _ in range(6))

    add = __add__
    sub = subtract = __sub__

    __mul__ = ProductDelegator()
    __rmul__ = ProductDelegator()
    __imul__ = ProductDelegator(incremental=True)
    ((__truediv__, __rtruediv__, __itruediv__),
     (__floordiv__, __rfloordiv__, __ifloordiv__),
     (__mod__, __rmod__, __imod__)) = (
        (ProductDelegator(op=op), ProductDelegator(op=op, flip=True), ProductDelegator(op=op, incremental=True))
        for op in (operator.truediv, operator.floordiv, operator.mod))

    mul = multiply = __mul__
    cross, dot, matmul, mm, mv, outer, vdot = (ProductDelegator() for _ in range(7))
    div = divide = true_divide = __truediv__
    floor_divide = ProductDelegator(op=operator.floordiv)
    fmod = remainder = __rmod__

    pow = __pow__ = PowerDelegator()
    __ipow__ = PowerDelegator()
    matrix_power = PowerDelegator()

    def __rpow__(self, other):
        if self.unit:
            raise TypeError('Cannot raise to the power of a quantity.')
        # TODO: other case of __rpow__

    __eq__, __ne__, __gt__, __ge__, __lt__, __le__ = (QuantityDelegator(out_unit=False) for _ in range(6))
    eq = equal = __eq__
    ne = not_equal = __ne__
    ge = greater_equal = __ge__
    gt = greater = __gt__
    le = less_equal = __le__
    lt = less = __lt__

    sqrt = QuantityDelegator(out_unit=lambda unit: unit**Fraction(1, 2))
    rsqrt = QuantityDelegator(out_unit=lambda unit: unit**Fraction(-1, 2))
    square = QuantityDelegator(out_unit=lambda unit: unit**Fraction(2))
    var = QuantityDelegator(out_unit=lambda unit: unit**Fraction(2))
    reciprocal, inverse, pinverse = (QuantityDelegator(out_unit=lambda unit: unit**Fraction(-1))
                                     for i in range(3))

    (exp, expm1, log, log10, log2, log1p, logaddexp, logaddexp2,
     logsumexp, logcumsumexp, matrix_exp,
     sinh, cosh, tanh, asinh, arcsinh, acosh, arccosh, atanh, arctanh,
     digamma, erf, erfc, erfinv, lgamma, logit, i0,
     mvlgamma, polygamma, sigmoid) = (
        QuantityDelegator(in_unit=Unit(), out_unit=False) for _ in range(30))

    # These are actually the same as the transcendental, but whatever
    sin, cos, tan = (QuantityDelegator(in_unit=rad, out_unit=False) for _ in range(3))
    (asin, arcsin), (acos, arccos), (atan, arctan) = (
        2*(QuantityDelegator(in_unit=Unit(), out_unit=rad),) for _ in range(3))

    atan2 = QuantityDelegator(out_unit=rad)
    angle = QuantityDelegator(out_unit=rad)

    (sign, argmax, argmin, count_nonzero, bincount, argsort,
     allclose, isclose, isfinite, isinf, isposinf, isneginf, isnan, isreal,
     histc) = (QuantityDelegator(out_unit=False) for _ in range(15))

    @property
    def value(self) -> _T:
        raise NotImplementedError()

    def to(self, unit: Unit = None, *args, **kwargs):
        if hasattr(super(), 'to'):
            with self.delegator_context:
                ret = super().to(*args, **kwargs)
        else:
            ret = self

        if self.unit == unit:
            return self._fill_quantity(ret)
        if unit is not None:
            if ret.unit is not None:
                with self.unitless_context(ret.clone() if ret is self else ret) as ret:
                    # TODO: better handling of float()...
                    ret *= float(self.unit.to(unit))
            ret.unit = unit
        return ret

    def clone(self, *args, **kwargs) -> 'GenericQuantity':
        return super().clone(*args, **kwargs)

    @contextlib.contextmanager
    def unitless_context(self, clone=None):
        clone = clone if clone is not None else self.clone()
        clone.__class__ = self._T
        yield clone
        clone.__class__ = type(self)


class UnitFuncType(Enum):
    NOUNIT = auto()  # function produces unitless value, allow all
    SAME = auto()    # ensure unit of output is the same as input
    SUM = auto()     # multiple tensors, allow only if all units the same
    PRODUCT = auto()  # any units allowed, final unit is product of units
    POWER = auto()   # two inputs, allow only if second is number or dimensionless
    TRANS = auto()   # transcendental, dimensionless -> dimensionless
    RAD = auto()     # convert to radians, output is dimensionless
    INVRAD = auto()  # input is dimensionless, interpret output as radians
    SAMETORAD = auto()  # ensure all same unit -> radians
    SAMETODIMLESS = auto()  # ensure all same unit -> dimensionless
    REJECT = auto()  # always reject
