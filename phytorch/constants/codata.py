from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, wraps
from math import pi
from numbers import Real
from types import ModuleType
from typing import Annotated, Callable, cast, Generic, get_args, get_type_hints, Iterable, Type, TypeVar, Union

from typing_extensions import ParamSpec, TypeAlias

from .constant import Constant
from ..units._si.base import K, kg, m, s
from ..units._si.coherent import C, J, N, Pa
from ..units.unit import Unit


def _cdvann(unit: Unit, descr: str) -> Type[Real]:
    return Annotated[Real, unit, descr]


@dataclass
class CODATA_vals:
    c: _cdvann(m / s, 'speed of light in vacuum')
    h: _cdvann(J * s, 'Planck constant')
    k: _cdvann(J / K, 'Boltzman constant')
    N_A: _cdvann(Unit(), 'Avogadro constant')
    e: _cdvann(C, 'elementary charge')
    atm: _cdvann(Pa, 'standard atmosphere')
    g: _cdvann(m / s**2, 'standard acceleration of gravity')
    α: _cdvann(Unit(), 'fine-structure constant')
    G: _cdvann(N * m**2 / kg**2, 'Newtonian constant of gravitation')
    m_p: _cdvann(kg, 'proton mass')
    m_n: _cdvann(kg, 'neutron mass')
    m_e: _cdvann(kg, 'electron mass')
    u: _cdvann(kg, 'atomic mass constant')


_cdvanns = {
    key: get_args(ann)[1:]
    for key, ann in get_type_hints(CODATA_vals, include_extras=True).items()
}


_T = TypeVar('_T')
_T1 = TypeVar('_T1')
_PS = ParamSpec('_PS')
_CODATA: TypeAlias = 'CODATA'


# noinspection PyUnusedLocal
def cast_(typ: Type[_T]) -> Callable[[Callable[[_PS], _T1]], Union[Type[_T1], Callable[[_PS], _T]]]:
    def f(cls):
        return cls
    return f


class cached_constant_property_base(cached_property, Generic[_T]):
    pass


class cached_constant_property(cached_constant_property_base, ABC):
    def __init__(self):
        super().__init__(self._get)

    @abstractmethod
    def _get(self, instance: _T): ...


@cast_(Constant)
class basic_constant(cached_constant_property[_CODATA]):
    def _get(self, instance: CODATA):
        unit, descr = _cdvanns[self.attrname]
        return Constant(self.attrname, getattr(instance._vals, self.attrname) * unit, descr)


@cast_(Constant)
class alias(cached_constant_property[_CODATA]):
    def __init__(self, other: cached_property):
        super().__init__()
        self.other = other

    def _get(self, instance: CODATA):
        return getattr(instance, self.other.attrname)


def derived_constant(descr: str):
    def outer(f):
        @cached_constant_property_base
        @wraps(f)
        def inner(self: CODATA):
            return Constant(f.__name__, f(self), descr)
        return inner
    return outer


class CODATA(ModuleType):
    @classmethod
    @property
    def __all__(cls):
        return tuple(key for key, val in vars(cls).items()
                     if isinstance(val, cached_constant_property_base))

    def __dir__(self) -> Iterable[str]:
        return set(super().__dir__()) | set(self.__all__)

    def __init__(self, name, doc, vals: CODATA_vals):
        super().__init__(name, doc)
        self._vals = vals

    c, h, k, N_A, e, atm, g, α, G, m_p, m_n, m_e, u = (
        basic_constant() for i in range(13))  # type: Constant

    alpha = alias(cast(cached_property, α))

    @derived_constant('reduced Planck constant')
    def ħ(self):
        return self.h / (2 * pi)

    hbar = alias(ħ)

    @derived_constant('vacuum magnetic permeability')
    def µ_0(self):
        return 4 * pi * self.alpha * self.hbar / self.e**2 / self.c

    mu_0 = alias(µ_0)

    @derived_constant('vacuum electric permittivity')
    def ε_0(self):
        return 1 / self.mu_0 / self.c**2

    eps_0 = alias(ε_0)

    @derived_constant('Bohr magneton')
    def µ_B(self):
        return self.e * self.hbar / (2 * self.m_e)

    mu_B = alias(µ_B)

    @derived_constant('nuclear magneton')
    def µ_N(self):
        return self.e * self.hbar / (2 * self.m_p)

    mu_N = alias(µ_N)

    @derived_constant('Rydberg constant')
    def Ryd(self):
        return self.alpha**2 * self.m_e * self.c / (2 * self.h)

    @derived_constant('Bohr radius')
    def a_0(self):
        return self.hbar / (self.alpha * self.m_e * self.c)

    @derived_constant('Thomson cross section')
    def σ_e(self):
        return (8 * pi / 3) * self.alpha**4 * self.a_0**2

    sigma_e = alias(σ_e)

    @derived_constant('Stefan-Boltzmann constant')
    def σ(self):
        return pi**2 / 60 * self.k**4 / (self.hbar**3 * self.c**2)

    sigma = alias(σ)

    @derived_constant('Wien wavelength displacement law constant')
    def b(self):
        return self.h * self.c / self.k / 4.965114231744276303

    @derived_constant('Wien frequency displacement law constant')
    def b_prime(self):
        return 2.821439372122078893 * self.k / self.h
