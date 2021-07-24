from dataclasses import dataclass
from functools import cache, cached_property, reduce
from math import factorial
from operator import mul
from typing import Callable, ClassVar, Generic, Iterable, Sequence, Union

from more_itertools import always_iterable

from ...special import ellipr as _ellipr
from ...utils._typing import _T
from ...utils.symmetry import elementary_symmetric, partitions, product, sign


def onerange(end):
    return range(1, end+1)


@dataclass(frozen=True)
class OneIndexedSequence(Sequence[_T]):
    seq: Sequence[_T]

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, item):
        return self.seq[item-1]

    def __iter__(self):
        return iter(self.seq)


@dataclass(frozen=True)
class OneIndexedFormula(Generic[_T]):
    a: OneIndexedSequence[_T]
    b: OneIndexedSequence[_T]

    def formula(self, *args):
        raise NotImplementedError

    @cache
    def __getitem__(self, item):
        return self.formula(*always_iterable(item))


class d(OneIndexedFormula):
    def formula(self, i, j):
        return (
            -self.formula(j, i) if j<i else
            self.b[j] if i==0 else
            self.a[i] * self.b[j] - self.a[j] * self.b[i]
        )


@dataclass(frozen=True)
class XorY(OneIndexedFormula):
    def formula(self, i):
        return 1 if i == 0 else (self.a[i] + self.b[i]*self.xory)**0.5

    xory: _T


@dataclass(unsafe_hash=True)
class EllipticReduction:
    x: _T
    y: _T
    a: Union[OneIndexedSequence[_T], Sequence[_T]]
    b: Union[OneIndexedSequence[_T], Sequence[_T]]
    h: int = 4

    elliprc: ClassVar[Callable[[_T, _T], _T]] = staticmethod(_ellipr.elliprc)
    elliprd: ClassVar[Callable[[_T, _T, _T], _T]] = staticmethod(_ellipr.elliprd)
    elliprf: ClassVar[Callable[[_T, _T, _T], _T]] = staticmethod(_ellipr.elliprf)
    elliprj: ClassVar[Callable[[_T, _T, _T, _T], _T]] = staticmethod(_ellipr.elliprj)

    @cached_property
    def n(self):
        return len(self.a)

    def __post_init__(self):
        if not isinstance(self.a, OneIndexedSequence):
            self.a = OneIndexedSequence(self.a)
        if not isinstance(self.b, OneIndexedSequence):
            self.b = OneIndexedSequence(self.b)

    @cached_property
    def d(self):
        return d(self.a, self.b)

    @cached_property
    def X(self):
        return XorY(self.a, self.b, self.x)

    @cached_property
    def Y(self):
        return XorY(self.a, self.b, self.y)

    @cache
    def U2(self, i: int, j: int):
        if j < i:
            return self.U2(j, i)
        elif i==3 and j==4:
            return self.U2(1, 2)
        elif i==1 and j==2:
            k, l = 3, 4
            return ((self.X[i]*self.X[j] * self.Y[k]*self.Y[l] + self.Y[i]*self.Y[j] * self.X[k]*self.X[l]) / (self.x - self.y))**2
        else:
            k, l = {1, 2, 3, 4} - {i, j}
            return self.d[i, l]*self.d[j, k] + self.U2(i, k)

    @cache
    def U2nu(self, i: int, nu: int):
        j, k, l = {1, 2, 3, 4} - {i}
        return self.U2(i, j) - self.d[i, k] * self.d[i, l] * self.d[j, nu] / self.d[i, nu]

    @cache
    def S2(self, i, nu):
        j, k, l = {1, 2, 3, 4} - {i}
        return ((self.X[j]*self.X[k]*self.X[l] / self.X[i] * self.Y[nu]**2 + self.Y[j]*self.Y[k]*self.Y[l] / self.Y[i] * self.X[nu]**2) / (self.x - self.y))**2

    @cache
    def Q2(self, i, nu):
        return self.X[nu]**2 * self.Y[nu]**2 / (self.X[i]**2 * self.Y[i]**2) * self.U2nu(i, nu)

    @cache
    def Ie(self, i: int):
        if abs(i) > self.n:
            raise ValueError(f'-{self.n} <= i <= {self.n}')
        if i < -self.h:
            i, j, k, l, nu = (1, 2, 3, 4, -i)
            return (
                2 * self.b[nu] * (
                    self.d[i, j]*self.d[i, k]*self.d[i, l] / self.d[i, nu] / 3 * self.elliprj(self.U2(1, 2), self.U2(1, 3), self.U2(2, 3), self.U2nu(i, nu))
                    + self.elliprc(self.S2(i, nu), self.Q2(i, nu))
                ) - self.b[i] * self.Ie(0)
            ) / self.d[i, nu]
        elif i < 0:
            i = -i
            j, k, l = {1, 2, 3, 4} - {i}
            return (2*self.b[i] * (self.d[j, k]*self.d[j, l]/3 * self.elliprd(self.U2(i, k), self.U2(j, k), self.U2(i, j)) + self.X[j]*self.Y[j] / (self.X[i]*self.Y[i]*self.U2(i, j)**0.5)) - self.b[j] * self.Ie(0)) / self.d[j, i]
        elif i==0:
            return 2*self.elliprf(self.U2(1, 2), self.U2(1, 3), self.U2(2, 3))
        elif i <= self.h:
            j, k, l = {1, 2, 3, 4} - {i}
            return 2 * (
                self.d[i, j] * self.d[i, k] * self.d[l, i] * self.elliprj(self.U2(1, 2), self.U2(1, 3), self.U2(2, 3), self.U2nu(i, 0)) / 3 / self.b[i]
                + self.elliprc(self.S2(i, 0), self.Q2(i, 0))
            )
        elif i <= self.n:
            return (self.b[i] * self.Ie(1) + self.d[i, 1] * self.Ie(0)) / self.b[1]

    @cache
    def sigma(self, p, j=0):
        # (3.3), (3.4)
        return (
            reduce(mul, (self.b[i] for i in onerange(self.h))) if p==0 else
            self.sigma(0) * elementary_symmetric(p, [self.d[i, j] / self.b[i] for i in onerange(self.h)])
        )

    def v(self, m: Iterable[int], z):
        # (3.2)
        return product(
            (self.a[i] + self.b[i] * z) ** (_m - (i <= self.h) * (1/2))
            for i, _m in zip(onerange(self.n), m)
        )

    @cache
    def vx(self, m: Iterable[int]):
        # (3.2)
        return self.v(m, self.x)

    @cache
    def vy(self, m: Iterable[int]):
        # (3.2)
        return - self.v(m, self.y)

    @cache
    def A(self, m: Iterable[int]):
        # (3.1)
        return self.vx(m) + self.vy(m)

    @cache
    def _Iqe_pref(self, q: int, r: int, j: int):
        return (q + r / 2 + 1) * self.sigma(self.h - r, j)

    @cache
    def Iqe(self, q: int, j: int):
        # (3.5) and see paragraph after proof
        if q > 1:
            ri = self.h
            low = 0 if j > self.h else 1
            high = self.h
        elif q < -1:
            if j > self.h:
                ri = 0
                low = 1
            else:
                ri = 1
                low = 2
            high = self.h+1
        else:
            return self.Ie(q*j)

        q = q - ri
        return (
            self.b[j]**(self.h-1) * self.A(tuple((i<=self.h) + (q+1) * (i == j) for i in onerange(self.n)))
            - sum(self._Iqe_pref(q, r, j) * self.Iqe(q+r, j) for r in range(low, high))
        ) / self._Iqe_pref(q, ri, j)

    @cache
    def B(self, m: Iterable[int]):
        # (2.5)
        return product(bj ** mj for bj, mj in zip(self.b, m))

    @cache
    def D(self, m: OneIndexedSequence[int], i: int):
        # (2.5)
        return product(self.d[j, i] ** m[j] for j in onerange(self.n) if j != i)

    @cache
    def mu(self, m: OneIndexedSequence[int], s: int, i: int):
        # (2.6)
        return (-1 / abs(s) * sum(
            m[j] * (self.d[i, j] / self.b[j])**s
            for j in onerange(self.n) if j != i
        ))

    @cache
    def C(self, m: OneIndexedSequence[int], s: int, i: int):
        # (2.7)
        return 1 if s==0 else sum(
            product(self.mu(m, ss*v, i)**c / factorial(c) for v, c in part)
            for ss in [sign(s)]
            for part in partitions(abs(s))
        )

    @cache
    def Im(self, m: Sequence[int]):
        i0 = 1
        M = sum(m)
        m = OneIndexedSequence(m)
        return (
            self.B(m) * self.b[i0]**(-M) * sum(self.C(m, M-q, i0) * self.Iqe(q, i0) for q in range(0, M+1))
            + sum(
                self.D(m, i) * self.b[i]**(m[i]-M) * sum(
                    self.C(m, m[i]+q, i) * self.Iqe(-q, i)
                    for q in onerange(-m[i])
                )
                for i in onerange(self.n)
            )
        )
