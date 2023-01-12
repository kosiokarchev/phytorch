from __future__ import annotations

from collections import OrderedDict
from typing import MutableMapping, Protocol, Type, TypeVar, Union

from typing_extensions import Self

from .delegating import Delegating
from ..meta import Meta
from ..units.unit import Unit


class QuantityBackendProtocol(Protocol):
    def as_subclass(self, cls: Type[_t]) -> _t: ...
    def clone(self, *args, **kwargs) -> Self: ...
    def to(self, *args, **kwargs) -> Self: ...


_t = TypeVar('_t', bound=QuantityBackendProtocol)


class GenericQuantity(Delegating[_t], Meta):
    """A *quantity*: combining a `Unit` with numerical functionalities provided by a backend type."""

    _generic_quantity_subtypes: MutableMapping[Type, Type[GenericQuantity]] = OrderedDict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._generic_quantity_subtypes.setdefault(cls._T, cls)

    @classmethod
    def _from_bare_and_unit(cls, bare: _t, unit: Unit) -> Self:
        ret = bare.as_subclass(cls)
        ret.unit = unit
        return ret

    unit: Meta.meta_attribute(Unit) = None
    _T: Union[Type[QuantityBackendProtocol], Type]

    @property
    def value(self) -> _t:
        raise NotImplementedError()

    def to(self, unit: Unit = None, *args, **kwargs):
        if hasattr(super(), 'to'):
            with self.delegator_context:
                ret = self._T.to(self, *args, **kwargs)
        else:
            ret = self

        if self.unit == unit:
            return ret

        if unit is not None:
            if ret.unit is not None:
                ret = ret._T.clone(ret) if ret is self else ret
                with ret.delegator_context:
                    # TODO: better handling of float()...
                    ret *= float(self.unit.to(unit))
        else:
            unit = self.unit
        return self._meta_update(self._T.as_subclass(ret, type(self)), unit=unit)

    def __repr__(self):
        with self.delegator_context:
            trep = super().__repr__()
        if '\n' in trep:
            trep = '\n' + trep
        return f'Quantity({trep} {self.unit!s})'
