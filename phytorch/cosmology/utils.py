from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Union

import forge
from typing_extensions import TypeAlias

from ..quantities.quantity import GenericQuantity
from ..units.unit import Unit, Dimension
from ..utils._typing import ValueProtocol


_GQuantity: TypeAlias = Union[GenericQuantity, Unit, ValueProtocol]


class _no_value_enum(enum.Enum):
    _no_value = enum.auto()

    def __repr__(self):
        return self.name


_no_value = _no_value_enum._no_value


@dataclass
class AbstractParameter:
    default: Any = forge.empty


class Parameter(AbstractParameter):
    pass


class IndirectPropertyParameter(property, AbstractParameter):
    pass


class PropertyParameter(IndirectPropertyParameter):
    default = _no_value


Hdim = Dimension('H')
Hunit = Unit(**{Hdim: 1})
hunit = lambda _: Hunit / _
h = h100 = hunit(100)
