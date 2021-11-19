from typing import Any, Iterable, Mapping, TypeVar, Union

from hypothesis import strategies as st
from torch import Tensor
from typing_extensions import TypeAlias

from phytorch.quantities.quantity import GenericQuantity
from phytorch.utils.symmetry import product

from tests.common.strategies.tensors import random_tensors
from tests.common.strategies.units import units_strategy


_VT: TypeAlias = Tensor
_nestedVT: TypeAlias = Union[Mapping[Any, '_nestedVT'], Iterable['_nestedVT'], _VT]
ConcreteQuantity = TypeVar('ConcreteQuantity', GenericQuantity[_VT], _VT)
quantities_strategy = st.tuples(random_tensors, units_strategy).map(product)