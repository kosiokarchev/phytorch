# If not for anything else, this import is necessary to ensure that
# TorchQuantity is registered in GenericQuantity._generic_quantiy_subtypes.
from . import quantity
from .torchquantity import TorchQuantity as Quantity
