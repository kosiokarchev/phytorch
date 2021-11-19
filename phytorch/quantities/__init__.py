from . import quantity

# If not for anything else, this import is necessary to ensure that
# TensorQuantity is registered in GenericQuantity._generic_quantiy_subtypes.
from .tensor_quantity import TensorQuantity as Quantity
