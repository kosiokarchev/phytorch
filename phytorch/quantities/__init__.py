from .quantity import GenericQuantity

# TODO: do this lazily:
#   Need to ensure that whenever Generic Quantity is defined, TensorQuantity is
#   registered in GenericQuantity._generic_quantiy_subtypes, so that
#   multiplication by Unit works.
from .tensor_quantity import TensorQuantity as Quantity
