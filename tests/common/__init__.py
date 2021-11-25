from itertools import starmap

from more_itertools import pairwise
from torch import Tensor


def are_same_view(*ts: Tensor):
    return all(starmap(Tensor.is_set_to, pairwise(ts)))