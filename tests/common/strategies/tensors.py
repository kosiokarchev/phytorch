from itertools import repeat

import torch
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_numpy


random_tensors = st.lists(st.integers(1, 16), min_size=0, max_size=4).map(torch.rand)


def n_broadcastable_random_tensors(n):
    return st_numpy.mutually_broadcastable_shapes(num_shapes=2, max_dims=4, max_side=14).map(
        lambda bs: map(torch.rand, bs.input_shapes)
    )


def n_tensors_strategy(n, elements: st.SearchStrategy = st.floats(min_value=1e-4, max_value=1e3), max_len=10):
    return st.integers(min_value=1, max_value=max_len).flatmap(lambda m: st.tuples(
        *repeat(st.lists(elements, min_size=m, max_size=m), n)
    )).map(torch.tensor)


def n_complex_tensors_strategy(n, max_len=10, min_magnitude=1e-4, max_magnitude=1e3):
    return n_tensors_strategy(n, st.complex_numbers(min_magnitude=min_magnitude, max_magnitude=max_magnitude),
                              max_len=max_len)
