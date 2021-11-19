from itertools import chain
from typing import Iterable

import torch
from more_itertools import split_into
from torch import broadcast_shapes, Size, Tensor


def broadcast_except(*tensors: Tensor, dim=-1):
    shape = broadcast_shapes(*(t.select(dim, 0).shape for t in tensors))
    return [t.expand(*shape[:t.ndim + dim + 1], t.shape[dim], *shape[t.ndim + dim + 1:])
            for t in pad_dims(*tensors, ndim=len(shape)+1)]


def broadcast_left(*tensors, ndim):
    shape = broadcast_shapes(*(t.shape[:ndim] for t in tensors))
    return (t.expand(*shape, *t.shape[ndim:]) for t in tensors)


def broadcast_gather(input, dim, index, sparse_grad=False, index_ndim=1):
    """
    input: Size(batch_shape..., N, event_shape...)
    index: Size(batch_shape..., index_shape...)
       ->: Size(batch_shape..., index_shape..., event_shape...)
    """
    index_shape = index.shape[-index_ndim:]
    index = index.flatten(-index_ndim)
    batch_shape = broadcast_shapes(input.shape[:dim], index.shape[:-1])
    input = input.expand(*batch_shape, *input.shape[dim:])
    index = index.expand(*batch_shape, index.shape[-1])
    return torch.gather(input, dim, index.reshape(
        *index.shape, *(input.ndim - index.ndim)*(1,)).expand(
        *index.shape, *input.shape[index.ndim:]
    ) if input.ndim > index.ndim else index, sparse_grad=sparse_grad).reshape(*index.shape[:-1], *index_shape, *input.shape[dim % input.ndim + 1:])


# TODO: improve so that nbatch=-1 means "auto-derive nbatch from number of
#  matching dimensions on the left"
def pad_dims(*tensors: Tensor, ndim: int = None, nbatch: int = 0) -> list[Tensor]:
    """Pad shapes with ones on the left until at least `ndim` dimensions."""
    if ndim is None:
        ndim = max([t.ndim for t in tensors])
    return [t.reshape(t.shape[:nbatch] + (1,)*(ndim-t.ndim) + t.shape[nbatch:]) for t in tensors]


def align_dims(t: Tensor, ndims: Iterable[int], target_ndims: Iterable[int]):
    assert sum(ndims) == t.ndim
    return t.reshape(*chain.from_iterable(
        (target_ndim - len(s)) * [1] + s for s, target_ndim
        in zip(split_into(t.shape, ndims), target_ndims)
    ))


def aligned_expand(t: Tensor, ndims: Iterable[int], shapes: Iterable[Size]):
    return align_dims(t, ndims, map(len, shapes)).expand(*chain.from_iterable(shapes))
