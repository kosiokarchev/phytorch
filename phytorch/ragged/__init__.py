from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from more_itertools import all_equal, split_into
from torch import Tensor


__all__ = 'RaggedShape', 'RaggedTensor'


@dataclass
class RaggedShape:
    lens: Sequence[Sequence[int]] = field(repr=False, default_factory=list)
    sizes: Sequence[Sequence[int]] = field(init=False, repr=False)
    ndim: int = field(init=False, compare=False)
    size: int = field(init=False, compare=False)

    def __post_init__(self):
        if self.lens:
            sizes = [self.lens[-1]]
            for i in range(len(self.lens)-1):
                sizes.append(list(map(sum, split_into(sizes[i], self.lens[-i-2]))))
        else:
            sizes = []
        self.sizes = sizes[::-1]
        self.ndim = len(self.lens)
        self.size = sum(sizes[0]) if sizes else 1

    @classmethod
    def from_nested_sizes(cls, nested_sizes, *args, **kwargs):
        q = deque(nested_sizes)

        lens = []
        while q:
            lens.append([])
            for i in range(len(q)):
                el = q.popleft()
                if isinstance(el, int):
                    lens[-1].append(el)
                else:
                    lens[-1].append(len(el))
                    q.extend(el)
        return cls(lens, *args, **kwargs)

    @classmethod
    def broadcastable(cls, *shapes: 'RaggedShape'):
        minndim = min(s.ndim for s in shapes)
        return all_equal(s.lens[:minndim] for s in shapes)


@dataclass
class RaggedTensor:
    data: Tensor
    shape: RaggedShape = field(default_factory=RaggedShape)

    def __post_init__(self):
        assert self.data.shape[-1] == self.shape.size

    def broadcast_to(self, shape: RaggedShape):
        assert self.shape.ndim < shape.ndim
        assert RaggedShape.broadcastable(self.shape, shape)
        return RaggedTensor(self.data.expand(self.data.shape[:-1] + (len(shape.lens[self.shape.ndim]),)).repeat_interleave(self.data.new_tensor(shape.sizes[self.shape.ndim], dtype=int), dim=-1, output_size=shape.size), shape)