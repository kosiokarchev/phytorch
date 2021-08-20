from functools import reduce
from itertools import combinations
from operator import and_

import torch
from hypothesis import assume


CLOSE_KWARGS = dict(atol=1e-6, rtol=1e-6)


def close(a, b, close_func=torch.allclose, **kwargs):
    b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    if torch.is_complex(a) and kwargs.get('equal_nan', True):
        return (close(a.real, b.real, close_func, equal_nan=True, **kwargs) and
                close(a.imag, b.imag, close_func, equal_nan=True, **kwargs))
    return close_func(a, b, **{
        'atol': max(1e-8, 100 * torch.finfo(a.dtype).eps),
        'rtol': max(1e-5, 100 * torch.finfo(a.dtype).eps),
        **kwargs})


def nice_and_close(a, b, close_func=torch.allclose, **kwargs):
    b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    assume(not torch.isnan(a) and not torch.isnan(b))
    return close(a, b, close_func, **kwargs, equal_nan=False)


def close_complex_nan(a, b, close_func=torch.allclose, accs=(torch.abs, torch.angle), **kwargs):
    return reduce(and_, (
        close_func(acc(a), acc(b), **{**CLOSE_KWARGS, 'equal_nan': True, **kwargs})
        for acc in accs))


def distinct(*args, d_min=1e-3):
    return all(abs(a - b) >= d_min for a, b in combinations(args, 2))
