import inspect
from itertools import repeat

import torch.overrides


def _forall(keys, value):
    return dict(zip(keys, repeat(value)))


# TODO: lazily load testing_overrides in ModuleType.__getattr__
#   can't because "Looking up a name as a module global will bypass module __getattr__"...
testing_overrides = torch.overrides.get_testing_overrides()
my_overrides = {
    torch.rsub: lambda input, other, *, alpha=1: None,
    torch.Tensor.stride: lambda self, dim=None: None,
    torch.std_mean: lambda input, *args, **kwargs: None,
    torch.Tensor.roll: lambda self, shifts, dims=None: None,
    **_forall((torch.rot90, torch.Tensor.rot90),
              lambda input, k=1, dims=(0, 1): None),
    **_forall((torch.std, torch.var, torch.var_mean),
              lambda input, dim=None, unbiased=True, keepdim=False, *, out=None: None),
    torch.Tensor.std: lambda self, dim=None, unbiased=True, keepdim=False: None,
    torch.true_divide: lambda dividend, divisor, *, out=None: None,
    **_forall((torch.index_add, torch.Tensor.index_add),
              lambda self, dim, index, tensor2, *, alpha=1: None),
    **_forall((torch.histogram, torch.Tensor.histogram),
              lambda input, bins=100, *, range=None, weight=None, density=False: None)
}


class SigDict(dict):
    def __missing__(self, func):
        ret = None

        if func in (torch.as_strided, torch.min, torch.max, torch.argmax, torch.all, torch.any,
                    torch.unique, torch.unique_consecutive):
            ret = self[getattr(torch.Tensor, func.__name__)]
        elif func is torch.trapz:
            ret = self[torch.trapezoid]
        elif func is torch.Tensor.put:
            ret = self[torch.put]
        else:
            _func = my_overrides.get(func, func)

            if func.__name__ == '__get__':
                _func = lambda self: None

            try:
                ret = inspect.signature(_func)
            except ValueError as e:
                if not e.args[0].startswith('no signature found'):
                    raise e from None

                try:
                    if (doc := getattr(func, '__doc__', None)) is not None:
                        sig = doc.split('\n', 2)[1]
                        if '(' in sig:
                            name, sig = sig.split('(', 1)
                            name = name.rsplit('.', 1)[-1]
                            sig = sig.split('->')[0]

                            if (oc := getattr(func, '__objclass__', None)) is not None and issubclass(torch.Tensor, oc):
                                sig = 'self, ' + sig

                            exec(f'def {name}({sig}: pass', {'torch': torch}, l := {})
                            ret = inspect.signature(next(iter(l.values())))
                except ValueError as e:
                    if not e.args[0].endswith('has invalid signature'):
                        raise e from None
                except SyntaxError:
                    pass

        if ret is None:
            ret = inspect.signature(testing_overrides[func])

        self[func] = ret
        return ret

    def __contains__(self, item):
        try:
            self.__getitem__(item)
            return True
        except KeyError:
            return True

    _nodefault = object()

    def get(self, key, default=_nodefault):
        if key in self:
            return self[key]
        if default is not self._nodefault:
            return default
        raise KeyError(key)


sigdict = SigDict()


# testing overrides for torch.pow, Tensor.pow and Tensor.__pow__ are the same object, so
# Tensor methods include out=None, but screw it
# sigdict[torch.Tensor.__pow__] = sigdict[torch.Tensor.pow]

# TODO: report bugs
#  - not in funcs:
#    - torch.as_strided
#  - errors in signature in docs:
#    - (torch.min, torch.max, torch.argmax, torch.trapz): optional args missing
#    - (Tensor.roll, Tensor.stride): dim is optional
#    - (torch.rot90, Tensor.rot90): k, dim optional
#    - torch.full_like
#    - torch.randint_like
#    - (torch, Tensor).(std, var), torch.(std_mean, var_mean)): wrong
#    - torch.true_div: out optional
#    - torch.rsub: alpha keyword-only
#    - Tensor.put, (torch, Tensor).histogram: has input parameter
#    - torch.lu: has (*args, **kwargs)
#    - Tensor.index_add: missing alpha
#    - (torch, Tensor).histogram: bins is optional
#    - torch.(unique, unique_consecutive): actual signature is (*args, **kwargs)
#  - missing in get_overridable_functions()
#    - torch.as_strided
#    - Tensor.new_*

