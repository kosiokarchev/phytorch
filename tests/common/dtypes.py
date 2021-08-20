import sys
from functools import update_wrapper

import torch
from pytest import fixture, mark


def with_default_double(func):
    def f(*args, **kwargs):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.double)
        ret = func(*args, **kwargs)
        torch.set_default_dtype(default_dtype)
        return ret

    return update_wrapper(f, func)


class BaseDtypeTest:
    dtype: torch.dtype
    cdtype: torch.dtype
    name: str

    @fixture(autouse=True, scope='class')
    def _set_default_dtype(self):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        yield
        torch.set_default_dtype(previous_dtype)

    @fixture(scope='class')
    def eps(self):
        return torch.finfo(self.dtype).eps


class ConcreteDtypeTest(BaseDtypeTest):
    pass


@mark.xfail(reason='8bit\'s dead, baby')
class FloatDtypeTest(ConcreteDtypeTest):
    dtype = torch.float
    cdtype = torch.cfloat
    name = 'Float'


class DoubleDtypeTest(ConcreteDtypeTest):
    dtype = torch.double
    cdtype = torch.cdouble
    name = 'Double'


class AllDtypeTest(BaseDtypeTest):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not issubclass(cls, ConcreteDtypeTest):
            print('dtyping:', cls, cls.__module__)
            subs = make_dtype_tests((cls,), cls.__name__.removesuffix('Test'))
            print('\t->', subs)
            vars(sys.modules[cls.__module__]).update(subs)


def make_dtype_tests(bases, name):
    return {_name: type(_name, bases + (cls,), {})
            for cls in (FloatDtypeTest, DoubleDtypeTest)
            for _name in [f'Test{cls.name}{name}']}
