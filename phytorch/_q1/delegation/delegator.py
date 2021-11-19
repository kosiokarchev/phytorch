import dataclasses
from dataclasses import dataclass
from functools import update_wrapper
from typing import Type

from .delegating import Delegating


@dataclass(eq=False)
class Delegator:
    name: str = dataclasses.field(init=False, default=None)
    func_takes_self: bool = True

    def set_func_takes_self(self, func_takes_self: bool):
        self.func_takes_self = func_takes_self
        return self

    def __set_name__(self, owner, name):
        if self.name is None:
            self.name = name

    def _get(self, func):
        def f(slf: Delegating, *args, **kwargs):
            with slf.delegator_context:
                if self.func_takes_self:
                    args = (slf,) + args
                return func(*args, **kwargs)
        return f

    def __get__(self, instance, owner: Type[Delegating]):
        if owner._is_abstract_delegating:
            return self
        func = getattr(owner._T, self.name)
        ret = update_wrapper(self._get(func), func)
        # print('getting', self.name, 'of', owner._T, 'result', ret)
        setattr(owner, self.name, ret)
        return getattr(instance, self.name) if instance is not None else ret
