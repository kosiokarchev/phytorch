import inspect
from _warnings import warn
from collections import UserDict
from contextlib import AbstractContextManager
from functools import partial
from itertools import starmap
from typing import Any, MutableMapping

from more_itertools import consume


class OverwrittenWarning(Warning):
    pass


class AbstractScope(UserDict):
    def __init__(self, context: MutableMapping[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.context = context if context is not None else inspect.stack()[1].frame.f_locals


class GlobalScope(AbstractScope, AbstractContextManager):
    active = True

    def register(self, name, val, ignore_if_exists=False):
        if name in self.data:
            if ignore_if_exists:
                return
            warn(f'"{name}" already defined in "{self.context}"', OverwrittenWarning)
        self.data[name] = val
        return self

    def register_many(self, ignore_if_exists=False, **kwargs):
        consume(starmap(partial(self.register, ignore_if_exists=ignore_if_exists), kwargs.items()))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()
        self.active = False

    def _del(self):
        self.context.clear()

    def __del__(self):
        if self.active:
            self._del()
            self.context.update(self)


class InitKeysScope(AbstractScope):
    def __init__(self, context=None, **kwargs):
        super().__init__(context if context is not None else inspect.stack()[1].frame.f_locals, **kwargs)
        self.init_keys = tuple(self.context.keys())


class AutoCleanupGlobalScope(InitKeysScope, GlobalScope):
    def _del(self):
        for key in self.init_keys:
            self.context.pop(key, None)


class AllScope(InitKeysScope):
    @property
    def __all__(self):
        return tuple(filter(lambda key: key not in self.init_keys, self.context.keys()))
