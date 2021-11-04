from functools import partial
from itertools import chain

from more_itertools import consume, first, unique_everseen

from ._prefixes import _generic_prefix, _prefix_map
from ..utils.scoping import AllScope


_all_scope = AllScope()

consume(globals().__setitem__(f'{name}_', partial(_generic_prefix, first(abbrevs), expon))
        for (names, abbrevs), expon in _prefix_map.items()
        for name in unique_everseen(chain(names, abbrevs)))

__all__ = tuple(filter(lambda name: not name.startswith('_'), _all_scope.__all__))
