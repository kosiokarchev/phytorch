# import needed to dynamically load extensions
# noinspection PyUnresolvedReferences
import torch


def __getattr__(name):
    import sys
    from importlib import import_module
    from importlib.machinery import SourceFileLoader
    from importlib.util import module_from_spec, spec_from_loader
    from pathlib import Path
    from warnings import warn

    modname = f'{__package__}.{name}'
    try:
        ret = import_module(modname)
    except ImportError:
        fname = (Path(__file__).parent / name).with_suffix('.pyi')
        if not fname.is_file():
            raise AttributeError(f'module {__name__} has no attribute {name}')

        warn(f'Loading dummy {modname} from {fname!s}. Please compile extensions.', ImportWarning)
        # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        # but as a one-liner :)
        (loader := SourceFileLoader(modname, str(fname))).exec_module(ret := module_from_spec(spec_from_loader(modname, loader)))

    globals()[name] = sys.modules[name] = ret
    return ret

