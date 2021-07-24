import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# That's for torch, need to be set in the terminal as well...
os.environ['CXX'] = 'g++-8'
os.environ['CUDA_HOME'] = '/usr/local/cuda'

setup(
    name='phytorch.extensions',
    ext_modules=[
        CUDAExtension(folder, [str(_) for _ in filter(lambda _: _.suffix.lower() in ('.cpp', '.cu'), Path(folder).iterdir())],
                      extra_compile_args={'nvcc': ['--expt-relaxed-constexpr', '--extended-lambda']})
        for folder in ('ellipr', 'roots',)
    ],
    cmdclass={'build_ext': BuildExtension}
)
