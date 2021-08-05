import os
import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, _is_cuda_file

# That's for torch, need to be set in the terminal as well...
os.environ['CXX'] = 'g++-8'
os.environ['CUDA_HOME'] = '/usr/local/cuda'


class MyBuildExtension(BuildExtension):
    def build_extension(self, ext: CUDAExtension):
        _compile = self.compiler.compile

        def patched_compile(sources, output_dir=None, *args, **kwargs):
            objects = _compile(sources, output_dir=output_dir, *args, **kwargs)
            cuda_objects = [obj for src, obj in zip(sources, objects)
                            if _is_cuda_file(src)]
            if cuda_objects:
                cuda_linked_object = os.path.join(os.path.dirname(
                    cuda_objects[0]), ext.name+'_device_linked.o')
                cuda_link = ['nvcc', '--device-link', '--forward-unknown-to-host-compiler', '-fPIC'] + cuda_objects + ['-o', cuda_linked_object]

                print('Linking device code')
                print(' '.join(cuda_link))
                subprocess.run(cuda_link, stdout=1, stderr=subprocess.PIPE, cwd=output_dir, check=True)

                objects.append(cuda_linked_object)

            return objects

        self.compiler.compile = patched_compile
        ret = super().build_extension(ext)
        self.compiler.compile = _compile
        return ret


setup(
    name='phytorch.extensions',
    ext_modules=[
        CUDAExtension(folder, [
            str(_) for _ in filter(lambda _: (_.suffix.lower() in ('.cpp', '.cu') and not _.name.startswith('_')), Path(folder).iterdir())
        ], extra_compile_args={'nvcc': ['--expt-relaxed-constexpr', '--extended-lambda', '--relocatable-device-code=true', '--gpu-architecture=sm_30']})
        for folder in (
            'elliptic', 'roots', 'special',
        )
    ],
    cmdclass={'build_ext': MyBuildExtension}
)
