import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_version(version_file):
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


os.chdir(osp.dirname(osp.abspath(__file__)))
csrc_directory = osp.join('ailut', 'csrc')

setup(
    name='ailut',
    version=get_version(osp.join('ailut', 'version.py')),
    description='Adaptive Interval 3D LookUp Table Transform',
    author='Charles',
    author_email='charles.young@sjtu.edu.cn',
    packages=find_packages(),

    ext_modules=[
        CUDAExtension(
            name='ailut._ext',
            sources=[
                osp.join(csrc_directory, 'ailut_transform.cpp'),
                osp.join(csrc_directory, 'ailut_transform_cpu.cpp'),
                osp.join(csrc_directory, 'ailut_transform_cuda.cu')
            ],
            extra_compile_args={
                'cxx': [
                    '-O2',
                    '-fPIC'
                ],
                'nvcc': [
                    '-D__int128=long long',    # Fix Ubuntu24 gcc headers
                    '--expt-relaxed-constexpr',
                    '--compiler-options=-fPIC',
                    '-I/usr/local/cuda-10.2/include',        # Force CUDA headers
                    '-I/usr/local/cuda-10.2/nvvm/libdevice'  # Make libdevice visible
                ]
            }
        )
    ],

    cmdclass={'build_ext': BuildExtension},
    license='Apache License 2.0',
    zip_safe=False
)
