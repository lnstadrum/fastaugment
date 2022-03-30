from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from os import path

SRC_FOLDER = path.realpath(path.join(path.dirname(__file__), "..", "src"))

setup(
    name='fast_augment_torch',
    ext_modules=[
        CUDAExtension('_fast_augment_torch_lib', [
            path.join(SRC_FOLDER, "pytorch.cpp"),
            path.join(SRC_FOLDER, "augment.cu")
        ])
    ],
    packages=['fast_augment_torch'],
    cmdclass={
        'build_ext': BuildExtension
    }
)