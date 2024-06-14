from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='binder',
    ext_modules=[
        CUDAExtension(
            name='binder',
            sources=['src/binder.cpp', 'src/fused_kernel_forward.cu'],
            include_dirs=['/ulrik/home/libtorch/include', '/ulrik/home/libtorch/include/torch/csrc/api/include', "/usr/local/cuda/include"],
            library_dirs=['/ulrik/home/libtorch/lib'],
            libraries=['c10', 'torch', 'torch_cpu', 'torch_cuda']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

#python3 setup.py build_ext --inplace