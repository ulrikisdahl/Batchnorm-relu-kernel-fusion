from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='binder',
      ext_modules=[cpp_extension.CppExtension('binder', ['binder.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})