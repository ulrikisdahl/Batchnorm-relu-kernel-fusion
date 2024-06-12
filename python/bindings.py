from setuptools import setup, Extension 
from torch.utils import cpp_extension

setup(name="res_cpp",
      ext_modules=[cpp_extension.CppExtension("res_cpp", ["res.cpp"])], #CppExtension is a wrapper that sets the language of the extension to C++
      cmdclass={"build_ext": cpp_extension.BuildExtension}) #BuildExtension manages mixed compilation (C++/CUDA)





