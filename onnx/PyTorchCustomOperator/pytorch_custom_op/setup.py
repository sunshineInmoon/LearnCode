from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='custom_group_norm',
      ext_modules=[cpp_extension.CppExtension('custom_group_norm', ['custom_group_norm.cpp'],
                                             include_dirs = ['/home/zhanghua/Tools/eigen'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
