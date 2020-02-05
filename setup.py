from distutils.core import setup, Extension
import numpy as np

my_module = Extension('hoshen_kopelman_module', sources=['seg.c'],
    include_dirs=[np.get_include()])

setup(py_modules=['__init__'],ext_modules=[my_module])
