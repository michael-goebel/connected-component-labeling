from distutils.core import setup, Extension
import numpy as np

c_module = Extension('seg_module.hoshen_kopelman_module', sources=['src/seg_module/seg.c'],
    include_dirs=[np.get_include()])

setup(name='seg_module',
      version='1.0',
      packages=['seg_module'],
      package_dir={'seg_module':'src/seg_module/'},
      ext_modules=[c_module],
      )





