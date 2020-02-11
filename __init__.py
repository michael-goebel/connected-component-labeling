# The __init__.py file allows for importing a directory as a module in python.
# To use this module, import this file's parent directory.

from .hoshen_kopelman_module import _hoshen_kopelman
import numpy as np

# Ensures that input is of type int32 and contiguous. This is hard coded into C function.
# Passing a different datatype will give strange results and/or segfault.
def connected_comps(X):
  X = np.ascontiguousarray(X,dtype=np.int32)
  Y = _hoshen_kopelman(X)
  return Y


