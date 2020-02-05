from .hoshen_kopelman_module import _hoshen_kopelman
import numpy as np

# Ensures that input is of type int32. This is hard coded into C function.
# Passing a different datatype will give strange results and/or segfault.
def connected_comps(X):
  X = X.astype(np.int32)
  Y = _hoshen_kopelman(X)
  return Y


