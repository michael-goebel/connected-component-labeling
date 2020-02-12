from ._hoshen_kopelman_module import _hoshen_kopelman, _get_edge_mtx, _get_enc_mtx
import numpy as np

# Ensures that input is of type int32. This is hard coded into C function.
# Passing a different datatype will give strange results and/or segfault.

def array2ccls(X):
  X = np.ascontiguousarray(X,dtype=np.int32)
  return _hoshen_kopelman(X)

def ccls2adj_mtx(X):
  X = np.ascontiguousarray(X,dtype=np.int32)
  return _get_edge_mtx(X)

def adj_mtx2enc_mtx(X):
  X = np.ascontiguousarray(X,dtype=np.uint8)
  return _get_enc_mtx(X)



class SegAndEnc:
  def __init__(self,X,find_enclosures=True):
    self.input = X
    self.ccls = array2ccls(self.input)
    self.n_ccls = self.ccls.max() + 1
    self.ccl2class = np.empty(self.n_ccls,dtype=np.int32)
    self.ccl2class[self.ccls.reshape(-1)] = self.input.reshape(-1)
    self.volumes = np.bincount(self.ccls.reshape(-1))

    if find_enclosures:
      self.enclosures = True
      self.adj_mtx = ccls2adj_mtx(self.ccls)
      self.enc_mtx = adj_mtx2enc_mtx(self.adj_mtx)
      np.fill_diagonal(self.enc_mtx,0)
      self.ancestors = {i:set(np.nonzero(row)[0]) for i,row in enumerate(self.enc_mtx[:-1].T)}
      self.descendants = {i:set(np.nonzero(row)[0]) for i,row in enumerate(self.enc_mtx[:-1])}

      self.parents = {k:list(v-set().union(*[self.ancestors[vi] for vi in v])) for k,v in self.ancestors.items()}
      self.parents = {k:v[0] for k,v in self.parents.items() if len(v) == 1}

      self.children = {v:set() for v in self.parents.values()}
      for k,v in self.parents.items(): self.children[v].add(k)
    else: self.enclosures = False

    


  def summary(self):
    out = dict()
    out['num classes'] = int(self.input.max()) + 1
    out['num ccls'] = self.n_ccls
    out['ccl to class'] = {i:cls for i,cls in enumerate(self.ccl2class)}
    out['volumes'] = {i:v for i,v in enumerate(self.volumes)}
    if self.enclosures:
      out['ancestors'] = self.ancestors
      out['descendants'] = self.descendants
      out['parents'] = self.parents
      out['children'] = self.children
    return out

