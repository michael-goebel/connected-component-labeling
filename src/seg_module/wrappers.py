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
      self.ancestors = {i:set(np.nonzero(row)[0]) for i,row in enumerate(self.enc_mtx[:-1,:-1].T)}
      self.descendants = {i:set(np.nonzero(row)[0]) for i,row in enumerate(self.enc_mtx[:-1,:-1])}

      self.parents = {k:list(v-set().union(*[self.ancestors[vi] for vi in v])) for k,v in self.ancestors.items()}
      self.parents = {k:v[0] for k,v in self.parents.items() if len(v) == 1}

      self.children = {v:set() for v in self.parents.values()}
      for k,v in self.parents.items(): self.children[v].add(k)
    else: self.enclosures = False

  def fill(self,ccl,val=None):
    if ccl in self.parents.keys():
      par = self.parents[ccl]
      for d in self.descendants[ccl]:
        self.parents[d] = par
        self.children[par].add(d)
      if val is None: val = par
      else: print('Error, tried to fill non-enclosed ccl without value to fill'); return None

    mask = (self.ccls == ccl)
    self.input[mask] = self.ccl2class[val]
    self.ccls[mask] = val
    self.volumes[val] += self.volumes[ccl]
    self.volumes[ccl] = 0
    if self.enclosures:
      self.adj_mtx[ccl,:] = 0
      self.adj_mtx[:,ccl] = 0
      self.enc_mtx[ccl,:] = 0
      self.enc_mtx[:,ccl] = 0
      ancestors_mod = self.ancestors.pop(ccl)
      for a in ancestors_mod: self.descendants[a].remove(ccl)
      parent_mod = self.parents.pop(ccl)
      self.children[parent_mod].remove(ccl)


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

