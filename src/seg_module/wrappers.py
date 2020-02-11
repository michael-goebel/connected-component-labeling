from .hoshen_kopelman_module import _hoshen_kopelman, _get_edge_mtx, _get_enc_mtx
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

    if find_enclosures:
      self.adj_mtx = ccls2adj_mtx(self.ccls)
      self.enc_mtx = adj_mtx2enc_mtx(self.adj_mtx)
      self.child_dict = {i:np.nonzero(row)[0] for i,row in enumerate(self.enc_mtx[:-1]) if np.sum(row)>0}
      self.parent_dict = {i:np.nonzero(row)[0] for i,row in enumerate(self.enc_mtx[:-1].T) if np.sum(row)>0}


#  def enc_mtx(self):
#    if self.enc_mtx_v is None:
#      self.enc_mtx_v = adj_mtx2enc_mtx(self.adj_mtx())
#      self.chld_dict = {i:np.nonzero(row)[0] for i,row in enumerate(self.enc_mtx()[:-1]) if np.sum(row)>0}
#      self.prnt_dict = {i:np.nonzero(row)[0] for i,row in enumerate(self.enc_mtx()[:-1].T) if np.sum(row)>0}


#      self.chld_dict = {(i+1)%(self.n_labels+1)-1:np.nonzero(row)[0] for i,row in enumerate(self.enc_mtx()) if np.sum(row)>0}
#      self.prnt_dict = {(i+1)%(self.n_labels+1)-1:np.nonzero(row)[0] for i,row in enumerate(self.enc_mtx().T) if np.sum(row)>0}
#    return self.enc_mtx_v





#  def enc_dict(self):
#    return {(i+1)%(self.n_labels+1)-1:np.nonzero(row)[0] for i,row in enumerate(self.enc_mtx()) if np.sum(row)>0}

  



#  def fill(self,label,new_label):
#    L = self.labels()
#    X[L==label] = new_cls
#    



#def get_enc_dict(ccls):
#  enc_mtx = get_enc_mtx(ccls)
#  output = {i:np.nonzero(row)[0] for i, row in enumerate(enc_mtx)}
#  return output



#def enc_mtx2enc_dict(X):
#  X = np.ascontiguousarray(X,dtype=np.uint8)
#  return {i:np.non


#def connected_comps(X):
#  X = np.ascontiguousarray(X,dtype=np.int32)
#  Y = _hoshen_kopelman(X)
#  return Y

#def get_edge_mtx(ccls):
#  ccls = np.ascontiguousarray(ccls,dtype=np.int32)
#  edge_mtx = _get_edge_mtx(ccls)
#  return edge_mtx

#def get_enc_mtx(ccls):
#  edge_mtx = get_edge_mtx(ccls)
#  enc_mtx = _get_enc_mtx(edge_mtx)
#  return enc_mtx

#def get_enc_dict(ccls):
#  enc_mtx = get_enc_mtx(ccls)
#  output = {i:np.nonzero(row)[0] for i, row in enumerate(enc_mtx)}
#  return output
