#import sys
#sys.path.append('build/lib.linux-x86_64-3.6/')
import sys
sys.path.append('../')
import seg_module
#import numpy as np


#L = 20
n_dim = 3
n_class = 2
sigma = 3


for L in [16,32,64,128]:
  A = seg_module.synth_map(L,n_dim,n_class,sigma)

  seg_obj = seg_module.SegAndEnc(A)

  E = seg_obj.adj_mtx

#  print(dir(seg_obj))
  print(A.shape)
  print('Parent node dictionary: ',seg_obj.parent_dict)
  print('Child node dictionary: ',seg_obj.child_dict)

  print(seg_obj.n_ccls)
