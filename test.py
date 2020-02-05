import numpy as np
import seg_module
from time import time


n_class = 3
L = 128
n_dims = 3
verbose = 0

A = np.random.randint(0,n_class,(L,)*n_dims,dtype=np.int32)
t1 = time()
B = seg_module.connected_comps(A)
t2 = time()

if verbose:
  print(A)
  print(B)


sizes = np.bincount(B.reshape(-1))
n_ccl = sizes.shape[0]

print(f'Input shape = {A.shape}')
print(f'Num of class = {n_class}')
print(f'Runtime: {t2-t1:0.3f}\n')


print(f'Num of clusters = {n_ccl}')
print(f'Cluster avg size = {np.mean(sizes):0.2f}')
print(f'Cluster max size = {np.max(sizes)}\n')

ccl2label = np.empty(n_ccl,dtype=int)
ccl2label[B.reshape(-1)] = A.reshape(-1)

for i in range(n_class):
  ccl_max = np.argmax(sizes * (ccl2label==i))
  print(f'Max cluster for {i}: ccl = {ccl_max}, size = {sizes[ccl_max]}')

print(str())
