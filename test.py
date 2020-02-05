import numpy as np
from time import time
import seg_module

# ccl = Connected Component Labels

# whether or not to print arrays
verbose = 0

# define parameters for random test array, then create array
n_class = 4
L = 256
n_dims = 3
A = np.random.randint(0,n_class,(L,)*n_dims,dtype=np.int32)


t1 = time()
# call function to get ccls
B = seg_module.connected_comps(A)
t2 = time()


# now that we have ccls, compute a few example stats

if verbose:
  print(A)
  print(B)

# get number of pixels in each ccl
sizes = np.bincount(B.reshape(-1))
n_ccl = sizes.shape[0]

print(f'Input shape = {A.shape}')
print(f'Num of class = {n_class}')
print(f'Runtime: {t2-t1:0.3f} secs\n')

print(f'Num of clusters = {n_ccl}')
print(f'Cluster avg size = {np.mean(sizes):0.2f}')
print(f'Cluster max size = {np.max(sizes)}\n')

# create mapping from ccl to class labels
ccl2label = np.empty(n_ccl,dtype=int)
ccl2label[B.reshape(-1)] = A.reshape(-1)

# For each class label, find the largest connected component
for i in range(n_class):
  ccl_max = np.argmax(sizes * (ccl2label==i))
  print(f'Max cluster for {i}: ccl = {ccl_max}, size = {sizes[ccl_max]}')

print(str())
