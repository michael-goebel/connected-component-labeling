import sys
sys.path.append('.')
import seg_module
print(dir(seg_module))

L = 20
n_dim = 2
n_class = 3
width = 3

A = seg_module.sample_arrays.nested_squares(L,n_dim,n_class,width)

seg_obj = seg_module.SegAndEnc(A)

E = seg_obj.adj_mtx

print('Input:\n',A)
print('\nCCLs:\n',seg_obj.ccls)


for k,v in seg_obj.summary().items(): print(f'{k}: {v}')
