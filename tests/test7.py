import sys
sys.path.append('.')
import seg_module
import matplotlib.pyplot as plt
from seg_module.sample_arrays import random_blurred, nested_squares, sample_three


L = 30
n_dim = 2
n_class = 3


#A = random_blurred(L,n_dim,n_class,sigma)
A = sample_three(L,n_dim,n_class,width=2)
seg_obj = seg_module.SegAndEnc(A)
print(A)
print(seg_obj.ccls)
print(seg_obj.adj_mtx.shape)

for k,v in seg_obj.summary().items(): print(f'{k}: {v}')

#for ccl, anc in seg_obj.ancestors.items():
#  if len(anc) > 2:
#    print(f'num: {ccl}')
#    seg_obj.fill(ccl)

inds_remove = [ccl for ccl, anc in seg_obj.ancestors.items() if len(anc)>2]

for ccl in inds_remove:
  seg_obj.fill(ccl)


for k,v in seg_obj.summary().items(): print(f'{k}: {v}')



#print(A)
#print(seg_obj.ccls)

#sum_dict = seg_obj.summary()
#print(f'num ccls: {sum_dict["num ccls"]}')
#print(f'num parents: {len(sum_dict["parents"])}')
#print(sum_dict)


#print(A)
