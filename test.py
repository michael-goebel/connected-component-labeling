#import sys
#sys.path.append('build/lib.linux-x86_64-3.6/')
import seg_module
import matplotlib.pyplot as plt
import numpy as np


L = 20
n_dim = 2
n_class = 3
sigma = 1

A = seg_module.synth_map(L,n_dim,n_class,sigma)

seg_obj = seg_module.SegAndEnc(A)

print('Input array:\n',A)
print('CCLs:\n',seg_obj.ccls)

print('Parent node dictionary: ',seg_obj.parent_dict)
print('Child node dictionary: ',seg_obj.child_dict)

E = seg_obj.adj_mtx

colors = seg_module.colors.edge_mtx2colors(E)

fig, axes = plt.subplots(2,2)

gt_colors = seg_module.colors.equispaced_colors(n_class)
axes[0,0].imshow(gt_colors[A])

axes[0,1].imshow(colors[seg_obj.ccls])
axes[1,0].imshow(gt_colors.reshape((1,-1,3)))
axes[1,1].imshow(colors.reshape((1,-1,3)))

axes[1,0].get_yaxis().set_visible(False)
axes[1,1].get_yaxis().set_visible(False)

plt.tight_layout()



plt.show()

