import sys
sys.path.append('.')
import seg_module
import matplotlib.pyplot as plt
from seg_module.sample_arrays import random_blurred, nested_squares, sample_three

def show_mtx(ax,mtx,cmap):
  h,w = mtx.shape
  C = cmap[mtx]
  ax.imshow(C)
  for i in range(h):
    for j in range(w):
      ax.text(j,i,str(mtx[i,j]),va='center',ha='center')


def one_iter(ax_col,A,index):
  seg_obj = seg_module.SegAndEnc(A)
  n_class = A.max() + 1
  input_colors = seg_module.colors.equispaced_colors(n_class)
  ccls_colors = seg_module.colors.edge_mtx2colors(seg_obj.adj_mtx)
  show_mtx(ax_col[0],A,input_colors)
  show_mtx(ax_col[1],seg_obj.ccls,ccls_colors)
  ax_col[0].set_title(f'Input {index}')
  ax_col[1].set_title(f'CCLs {index}')
  print(f'\n\n===   Input {index}   ===')
  for k,v in seg_obj.summary().items(): print(f'{k}: {v}')
  print('\n\n')

L = 20
n_dim = 2
n_class = 3

test_arrays = [
    random_blurred(L,n_dim,n_class,sigma=1),
    nested_squares(L,n_dim,n_class,width=2),
    sample_three(L,n_dim,n_class,width=2)
]


n_arrays = len(test_arrays)

fig, axes = plt.subplots(2,n_arrays)

for i,A in enumerate(test_arrays):
  one_iter(axes[:,i],A,i+1)

plt.tight_layout()
plt.show()
