#import sys
#sys.path.append('build/lib.linux-x86_64-3.6/')
import sys
sys.path.append('.')
#sys.path.append('../')
import seg_module
#import numpy as np


L = 20
n_dim = 2
n_class = 3
width = 3

A = seg_module.nested_squares(L,n_dim,n_class,width)

seg_obj = seg_module.SegAndEnc(A)

E = seg_obj.adj_mtx

print(A)
print(E)
print(seg_obj.ccls)

#  E = seg_obj.adj_mtx

#  print(dir(seg_obj))
#  print(A.shape)
print('Ancestor nodes: ',seg_obj.ancestors)
print('Descendant nodes: ',seg_obj.descendants)
print('


#an = seg_obj.ancestors
#des = seg_obj.descendants

#par = seg_obj.parents

#children = {v:set() for v in par.values()}
#for k,v in par.items(): children[v].add(k)

#print(children)

#print(seg_obj.parents)

#parent = {k:(v-set().union(*[an[vi] for vi in v])) for k,v in an.items()}
#print(parent)

#parent = {k:   for k,v in an.items()}
#print(parent)

#for k,v in an.items():
#  print(set().union(*[an[v_i] for v_i in v]))

#  for v_i in v:
#    print(k,an[v_i])

#  print(k,(v - set().union({an[v_i] for v_i in v})))
#  print(k,v)


