import sys
sys.path.append('.')
import seg_module
import matplotlib.pyplot as plt
from seg_module.sample_arrays import random_blurred, nested_squares, sample_three


L = 256
n_dim = 3
n_class = 3
sigma = 10

A = random_blurred(L,n_dim,n_class,sigma)
seg_obj = seg_module.SegAndEnc(A)
sum_dict = seg_obj.summary()
print(f'num ccls: {sum_dict["num ccls"]}')
print(f'num parents: {len(sum_dict["parents"])}')



