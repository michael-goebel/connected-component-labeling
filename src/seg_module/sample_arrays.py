import numpy as np
from scipy.ndimage import gaussian_filter1d


def random_map(L,n_dim,n_class,sigma):
  unfilt = np.random.randint(0,n_class,(L,)*n_dim)
  one_hot = np.eye(n_class)[unfilt]
  for i in range(n_dim):
    one_hot = gaussian_filter1d(one_hot,sigma,axis=i)

  return np.argmax(one_hot,axis=-1)

def nested_squares(L,n_dim,n_class,width):
  one_dim = (np.abs(np.arange(L)-L/2)//width).astype(int)
  out_shape = (L,)*n_dim
  A = np.zeros(out_shape,dtype=int)
  for i in range(n_dim):
    A = np.swapaxes(A,0,i).reshape((L,-1))
    A = np.maximum(A,one_dim[:,None])
    A.reshape(out_shape)
  return A % n_class


if __name__=='__main__':
  print(random_map(20,2,2,2))
  print(nested_squares(16,2,3,2))
