import numpy as np
from scipy.ndimage import gaussian_filter1d


def synth_map(L,n_dim,n_class,sigma):
  unfilt = np.random.randint(0,n_class,(L,)*n_dim)
  one_hot = np.eye(n_class)[unfilt]
  for i in range(n_dim):
    one_hot = gaussian_filter1d(one_hot,sigma,axis=i)

  return np.argmax(one_hot,axis=-1)


if __name__=='__main__':
  print(synth_map(20,2,2,2))
