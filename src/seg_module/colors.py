import numpy as np
from colorsys import hsv_to_rgb

a = 1/6

def _furthest(x):
  if x.shape[0] == 0:
    return np.random.uniform()
  elif x.shape[0] == 1:
    return np.random.uniform(x[0]+a,x[0]+(1-a)) % 1
  else:
    x = np.sort(x)
    d = np.diff(x,append=x[0]+1)
    ind = np.argmax(d)
    return (d[ind]/2 + x[ind]) % 1

def edge_mtx2hues(E):
  E = E[:-1,:-1]
  s = E.sum(1)
  l = np.argsort(-s)
  n = l.shape[0]
  colors = np.zeros(n)
  visited = np.zeros(n)
  for i in l:
    neighs = E[i]
    used_colors = colors[np.nonzero(visited*neighs)[0]]
    colors[i] = _furthest(used_colors)
    visited[i] = 1
  return colors

def edge_mtx2colors(E):
  hues = edge_mtx2hues(E)
  colors = (255*np.array([hsv_to_rgb(h,1,1) for h in hues])).astype(int)
  return colors


def equispaced_colors(n):
  return (255*np.array([hsv_to_rgb(i/n,1,1) for i in range(n)])).astype(int)
