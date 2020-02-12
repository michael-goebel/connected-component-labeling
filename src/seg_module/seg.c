#include <stdio.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>


int prod(int * v, int L){    // computes product of elements in array
  int out = 1;
  for(int i=0; i<L; i++){ out *= v[i]; }
  return out;
}

int min(int * v, int L){    // computes minimum of elements in array
  int out = v[0];
  for(int i=1; i<L; i++){
    if(v[i] < out){out = v[i];}
  }
  return out;
}


int max(int * v, int L){    // computes maximum of elements in array
  int out = v[0];
  for(int i=1; i<L; i++){
    if(v[i] > out){out = v[i];}
  }
  return out;
}


// Given sparse labels in the range [0,L), relabel with values in
// the range [0,n_label) so that there are no skipped values and
// order is maintained.
void relabel(int *v, int L){
  int* labels = (int*)malloc(L*sizeof(int));
  for(int i=0;i<L;i++){ labels[i] = 0;}
  for(int i=0;i<L;i++){ labels[v[i]] = 1;}
  for(int i=1;i<L;i++){ labels[i] += labels[i-1];}
  for(int i=0;i<L;i++){ v[i] = labels[v[i]] - 1; }
  free(labels);
}


// The algorithm creates a directional "tree" of references when it
// merges two labels. The structure of the tree is unimportant, only the
// final value that each node points to is needed. So if a value is not a node,
// see where it points (1), then reassign all points along the path to also point
// to that leaf (2) to make access next time faster
int find(int x, int *labels){
  int z; int y = x;
  while(labels[y] != y){y = labels[y];}  // (1)
  while(labels[x] != x){                 // (2)
    z = labels[x]; labels[x] = y; x = z;
  }
  return y;
}

// Convert flat index into list of coordinates (like numpy.unravel_index)
void ind_unravel(int ind_in, int * ind_out, int * dims, int n_dims){
  int prod = 1;
  for(int i=n_dims-1; i>=0; i--){
    ind_out[i] = (ind_in/prod) % dims[i];
    prod *= dims[i];
  }
}

// Convert list of coordinates to flat index (like numpy.ravel_multi_index)
int ind_ravel(int * ind_in, int * dims, int n_dims){
  int out = 0; int prod = 1;
  for(int i=n_dims-1; i>=0; i--){
    out += prod*ind_in[i];
    prod *= dims[i];
  }
  return out;
}

// Given a flat index, find all neighbors with lesser flat indices, and
// return these flat indices as a list. Edge cases will make the number of
// neighbors variable, so number of valid neighbors is returned.
int get_neighbors(int ind_in, int * out_list, int *dims, int n_dims){
  int ind_in_ur[n_dims]; int n_neigh = 0;
  ind_unravel(ind_in, ind_in_ur, dims, n_dims);
  for(int i=0; i<n_dims; i++){
    if(ind_in_ur[i] != 0){
      ind_in_ur[i] -= 1;
      out_list[n_neigh] = ind_ravel(ind_in_ur,dims,n_dims);
      ind_in_ur[i] += 1;
      n_neigh += 1;
    }
  }
  return n_neigh;
}

char is_r_bound(int ind_in, int *dims, int n_dims){
  char val = 0;
  int ind_ur[n_dims];
  ind_unravel(ind_in,ind_ur,dims,n_dims);
  for(int i=0; i<n_dims; i++){
    if(ind_ur[i]==dims[i]-1){ val=1; }
  }
  return val;
}


// Debugging function to print array in C.
void print_arr(int * v, int * dims, char ** delims, int n_dims){
  int prods[n_dims]; prods[n_dims-1] = 1;
  for(int i=n_dims-1; i>0; i--){prods[i-1] = prods[i]*dims[i];}
  for(int i=0; i<prods[0]*dims[0]; i++){
    for(int j=0; j<n_dims; j++){
      if(i % prods[j] == 0){printf("%s",delims[n_dims-j-1]); break;}
    }
    printf("%2d",v[i]);
  }
  printf("\n\n");
}

// Debugging function to print vector in C.
void print_vec(int *v, int L){
  char *delims[] = {" "}; int dims[] = {L};
  print_arr(v,dims,delims,1);
}

// Hoshen-Kopelman algorithm. Two pass algorithm to find CLLs.
void hoshen_kopelman(int *v, int *out, int *dims, int n_dims){
  int i, j, n_neigh, n_match;
  int len = prod(dims,n_dims);
  int neigh[n_dims]; int vals[n_dims];

  for(i=0; i<len; i++){                              // first pass
    out[i] = i; n_match = 0;
    n_neigh = get_neighbors(i,neigh,dims,n_dims);    // put neighbor inds in neigh, and get len(neigh)
    for(j=0;j<n_neigh;j++){                          // for each neighbor
      if(v[i]==v[neigh[j]]){                         // check if neighbor matches pixel of interest
        vals[n_match] = find(neigh[j],out);          // search tree for the root (smallest) cll
        n_match += 1;
      }
    }
    if(n_match > 0){ out[i] = min(vals,n_match); }   // set pixel of interest to smallest equiv cll
    for(j=0;j<n_match;j++){ out[vals[j]] = out[i]; } // set all neighbors to smallest equiv cll
  }
  // second pass
  // After the first pass is done, iterate through all output values / cll tree labels to ensure that
  // no labels reference other labels as being equiv
  for(i=0;i<len;i++){ find(i,out); }
  relabel(out,len);  // run relabel on clls to de-sparse
}


void edge_mtx(int *l, char *mtx, int *dims, int n_dims){
  int len = prod(dims,n_dims);
  int n_label = max(l,len) + 1;
  int L = n_label + 1;
  int i, j;
  int neigh[n_dims];
  int n_neigh;

  for(i=0;i<L*L;i++){ mtx[i] = 0; }
  for(i=0;i<len;i++){
    n_neigh = get_neighbors(i,neigh,dims,n_dims);
    for(j=0;j<n_neigh;j++){
      mtx[l[i] + L*l[neigh[j]]] = 1;
      mtx[l[neigh[j]] + L*l[i]] = 1;
    }
    if((n_neigh < n_dims) || is_r_bound(i,dims,n_dims)){
      mtx[l[i] + L*(L-1)] = 1;
      mtx[(L-1) + L*l[i]] = 1;
    }
  }
}


void enc_step(char *mtx, int L, int ind, char* visited){
  visited[ind] = 1;
  for(int i=0;i<L;i++){
    if(mtx[ind+L*i]==1 && visited[i]==0){
      enc_step(mtx,L,i,visited);
    }
  }
}


void get_enclosures(char *children, char *mtx, int L){

  int i, j, k;
  char * visited = (char *) malloc(L*sizeof(char));
  for(i=0;i<L*L;i++){ children[i] = 0; }

  for(i=0;i<L;i++){
    for(j=0;j<L;j++){ visited[j] = 0; }
    visited[i] = 1;
    for(j=0;j<L;j++){
      if(mtx[i*L+j] & (1-visited[j])){
        enc_step(mtx,L,j,visited);
        if(visited[L-1]){
          for(k=0;k<L;k++){
            children[L*i+k] ^= (visited[k] ^ 1);
          }
          break;
        } else {
          for(k=0;k<L;k++){
            children[L*i+k] |= visited[k];
          }
        }
      }
    }
  }
  for(i=0;i<L-1;i++){
    children[L*(L-1)+i] = 1;
  }
}



// All code below this line is used to join the above C code with python

// C function for h-k alg. Takes in numpy array instead of C array.
static PyObject* _hk(PyObject* self, PyObject* args) {
  PyArrayObject *in_array;
  PyObject *out_array;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array)){return NULL;}  // boilerplate code to parse out input

  npy_intp * dims = PyArray_DIMS(in_array);    // get dims of input
  int n_dims = PyArray_NDIM(in_array);         // get number of dimensions
  int dims_int[n_dims];                        // next two lines convert dims array from npy_intp to int
  for(int i=0; i<n_dims; i++){ dims_int[i] = (int) dims[i];}

  out_array = PyArray_SimpleNew(n_dims,dims,NPY_INT);  // initialize output array

  int * in_data = (int *) PyArray_DATA(in_array);      // get pointer to data for each array
  int * out_data = (int *) PyArray_DATA((PyArrayObject *) out_array);

  hoshen_kopelman(in_data, out_data, dims_int, n_dims);  // run alg
  Py_INCREF(out_array);                                  // make REFCOUNT for array non-zero, so it is not deleted by garbage collector
  return out_array;
}

static PyObject* _edge_mtx(PyObject* self, PyObject* args) {
  PyArrayObject *in_array;
  PyObject *out_array;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array)){return NULL;}

  npy_intp * dims = PyArray_DIMS(in_array);    // get dims of input
  int n_dims = PyArray_NDIM(in_array);         // get number of dimensions
  int dims_int[n_dims];                        // next two lines convert dims array from npy_intp to int
  for(int i=0; i<n_dims; i++){ dims_int[i] = (int) dims[i];}
  int len = prod(dims_int,n_dims);

  int * in_data = (int *) PyArray_DATA(in_array);
  int L = max(in_data,len) + 2;
  npy_intp out_dims[] = {L,L};

  out_array = PyArray_SimpleNew(2,out_dims,NPY_UINT8);
  char * out_data = (char *) PyArray_DATA((PyArrayObject *) out_array);
  edge_mtx(in_data, out_data, dims_int, n_dims);
  Py_INCREF(out_array);
  return out_array;
}

static PyObject* _get_enclosures(PyObject* self, PyObject* args) {
  PyArrayObject *in_array;
  PyObject *out_array;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array)){return NULL;}

  npy_intp *dims = PyArray_DIMS(in_array);    // get dims of input
  int L = (int) dims[0];
  out_array = PyArray_NewLikeArray(in_array,NPY_ANYORDER,NULL,0);
  char * in_data = (char *) PyArray_DATA(in_array);
  char * out_data = (char *) PyArray_DATA((PyArrayObject *) out_array);
  get_enclosures(out_data,in_data,L);
  Py_INCREF(out_array);
  return out_array;
}


// Everything below this is boilerplate to register module in python
static PyMethodDef myMethods[] = {
  {"_hoshen_kopelman", _hk, METH_VARARGS, "Private hoshen-kopelman interface. Only accepts int32."},
  {"_get_edge_mtx", _edge_mtx, METH_VARARGS, "Private edge matrix function."},
  {"_get_enc_mtx", _get_enclosures, METH_VARARGS, "Private function to find enclosing labels"},
  {NULL,NULL,0,NULL}
};

static struct PyModuleDef cModPyDem = {
  PyModuleDef_HEAD_INIT,
  "_hoshen_kopelman_module", "C implementation of hoshen-kopelman algorithm for numpy.",
  -1,
  myMethods
};

PyMODINIT_FUNC PyInit__hoshen_kopelman_module(void) {
  PyObject *module;
  module = PyModule_Create(&cModPyDem);
  import_array();
  return module;
}


