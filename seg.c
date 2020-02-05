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

// Everything below this is boilerplate to register module in python
static PyMethodDef myMethods[] = {
  {"_hoshen_kopelman", _hk, METH_VARARGS, "Private hoshen-kopelman interface. Only accepts int32."},
  {NULL,NULL,0,NULL}
};

static struct PyModuleDef cModPyDem = {
  PyModuleDef_HEAD_INIT,
  "hoshen_kopelman_module", "C implementation of hoshen-kopelman algorithm for numpy.",
  -1,
  myMethods
};

PyMODINIT_FUNC PyInit_hoshen_kopelman_module(void) {
  PyObject *module;
  module = PyModule_Create(&cModPyDem);
  import_array();
  return module;
}


