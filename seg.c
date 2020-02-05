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

void relabel(int *v, int L){
  int* labels = (int*)malloc(L*sizeof(int));
  for(int i=0;i<L;i++){ labels[i] = 0;}
  for(int i=0;i<L;i++){ labels[v[i]] = 1;}
  for(int i=1;i<L;i++){ labels[i] += labels[i-1];}
  for(int i=0;i<L;i++){ v[i] = labels[v[i]] - 1; }
  free(labels);
}

int find(int x, int *labels){
  int z; int y = x;
  while(labels[y] != y){y = labels[y];}
  while(labels[x] != x){
    z = labels[x]; labels[x] = y; x = z;
  }
  return y;
}


void ind_unravel(int ind_in, int * ind_out, int * dims, int n_dims){
  int prod = 1;
  for(int i=n_dims-1; i>=0; i--){
    ind_out[i] = (ind_in/prod) % dims[i];
    prod *= dims[i];
  }
}

int ind_ravel(int * ind_in, int * dims, int n_dims){
  int out = 0; int prod = 1;
  for(int i=n_dims-1; i>=0; i--){
    out += prod*ind_in[i];
    prod *= dims[i];
  }
  return out;
}

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

void print_vec(int *v, int L){
  char *delims[] = {" "}; int dims[] = {L};
  print_arr(v,dims,delims,1);
}


void hoshen_kopelman(int *v, int *out, int *dims, int n_dims){
  int i, j, n_neigh, n_match;
  int len = prod(dims,n_dims);
  int neigh[n_dims]; int vals[n_dims];

  for(i=0; i<len; i++){
    out[i] = i; n_match = 0;
    n_neigh = get_neighbors(i,neigh,dims,n_dims);
    for(j=0;j<n_neigh;j++){
      if(v[i]==v[neigh[j]]){
        vals[n_match] = find(neigh[j],out);
        n_match += 1;
      }
    }
    if(n_match > 0){ out[i] = min(vals,n_match); }
    for(j=0;j<n_match;j++){ out[vals[j]] = out[i]; }
  }
  for(i=0;i<len;i++){ find(i,out); }
  relabel(out,len);
}


static PyObject* _hk(PyObject* self, PyObject* args) {
  PyArrayObject *in_array;
  PyObject *out_array;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array)){return NULL;}

  npy_intp * dims = PyArray_DIMS(in_array);
  int n_dims = PyArray_NDIM(in_array);
  int dims_int[n_dims];
  for(int i=0; i<n_dims; i++){ dims_int[i] = (int) dims[i];}

  out_array = PyArray_SimpleNew(n_dims,dims,NPY_INT);

  int * in_data = (int *) PyArray_DATA(in_array);
  int * out_data = (int *) PyArray_DATA((PyArrayObject *) out_array);

  hoshen_kopelman(in_data, out_data, dims_int, n_dims);
  Py_INCREF(out_array);
  return out_array;
}


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


