#ifndef NUMPY_INTERFACE_H
#define NUMPY_INTERFACE_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include "matrix.h"
#include "matrix_real.h"


/**
@brief Method to cleanup the memory when the python object becomes released
@param capsule Pointer to the memory capsule
*/
void capsule_cleanup(PyObject* capsule);



/**
@brief Call to make a numpy array from data stored via void pointer.
@param ptr pointer pointing to the data
@param dim The number of dimensions
@param shape array containing the dimensions.
@param np_type The data type stored in the numpy array (see possible values at https://numpy.org/doc/1.17/reference/c-api.dtype.html)
*/
PyObject* array_from_ptr(void * ptr, int dim, npy_intp* shape, int np_type);



/**
@brief Call to make a numpy array from an instance of matrix class.
@param mtx a matrix instance
*/
PyObject* matrix_to_numpy( Matrix &mtx );


/**
@brief Call to make a numpy array from an instance of matrix class.
@param mtx a matrix instance
*/
PyObject* matrix_real_to_numpy( Matrix_real &mtx );


/**
@brief Call to make a numpy array from an instance of matrix_base<int8_t> class.
@param mtx a matrix instance
*/
PyObject* matrix_int8_to_numpy( matrix_base<int8_t> &mtx );

/**
@brief Call to create a PIC matrix representation of a numpy array
*/
Matrix numpy2matrix(PyObject *arr);


/**
@brief Call to create a PIC matrix_real representation of a numpy array
*/
Matrix_real numpy2matrix_real(PyObject *arr);





#endif
