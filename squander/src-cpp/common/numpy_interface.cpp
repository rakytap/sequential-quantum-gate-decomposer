/*
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include "matrix.h"
#include "matrix_real.h"


/**
@brief Method to cleanup the memory when the python object becomes released
@param capsule Pointer to the memory capsule
*/
void capsule_cleanup(PyObject* capsule) {

    void *memory = PyCapsule_GetPointer(capsule, NULL);
    // I'm going to assume your memory needs to be freed with free().
    // If it needs different cleanup, perform whatever that cleanup is
    // instead of calling free().
    scalable_aligned_free(memory);


}




/**
@brief Call to make a numpy array from data stored via void pointer.
@param ptr pointer pointing to the data
@param dim The number of dimensions
@param shape array containing the dimensions.
@param np_type The data type stored in the numpy array (see possible values at https://numpy.org/doc/1.17/reference/c-api.dtype.html)
*/
PyObject* array_from_ptr(void * ptr, int dim, npy_intp* shape, int np_type) {

        if (PyArray_API == NULL) {
            import_array();
        }


        // create numpy array
        PyObject* arr = PyArray_SimpleNewFromData(dim, shape, np_type, ptr);

        // set memory keeper for the numpy array
        PyObject *capsule = PyCapsule_New(ptr, NULL, capsule_cleanup);
        PyArray_SetBaseObject((PyArrayObject *) arr, capsule);

        return arr;

}



/**
@brief Call to make a numpy array from an instance of matrix class.
@param mtx a matrix instance
*/
PyObject* matrix_to_numpy( Matrix &mtx ) {
        // initialize Numpy API
        import_array();


        npy_intp shape[2];
        shape[0] = (npy_intp) mtx.rows;
        shape[1] = (npy_intp) mtx.cols;

        QGD_Complex16* data = mtx.get_data();
        return array_from_ptr( (void*) data, 2, shape, NPY_COMPLEX128);


}



/**
@brief Call to make a numpy array from an instance of matrix class.
@param mtx a matrix instance
*/
PyObject* matrix_real_to_numpy( Matrix_real &mtx ) {
        // initialize Numpy API
        import_array();

        npy_intp shape[2];

        // TODO: this logic should be implemented on higher level of API logic in the interfaces.
        // fix this issue after redesigning the decomposition interfaces
        int dims;
        if ( mtx.rows == 1 ) {
            dims = 1;
            shape[0] = mtx.cols;
        }
        else if ( mtx.cols == 1 ) {
            dims = 1;
            shape[0] = mtx.rows;
        }
        else {
            dims = 2;
            shape[0] = (npy_intp) mtx.rows;
            shape[1] = (npy_intp) mtx.cols;
        }

        double* data = mtx.get_data();
        return array_from_ptr( (void*) data, dims, shape, NPY_DOUBLE);


}



/**
@brief Call to make a numpy array from an instance of matrix_base<int8_t> class.
@param mtx a matrix instance
*/
PyObject* matrix_int8_to_numpy( matrix_base<int8_t> &mtx ) {
        // initialize Numpy API
        import_array();


        npy_intp shape[2];
        shape[0] = (npy_intp) mtx.rows;
        shape[1] = (npy_intp) mtx.cols;

        int8_t* data = mtx.get_data();
        return array_from_ptr( (void*) data, 2, shape, NPY_INT8);


}




/**
@brief Call to create a matrix representation of a numpy array
*/
Matrix
numpy2matrix(PyArrayObject* arr) {

    if ( (PyObject*)arr == Py_None ) {
        return Matrix(0,0);
    }

#ifdef DEBUG
    // test C-style contiguous memory allocation of the arrays
    // in production this case has to be handled outside
    assert( PyArray_IS_C_CONTIGUOUS(arr) && "array is not memory contiguous" );
#endif

    // get the pointer to the data stored in the input matrices
    QGD_Complex16* data = (QGD_Complex16*)PyArray_DATA(arr);

    // get the dimensions of the array self->C
    int dim_num = PyArray_NDIM( arr );
    npy_intp* dims = PyArray_DIMS(arr);

    // create PIC version of the input matrices
    if (dim_num == 2) {
        Matrix mtx = Matrix(data, dims[0], dims[1]);
        return mtx;
    }
    else if (dim_num == 1) {
        Matrix mtx = Matrix(data, dims[0], 1);
        return mtx;
    }
    else {
        std::string err( "numpy2matrix: Wrong matrix dimension was given");
        throw err;
    }



}


/**
@brief Call to create a PIC matrix_real representation of a numpy array
*/
Matrix_real
numpy2matrix_real(PyArrayObject* arr) {


    if ( (PyObject*)arr == Py_None ) {
        return Matrix_real(0,0);
    }

#ifdef DEBUG
    // test C-style contiguous memory allocation of the arrays
    // in production this case has to be handled outside
    assert( PyArray_IS_C_CONTIGUOUS(arr) && "array is not memory contiguous" );
#endif

    // get the pointer to the data stored in the input matrices
    double *data = (double *)PyArray_DATA(arr);

    // get the dimensions of the array self->C
    int dim_num = PyArray_NDIM( arr );
    npy_intp* dims = PyArray_DIMS(arr);

    // create PIC version of the input matrices
    if (dim_num == 2) {
        Matrix_real mtx = Matrix_real(data, dims[0], dims[1]);
        return mtx;
    }
    else if (dim_num == 1) {
        Matrix_real mtx = Matrix_real(data, dims[0], 1);
        return mtx;
    }
    else {
        std::string err( "numpy2matrix: Wrong matrix dimension was given");
        throw err;
    }



}





