/*
Created on Fri Jun 26 14:42:56 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*
\file qgd_RY.cpp
\brief Python interface for the RY gate class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "RY.h"
#include "numpy_interface.h"




/**
@brief Type definition of the qgd_RY Python class of the qgd_RY module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the RY gate
    RY* gate;
} qgd_RY;


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param qbit_num The number of qubits spanning the operation.
@param target_qbit The 0<=ID<qbit_num of the target qubit.
*/
RY* 
create_RY( int qbit_num, int target_qbit ) {

    return new RY( qbit_num, target_qbit );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_RY( RY*  instance ) {
    delete instance;
    return;
}





extern "C"
{


/**
@brief Method called when a python instance of the class qgd_RY is destroyed
@param self A pointer pointing to an instance of class qgd_RY.
*/
static void
qgd_RY_dealloc(qgd_RY *self)
{

    // release the RY gate
    release_RY( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class qgd_RY is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_RY.
*/
static PyObject *
qgd_RY_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_RY *self;
    self = (qgd_RY *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_RY is initialized
@param self A pointer pointing to an instance of the class qgd_RY.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), Theta (bool) , Phi (bool), Lambda (bool)
@param kwds A tuple of keywords
*/
static int
qgd_RY_init(qgd_RY *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"qbit_num", (char*)"target_qbit", NULL};
    int  qbit_num = -1; 
    int target_qbit = -1;

    if (PyArray_API == NULL) {
        import_array();
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,
                                     &qbit_num, &target_qbit))
        return -1;

    if (qbit_num != -1 && target_qbit != -1) {
        self->gate = create_RY( qbit_num, target_qbit );
    }
    return 0;
}

/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_RY_get_Matrix( qgd_RY *self, PyObject *args ) {

    PyObject * parameters_arr = NULL;


    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arr )) 
        return Py_BuildValue("i", -1);

    
    if ( PyArray_IS_C_CONTIGUOUS(parameters_arr) ) {
        Py_INCREF(parameters_arr);
    }
    else {
        parameters_arr = PyArray_FROM_OTF(parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }


    // get the C++ wrapper around the data
    Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );


    Matrix RY_mtx = self->gate->get_matrix( parameters_mtx );
    
    // convert to numpy array
    RY_mtx.set_owner(false);
    PyObject *RY_py = matrix_to_numpy( RY_mtx );


    Py_DECREF(parameters_arr);

    return RY_py;
}



/**
@brief Call to apply the gate operation on the inut matrix
*/
static PyObject *
qgd_RY_apply_to( qgd_RY *self, PyObject *args ) {

    PyObject * parameters_arr = NULL;
    PyObject * unitary_arg = NULL;



    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|OO", &parameters_arr, &unitary_arg )) 
        return Py_BuildValue("i", -1);
    
    if ( PyArray_IS_C_CONTIGUOUS(parameters_arr) ) {
        Py_INCREF(parameters_arr);
    }
    else {
        parameters_arr = PyArray_FROM_OTF(parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }

    // get the C++ wrapper around the data
    Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );

    // convert python object array to numpy C API array
    if ( unitary_arg == NULL ) {
        PyErr_SetString(PyExc_Exception, "Input matrix was not given");
        return NULL;
    }

    PyObject* unitary = PyArray_FROM_OTF(unitary_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(unitary) ) {
        PyErr_SetString(PyExc_Exception, "input mtrix is not memory contiguous");
        return NULL;
    }

    // create QGD version of the input matrix
    Matrix unitary_mtx = numpy2matrix(unitary);

    self->gate->apply_to( parameters_mtx, unitary_mtx );
    
    Py_DECREF(parameters_arr);
    Py_DECREF(unitary);

    return Py_BuildValue("i", 0);
}



/**
@brief Structure containing metadata about the members of class qgd_RY.
*/
static PyMemberDef qgd_RY_members[] = {
    {NULL}  /* Sentinel */
};


/**
@brief Structure containing metadata about the methods of class qgd_RY.
*/
static PyMethodDef qgd_RY_methods[] = {
    {"get_Matrix", (PyCFunction) qgd_RY_get_Matrix, METH_VARARGS,
     "Method to get the matrix of the operation."
    },
    {"apply_to", (PyCFunction) qgd_RY_apply_to, METH_VARARGS,
     "Call to apply the gate on the input matrix."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class qgd_RY.
*/
static PyTypeObject  qgd_RY_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_RY.qgd_RY", /*tp_name*/
  sizeof(qgd_RY), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_RY_dealloc, /*tp_dealloc*/
  #if PY_VERSION_HEX < 0x030800b4
  0, /*tp_print*/
  #endif
  #if PY_VERSION_HEX >= 0x030800b4
  0, /*tp_vectorcall_offset*/
  #endif
  0, /*tp_getattr*/
  0, /*tp_setattr*/
  #if PY_MAJOR_VERSION < 3
  0, /*tp_compare*/
  #endif
  #if PY_MAJOR_VERSION >= 3
  0, /*tp_as_async*/
  #endif
  0, /*tp_repr*/
  0, /*tp_as_number*/
  0, /*tp_as_sequence*/
  0, /*tp_as_mapping*/
  0, /*tp_hash*/
  0, /*tp_call*/
  0, /*tp_str*/
  0, /*tp_getattro*/
  0, /*tp_setattro*/
  0, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "Object to represent a RY gate of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  qgd_RY_methods, /*tp_methods*/
  qgd_RY_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_RY_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_RY_new, /*tp_new*/
  0, /*tp_free*/
  0, /*tp_is_gc*/
  0, /*tp_bases*/
  0, /*tp_mro*/
  0, /*tp_cache*/
  0, /*tp_subclasses*/
  0, /*tp_weaklist*/
  0, /*tp_del*/
  0, /*tp_version_tag*/
  #if PY_VERSION_HEX >= 0x030400a1
  0, /*tp_finalize*/
  #endif
  #if PY_VERSION_HEX >= 0x030800b1
  0, /*tp_vectorcall*/
  #endif
  #if PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
  0, /*tp_print*/
  #endif
};


/**
@brief Structure containing metadata about the module.
*/
static PyModuleDef  qgd_RY_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "qgd_RY",
    .m_doc = "Python binding for QGD RY gate",
    .m_size = -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_RY(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(& qgd_RY_Type) < 0)
        return NULL;

    m = PyModule_Create(& qgd_RY_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(& qgd_RY_Type);
    if (PyModule_AddObject(m, "qgd_RY", (PyObject *) & qgd_RY_Type) < 0) {
        Py_DECREF(& qgd_RY_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}




}
