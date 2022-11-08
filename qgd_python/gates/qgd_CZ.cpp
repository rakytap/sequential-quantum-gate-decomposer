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
\file qgd_CZ.cpp
\brief Python interface for the CZ gate class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "CZ.h"
#include "numpy_interface.h"



/**
@brief Type definition of the  qgd_CZ Python class of the  qgd_CZ module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the CZ gate
    CZ* gate;
} qgd_CZ;


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param qbit_num Number of qubits spanning the unitary
@param target_qbit The Id (0<= target_qbit < qbit_num ) of the target qubit.
@param control_qbit The Id (0<= control_qbit < qbit_num ) of the control qubit.
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
CZ* 
create_CZ( int qbit_num, int target_qbit, int control_qbit ) {

    return new CZ( qbit_num, target_qbit, control_qbit );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_CZ( CZ*  instance ) {
    delete instance;
    return;
}




extern "C"
{


/**
@brief Method called when a python instance of the class  qgd_CZ is destroyed
@param self A pointer pointing to an instance of class  qgd_CZ.
*/
static void
 qgd_CZ_dealloc(qgd_CZ *self)
{
    release_CZ( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class  qgd_CZ is allocated
@param type A pointer pointing to a structure describing the type of the class  qgd_CZ.
*/
static PyObject *
 qgd_CZ_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_CZ *self;
    self = (qgd_CZ *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class  qgd_CZ is initialized
@param self A pointer pointing to an instance of the class  qgd_CZ.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), control_qbit (int)
@param kwds A tuple of keywords
*/
static int
 qgd_CZ_init(qgd_CZ *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"qbit_num", (char*)"target_qbit", (char*)"control_qbit", NULL};
    int  qbit_num = -1; 
    int target_qbit = -1;
    int control_qbit = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist,
                                     &qbit_num, &target_qbit, &control_qbit))
        return -1;

    if (qbit_num != -1 && target_qbit != -1) {
        self->gate = create_CZ( qbit_num, target_qbit, control_qbit );
    }
    return 0;
}

/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_CZ_get_Matrix( qgd_CZ *self, PyObject *args ) {

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


    Matrix CZ_mtx = self->gate->get_matrix(  );
    
    // convert to numpy array
    CZ_mtx.set_owner(false);
    PyObject *CZ_py = matrix_to_numpy( CZ_mtx );


    Py_DECREF(parameters_arr);

    return CZ_py;
}

/**
@brief Structure containing metadata about the members of class  qgd_CZ.
*/
static PyMemberDef  qgd_CZ_members[] = {
    {NULL}  /* Sentinel */
};



/**
@brief Structure containing metadata about the methods of class  qgd_CZ.
*/
static PyMethodDef  qgd_CZ_methods[] = {
    {"get_Matrix", (PyCFunction) qgd_CZ_get_Matrix, METH_VARARGS,
     "Method to get the matrix of the operation."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class  qgd_CZ.
*/
static PyTypeObject  qgd_CZ_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_CZ.qgd_CZ", /*tp_name*/
  sizeof(qgd_CZ), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor)  qgd_CZ_dealloc, /*tp_dealloc*/
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
  "Object to represent a CZ gate of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
   qgd_CZ_methods, /*tp_methods*/
   qgd_CZ_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc)  qgd_CZ_init, /*tp_init*/
  0, /*tp_alloc*/
   qgd_CZ_new, /*tp_new*/
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
static PyModuleDef  qgd_CZ_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "qgd_CZ",
    .m_doc = "Python binding for QGD CZ gate",
    .m_size = -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_CZ(void)
{
    PyObject *m;
    if (PyType_Ready(& qgd_CZ_Type) < 0)
        return NULL;

    m = PyModule_Create(& qgd_CZ_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(& qgd_CZ_Type);
    if (PyModule_AddObject(m, "qgd_CZ", (PyObject *) & qgd_CZ_Type) < 0) {
        Py_DECREF(& qgd_CZ_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
