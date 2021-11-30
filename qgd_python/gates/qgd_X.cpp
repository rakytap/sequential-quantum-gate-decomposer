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
\file qgd_X.cpp
\brief Python interface for the X gate class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "X.h"





/**
@brief Type definition of the qgd_X Python class of the qgd_X module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the X gate
    X* gate;
} qgd_X;


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param qbit_num The number of qubits spanning the operation.
@param target_qbit The 0<=ID<qbit_num of the target qubit.
*/
X* 
create_X( int qbit_num, int target_qbit ) {

    return new X( qbit_num, target_qbit );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_X( X*  instance ) {
    delete instance;
    return;
}





extern "C"
{


/**
@brief Method called when a python instance of the class qgd_X is destroyed
@param self A pointer pointing to an instance of class qgd_X.
*/
static void
qgd_X_dealloc(qgd_X *self)
{

    // release the X gate
    release_X( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class qgd_X is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_X.
*/
static PyObject *
qgd_X_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_X *self;
    self = (qgd_X *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_X is initialized
@param self A pointer pointing to an instance of the class qgd_X.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), Theta (bool) , Phi (bool), Lambda (bool)
@param kwds A tuple of keywords
*/
static int
qgd_X_init(qgd_X *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"qbit_num", (char*)"target_qbit", NULL};
    int  qbit_num = -1; 
    int target_qbit = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,
                                     &qbit_num, &target_qbit))
        return -1;

    if (qbit_num != -1 && target_qbit != -1) {
        self->gate = create_X( qbit_num, target_qbit );
    }
    return 0;
}


/**
@brief Structure containing metadata about the members of class qgd_X.
*/
static PyMemberDef qgd_X_members[] = {
    {NULL}  /* Sentinel */
};


/**
@brief Structure containing metadata about the methods of class qgd_X.
*/
static PyMethodDef qgd_X_methods[] = {
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class qgd_X.
*/
static PyTypeObject  qgd_X_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_X.qgd_X", /*tp_name*/
  sizeof(qgd_X), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_X_dealloc, /*tp_dealloc*/
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
  "Object to represent a X gate of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  qgd_X_methods, /*tp_methods*/
  qgd_X_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_X_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_X_new, /*tp_new*/
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
static PyModuleDef  qgd_X_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "qgd_X",
    .m_doc = "Python binding for QGD X gate",
    .m_size = -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_X(void)
{
    PyObject *m;
    if (PyType_Ready(& qgd_X_Type) < 0)
        return NULL;

    m = PyModule_Create(& qgd_X_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(& qgd_X_Type);
    if (PyModule_AddObject(m, "qgd_X", (PyObject *) & qgd_X_Type) < 0) {
        Py_DECREF(& qgd_X_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}




}