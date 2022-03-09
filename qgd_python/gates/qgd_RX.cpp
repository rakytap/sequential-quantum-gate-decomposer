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
\file qgd_RX.cpp
\brief Python interface for the RX gate class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "RX.h"





/**
@brief Type definition of the qgd_RX Python class of the qgd_RX module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the RX gate
    RX* gate;
} qgd_RX;


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param qbit_num The number of qubits spanning the operation.
@param target_qbit The 0<=ID<qbit_num of the target qubit.
*/
RX* 
create_RX( int qbit_num, int target_qbit ) {

    return new RX( qbit_num, target_qbit );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_RX( RX*  instance ) {
    delete instance;
    return;
}





extern "C"
{


/**
@brief Method called when a python instance of the class qgd_RX is destroyed
@param self A pointer pointing to an instance of class qgd_RX.
*/
static void
qgd_RX_dealloc(qgd_RX *self)
{

    // release the RX gate
    release_RX( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class qgd_RX is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_RX.
*/
static PyObject *
qgd_RX_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_RX *self;
    self = (qgd_RX *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_RX is initialized
@param self A pointer pointing to an instance of the class qgd_RX.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), Theta (bool) , Phi (bool), Lambda (bool)
@param kwds A tuple of keywords
*/
static int
qgd_RX_init(qgd_RX *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"qbit_num", (char*)"target_qbit", NULL};
    int  qbit_num = -1; 
    int target_qbit = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,
                                     &qbit_num, &target_qbit))
        return -1;

    if (qbit_num != -1 && target_qbit != -1) {
        self->gate = create_RX( qbit_num, target_qbit );
    }
    return 0;
}


/**
@brief Structure containing metadata about the members of class qgd_RX.
*/
static PyMemberDef qgd_RX_members[] = {
    {NULL}  /* Sentinel */
};


/**
@brief Structure containing metadata about the methods of class qgd_RX.
*/
static PyMethodDef qgd_RX_methods[] = {
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class qgd_RX.
*/
static PyTypeObject  qgd_RX_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_RX.qgd_RX", /*tp_name*/
  sizeof(qgd_RX), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_RX_dealloc, /*tp_dealloc*/
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
  "Object to represent a RX gate of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  qgd_RX_methods, /*tp_methods*/
  qgd_RX_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_RX_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_RX_new, /*tp_new*/
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
static PyModuleDef  qgd_RX_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "qgd_RX",
    .m_doc = "Python binding for QGD RX gate",
    .m_size = -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_RX(void)
{
    PyObject *m;
    if (PyType_Ready(& qgd_RX_Type) < 0)
        return NULL;

    m = PyModule_Create(& qgd_RX_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(& qgd_RX_Type);
    if (PyModule_AddObject(m, "qgd_RX", (PyObject *) & qgd_RX_Type) < 0) {
        Py_DECREF(& qgd_RX_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}




}
