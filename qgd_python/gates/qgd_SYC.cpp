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
\file qgd_SYC.cpp
\brief Python interface for the Sycamore gate class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "SYC.h"
#include "numpy_interface.h"



/**
@brief Type definition of the  qgd_SYC Python class of the  qgd_SYC module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the SYC gate
    SYC* gate;
} qgd_SYC;


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param qbit_num Number of qubits spanning the unitary
@param target_qbit The Id (0<= target_qbit < qbit_num ) of the target qubit.
@param control_qbit The Id (0<= control_qbit < qbit_num ) of the control qubit.
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
SYC* 
create_SYC( int qbit_num, int target_qbit, int control_qbit ) {

    return new SYC( qbit_num, target_qbit, control_qbit );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_SYC( SYC*  instance ) {
    delete instance;
    return;
}




extern "C"
{


/**
@brief Method called when a python instance of the class  qgd_SYC is destroyed
@param self A pointer pointing to an instance of class  qgd_SYC.
*/
static void
 qgd_SYC_dealloc(qgd_SYC *self)
{
    release_SYC( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class  qgd_SYC is allocated
@param type A pointer pointing to a structure describing the type of the class  qgd_SYC.
*/
static PyObject *
 qgd_SYC_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_SYC *self;
    self = (qgd_SYC *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class  qgd_SYC is initialized
@param self A pointer pointing to an instance of the class  qgd_SYC.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), control_qbit (int)
@param kwds A tuple of keywords
*/
static int
 qgd_SYC_init(qgd_SYC *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"qbit_num", (char*)"target_qbit", (char*)"control_qbit", NULL};
    int  qbit_num = -1; 
    int target_qbit = -1;
    int control_qbit = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist,
                                     &qbit_num, &target_qbit, &control_qbit))
        return -1;

    if (qbit_num != -1 && target_qbit != -1) {
        self->gate = create_SYC( qbit_num, target_qbit, control_qbit );
    }
    return 0;
}

/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/

static PyObject *
qgd_SYC_get_Matrix( qgd_SYC *self ) {

    Matrix SYC_mtx = self->gate->get_matrix(  );
    
    // convert to numpy array
    SYC_mtx.set_owner(false);
    PyObject *SYC_py = matrix_to_numpy( SYC_mtx );

    return SYC_py;
}



/**
@brief Structure containing metadata about the members of class  qgd_SYC.
*/
static PyMemberDef  qgd_SYC_members[] = {
    {NULL}  /* Sentinel */
};



/**
@brief Structure containing metadata about the methods of class  qgd_SYC.
*/
static PyMethodDef  qgd_SYC_methods[] = {
    {"get_Matrix", (PyCFunction) qgd_SYC_get_Matrix, METH_NOARGS,
     "Method to get the matrix of the operation."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class  qgd_SYC.
*/
static PyTypeObject  qgd_SYC_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_SYC.qgd_SYC", /*tp_name*/
  sizeof(qgd_SYC), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor)  qgd_SYC_dealloc, /*tp_dealloc*/
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
  "Object to represent a SYC gate of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
   qgd_SYC_methods, /*tp_methods*/
   qgd_SYC_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc)  qgd_SYC_init, /*tp_init*/
  0, /*tp_alloc*/
   qgd_SYC_new, /*tp_new*/
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
static PyModuleDef  qgd_SYC_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "qgd_SYC",
    .m_doc = "Python binding for QGD SYC gate",
    .m_size = -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_SYC(void)
{
    PyObject *m;
    if (PyType_Ready(& qgd_SYC_Type) < 0)
        return NULL;

    m = PyModule_Create(& qgd_SYC_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(& qgd_SYC_Type);
    if (PyModule_AddObject(m, "qgd_SYC", (PyObject *) & qgd_SYC_Type) < 0) {
        Py_DECREF(& qgd_SYC_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
