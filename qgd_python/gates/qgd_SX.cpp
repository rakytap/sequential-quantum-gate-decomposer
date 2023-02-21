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
\file qgd_SX.cpp
\brief Python interface for the SX gate class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "SX.h"
#include "numpy_interface.h"




/**
@brief Type definition of the qgd_SX Python class of the qgd_SX module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the SX gate
    SX* gate;
} qgd_SX;


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param qbit_num The number of qubits spanning the operation.
@param target_qbit The 0<=ID<qbit_num of the target qubit.
*/
SX* 
create_SX( int qbit_num, int target_qbit ) {

    return new SX( qbit_num, target_qbit );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_SX( SX*  instance ) {
    delete instance;
    return;
}





extern "C"
{


/**
@brief Method called when a python instance of the class qgd_SX is destroyed
@param self A pointer pointing to an instance of class qgd_SX.
*/
static void
qgd_SX_dealloc(qgd_SX *self)
{

    // release the SX gate
    release_SX( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class qgd_SX is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_SX.
*/
static PyObject *
qgd_SX_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_SX *self;
    self = (qgd_SX *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_SX is initialized
@param self A pointer pointing to an instance of the class qgd_SX.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), Theta (bool) , Phi (bool), Lambda (bool)
@param kwds A tuple of keywords
*/
static int
qgd_SX_init(qgd_SX *self, PyObject *args, PyObject *kwds)
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
        self->gate = create_SX( qbit_num, target_qbit );
    }
    return 0;
}

/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/

static PyObject *
qgd_SX_get_Matrix( qgd_SX *self ) {

    Matrix SX_mtx = self->gate->get_matrix(  );
    
    // convert to numpy array
    SX_mtx.set_owner(false);
    PyObject *SX_py = matrix_to_numpy( SX_mtx );

    return SX_py;
}


/**
@brief Structure containing metadata about the members of class qgd_SX.
*/
static PyMemberDef qgd_SX_members[] = {
   {NULL}  /* Sentinel */
};


/**
@brief Structure containing metadata about the methods of class qgd_SX.
*/
static PyMethodDef qgd_SX_methods[] = {
    {"get_Matrix", (PyCFunction) qgd_SX_get_Matrix, METH_NOARGS,
     "Method to get the matrix of the operation."
    },   
  {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class qgd_SX.
*/
static PyTypeObject  qgd_SX_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_SX.qgd_SX", /*tp_name*/
  sizeof(qgd_SX), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_SX_dealloc, /*tp_dealloc*/
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
  "Object to represent a SX gate of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  qgd_SX_methods, /*tp_methods*/
  qgd_SX_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_SX_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_SX_new, /*tp_new*/
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
static PyModuleDef  qgd_SX_Module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "qgd_SX",
    .m_doc = "Python binding for QGD SX gate",
    .m_size = -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_SX(void)
{
    PyObject *m;
    if (PyType_Ready(& qgd_SX_Type) < 0)
        return NULL;

    m = PyModule_Create(& qgd_SX_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(& qgd_SX_Type);
    if (PyModule_AddObject(m, "qgd_SX", (PyObject *) & qgd_SX_Type) < 0) {
        Py_DECREF(& qgd_SX_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}




}
