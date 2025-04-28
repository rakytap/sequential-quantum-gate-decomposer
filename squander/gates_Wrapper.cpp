/*
Created on Fri Jun 26 14:42:56 2020
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

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*
\file gates_Wrapper.cpp
\brief Python interface for the CH (i.e. controlled Hadamard) gate class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "CH.h"
#include "numpy_interface.h"


//////////////////////////////////////

/**
@brief Type definition of the  qgd_CH_Wrapper Python class of the  qgd_CH_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the CH gate
    //CH* gate;
} qgd_Gate_Wrapper;


extern "C"
{

/**
@brief Method called when a python instance of the class  qgd_CH_Wrapper is destroyed
@param self A pointer pointing to an instance of class  qgd_CH_Wrapper.
*/
static void
 qgd_Gate_Wrapper_dealloc(qgd_Gate_Wrapper *self)
{
    //release_Gate( self->gate );

    //Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class  qgd_CH_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class  qgd_CH_Wrapper.
*/
static PyObject *
 qgd_Gate_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_Gate_Wrapper *self;
    self = (qgd_Gate_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class  qgd_CH_Wrapper is initialized
@param self A pointer pointing to an instance of the class  qgd_CH_Wrapper.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), control_qbit (int)
@param kwds A tuple of keywords
*/
static int
 qgd_Gate_Wrapper_init(qgd_Gate_Wrapper *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"qbit_num", (char*)"target_qbit", (char*)"control_qbit", NULL};
    int  qbit_num = -1; 
    int target_qbit = -1;
    int control_qbit = -1;


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist,
                                     &qbit_num, &target_qbit, &control_qbit))
        return -1;
/*
    if (qbit_num != -1 && target_qbit != -1) {
        self->gate = create_Gate( qbit_num, target_qbit, control_qbit );
    }
*/
    return 0;
}


/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_Gate_Wrapper_get_Name( qgd_Gate_Wrapper *self ) {

    

    return Py_BuildValue("i", -1);
}

/**
@brief Structure containing metadata about the methods of class qgd_U3.
*/
static PyMethodDef qgd_Gate_Wrapper_methods[] = {
    {"get_Name", (PyCFunction) qgd_Gate_Wrapper_get_Name, METH_NOARGS,
     "Method to get the name label of the gate"
    },
    {NULL}  /* Sentinel */
};


/**
@brief Structure containing metadata about the members of class  qgd_CH_Wrapper.
*/
static PyMemberDef  qgd_Gate_Wrapper_members[] = {
    {NULL}  /* Sentinel */
};


static PyTypeObject  Gate_Wrapper_Type = {

PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_CH_Wrapper.qgd_Gate_Wrapper", /*tp_name*/
  sizeof(qgd_Gate_Wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor)  qgd_Gate_Wrapper_dealloc, /*tp_dealloc*/
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
  "Object to represent a Gate gate of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
   qgd_Gate_Wrapper_methods, /*tp_methods*/
   qgd_Gate_Wrapper_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc)  qgd_Gate_Wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
   qgd_Gate_Wrapper_new, /*tp_new*/
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








////////////////////////////////////////




/**
@brief Structure containing metadata about the module.
*/
static PyModuleDef  gates_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    "gates_Wrapper",
    "Python binding for gates implemented in Squander C++",
    -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_CH_Wrapper(void)
{

    // initialize Numpy API
    import_array();


    PyObject * m= PyModule_Create(& gates_Wrapper_Module);
    if (m == NULL)
        return NULL;


    if (PyType_Ready(& Gate_Wrapper_Type) < 0)
        return NULL;


    Py_INCREF(&Gate_Wrapper_Type);
    if (PyModule_AddObject(m, "qgd_Gate_Wrapper", (PyObject *) & Gate_Wrapper_Type) < 0) {
        Py_DECREF(& Gate_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
