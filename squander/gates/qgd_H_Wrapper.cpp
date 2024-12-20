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
\file qgd_H_Wrapper.cpp
\brief Python interface for the Hadamard gate class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "X.h"
#include "numpy_interface.h"




/**
@brief Type definition of the qgd_H_Wrapper Python class of the qgd_H_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the X gate
    X* gate;
} qgd_H_Wrapper;


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
@brief Method called when a python instance of the class qgd_H_Wrapper is destroyed
@param self A pointer pointing to an instance of class qgd_H_Wrapper.
*/
static void
qgd_H_Wrapper_dealloc(qgd_H_Wrapper *self)
{

    // release the X gate
    release_X( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class qgd_H_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_H_Wrapper.
*/
static PyObject *
qgd_H_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_H_Wrapper *self;
    self = (qgd_H_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_H_Wrapper is initialized
@param self A pointer pointing to an instance of the class qgd_H_Wrapper.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), Theta (bool) , Phi (bool), Lambda (bool)
@param kwds A tuple of keywords
*/
static int
qgd_H_Wrapper_init(qgd_H_Wrapper *self, PyObject *args, PyObject *kwds)
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
        self->gate = create_X( qbit_num, target_qbit );
    }




    return 0;
}

/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_H_Wrapper_get_Matrix( qgd_H_Wrapper *self ) {

    int parallel = 1;   
    Matrix X_mtx = self->gate->get_matrix( parallel  );
    
    // convert to numpy array
    X_mtx.set_owner(false);
    PyObject *X_py = matrix_to_numpy( X_mtx );


    return X_py;
}



/**
@brief Call to apply the gate operation on the inut matrix
*/
static PyObject *
qgd_H_Wrapper_apply_to( qgd_H_Wrapper *self, PyObject *args ) {

    PyObject * unitary_arg = NULL;



    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &unitary_arg )) 
        return Py_BuildValue("i", -1);


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

    int parallel = 1;
    self->gate->apply_to( unitary_mtx, parallel );
    
    if (unitary_mtx.data != PyArray_DATA(unitary)) {
        memcpy(PyArray_DATA(unitary), unitary_mtx.data, unitary_mtx.size() * sizeof(QGD_Complex16));
    }

    Py_DECREF(unitary);

    return Py_BuildValue("i", 0);
}

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on a single qbit space.
@param ThetaOver2 Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/

static PyObject *
qgd_H_Wrapper_get_Gate_Kernel( qgd_H_Wrapper *self ) {



    // create QGD version of the input matrix

    Matrix X_1qbit_ = self->gate->calc_one_qubit_u3(  );
    
    PyObject *X_1qbit = matrix_to_numpy( X_1qbit_ );

    return X_1qbit;

}

/**
@brief Structure containing metadata about the members of class qgd_H_Wrapper.
*/
static PyMemberDef qgd_H_Wrapper_members[] = {
    {NULL}  /* Sentinel */
};


/**
@brief Structure containing metadata about the methods of class qgd_H_Wrapper.
*/
static PyMethodDef qgd_H_Wrapper_methods[] = {
    {"get_Matrix", (PyCFunction) qgd_H_Wrapper_get_Matrix, METH_NOARGS,
     "Method to get the matrix of the operation."
    },
    {"apply_to", (PyCFunction) qgd_H_Wrapper_apply_to, METH_VARARGS,
     "Call to apply the gate on the input matrix."
    },
    {"get_Gate_Kernel", (PyCFunction) qgd_H_Wrapper_get_Gate_Kernel, METH_NOARGS,
     "Call to calculate the gate matrix acting on a single qbit space."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class qgd_H_Wrapper.
*/
static PyTypeObject  qgd_H_Wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_H_Wrapper.qgd_H_Wrapper", /*tp_name*/
  sizeof(qgd_H_Wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_H_Wrapper_dealloc, /*tp_dealloc*/
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
  qgd_H_Wrapper_methods, /*tp_methods*/
  qgd_H_Wrapper_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_H_Wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_H_Wrapper_new, /*tp_new*/
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
static PyModuleDef  qgd_H_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    "qgd_H_Wrapper",
    "Python binding for QGD X gate",
    -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_H_Wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(& qgd_H_Wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(& qgd_H_Wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(& qgd_H_Wrapper_Type);
    if (PyModule_AddObject(m, "qgd_H_Wrapper", (PyObject *) & qgd_H_Wrapper_Type) < 0) {
        Py_DECREF(& qgd_H_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}




}
