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
\file qgd_CNOT_Wrapper.cpp
\brief Python interface for the CNOT gate class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "CNOT.h"
#include "numpy_interface.h"



/**
@brief Type definition of the  qgd_CNOT_Wrapper Python class of the  qgd_CNOT_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the CNOT gate
    CNOT* gate;
} qgd_CNOT_Wrapper;


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param qbit_num Number of qubits spanning the unitary
@param target_qbit The Id (0<= target_qbit < qbit_num ) of the target qubit.
@param control_qbit The Id (0<= control_qbit < qbit_num ) of the control qubit.
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
CNOT* 
create_CNOT( int qbit_num, int target_qbit, int control_qbit ) {

    return new CNOT( qbit_num, target_qbit, control_qbit );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_CNOT( CNOT*  instance ) {
    delete instance;
    return;
}




extern "C"
{


/**
@brief Method called when a python instance of the class  qgd_CNOT_Wrapper is destroyed
@param self A pointer pointing to an instance of class  qgd_CNOT_Wrapper.
*/
static void
 qgd_CNOT_Wrapper_dealloc(qgd_CNOT_Wrapper *self)
{
    release_CNOT( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class  qgd_CNOT_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class  qgd_CNOT_Wrapper.
*/
static PyObject *
 qgd_CNOT_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_CNOT_Wrapper *self;
    self = (qgd_CNOT_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class  qgd_CNOT_Wrapper is initialized
@param self A pointer pointing to an instance of the class  qgd_CNOT_Wrapper.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), control_qbit (int)
@param kwds A tuple of keywords
*/
static int
 qgd_CNOT_Wrapper_init(qgd_CNOT_Wrapper *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"qbit_num", (char*)"target_qbit", (char*)"control_qbit", NULL};
    int  qbit_num = -1; 
    int target_qbit = -1;
    int control_qbit = -1;

    if (PyArray_API == NULL) {
        import_array();
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist,
                                     &qbit_num, &target_qbit, &control_qbit))
        return -1;

    if (qbit_num != -1 && target_qbit != -1) {
        self->gate = create_CNOT( qbit_num, target_qbit, control_qbit );
    }
    return 0;
}

/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_CNOT_Wrapper_get_Matrix( qgd_CNOT_Wrapper *self ) {

    int parallel = 1;
    Matrix CNOT_mtx = self->gate->get_matrix( parallel );
    
    // convert to numpy array
    CNOT_mtx.set_owner(false);
    PyObject *CNOT_py = matrix_to_numpy( CNOT_mtx );

    return CNOT_py;
}



/**
@brief Call to apply the gate operation on the inut matrix
*/
static PyObject *
qgd_CNOT_Wrapper_apply_to( qgd_CNOT_Wrapper *self, PyObject *args ) {

    PyArrayObject * unitary_arg = NULL;



    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &unitary_arg )) 
        return Py_BuildValue("i", -1);

    // convert python object array to numpy C API array
    if ( unitary_arg == NULL ) {
        PyErr_SetString(PyExc_Exception, "Input matrix was not given");
        return NULL;
    }


    if ( PyArray_Check(unitary_arg) && PyArray_IS_C_CONTIGUOUS(unitary_arg) ) {
        Py_INCREF(unitary_arg);
    }
    else {
        unitary_arg = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)unitary_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

   PyArrayObject* unitary = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)unitary_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

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
qgd_CNOT_Wrapper_get_Gate_Kernel( qgd_CNOT_Wrapper *self ) {



    // create QGD version of the input matrix

    Matrix CNOT_1qbit_ = self->gate->calc_one_qubit_u3( );
    PyObject *CNOT_1qbit = matrix_to_numpy( CNOT_1qbit_ );

    return CNOT_1qbit;
}



/**
@brief Call to get the number of free parameters in the gate
@return Returns with the starting index
*/
static PyObject *
qgd_CNOT_Wrapper_get_Parameter_Num( qgd_CNOT_Wrapper *self ) {

    int parameter_num = self->gate->get_parameter_num();

    return Py_BuildValue("i", parameter_num);

}

/**
@brief Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated
@return Returns with the starting index
*/
static PyObject *
qgd_CNOT_Wrapper_get_Parameter_Start_Index( qgd_CNOT_Wrapper *self ) {

    int start_index = self->gate->get_parameter_start_idx();

    return Py_BuildValue("i", start_index);

}


/**
@brief Call to get the target qbit
@return Returns with the target qbit
*/
static PyObject *
qgd_CNOT_Wrapper_get_Target_Qbit( qgd_CNOT_Wrapper *self ) {

    int target_qbit = self->gate->get_target_qbit();

    return Py_BuildValue("i", target_qbit);

}

/**
@brief Call to get the control qbit (returns with -1 if no control qbit is used in the gate)
@return Returns with the control qbit
*/
static PyObject *
qgd_CNOT_Wrapper_get_Control_Qbit( qgd_CNOT_Wrapper *self ) {

    int control_qbit = self->gate->get_control_qbit();

    return Py_BuildValue("i", control_qbit);

}

/**
@brief Call to set the target qbit
*/
static PyObject *
qgd_CNOT_Wrapper_set_Target_Qbit( qgd_CNOT_Wrapper *self, PyObject *args ) {
    int target_qbit_in = -1;
    if (!PyArg_ParseTuple(args, "|i", &target_qbit_in )) 
        return Py_BuildValue("i", -1);
    self->gate->set_target_qbit(target_qbit_in);

    return Py_BuildValue("i", 0);

}

/**
@brief Call to set the target qbit
*/
static PyObject *
qgd_CNOT_Wrapper_set_Control_Qbit( qgd_CNOT_Wrapper *self, PyObject *args ) {
    int control_qbit_in = -1;
    if (!PyArg_ParseTuple(args, "|i", &control_qbit_in )) 
        return Py_BuildValue("i", -1);
    self->gate->set_control_qbit(control_qbit_in);

    return Py_BuildValue("i", 0);

}

/**
@brief Call to extract the paramaters corresponding to the gate, from a parameter array associated to the circuit in which the gate is embedded.
*/
static PyObject *
qgd_CNOT_Wrapper_Extract_Parameters( qgd_CNOT_Wrapper *self, PyObject *args ) {

    PyArrayObject * parameters_arr = NULL;


    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arr )) 
        return Py_BuildValue("i", -1);

    
    if ( PyArray_IS_C_CONTIGUOUS(parameters_arr) ) {
        Py_INCREF(parameters_arr);
    }
    else {
        parameters_arr = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }

    // get the C++ wrapper around the data
    Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );


    
    Matrix_real extracted_parameters;

    try {
        extracted_parameters = self->gate->extract_parameters( parameters_mtx );
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to circuit class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    // convert to numpy array
    extracted_parameters.set_owner(false);
    PyObject *extracted_parameters_py = matrix_real_to_numpy( extracted_parameters );
   

    return extracted_parameters_py;
}




/**
@brief Structure containing metadata about the members of class  qgd_CNOT_Wrapper.
*/
static PyMemberDef  qgd_CNOT_Wrapper_members[] = {
    {NULL}  /* Sentinel */
};


/**
@brief Structure containing metadata about the methods of class  qgd_CNOT_Wrapper.
*/
static PyMethodDef  qgd_CNOT_Wrapper_methods[] = {
    {"get_Matrix", (PyCFunction) qgd_CNOT_Wrapper_get_Matrix, METH_NOARGS,
     "Method to get the matrix of the operation."
    },
    {"apply_to", (PyCFunction) qgd_CNOT_Wrapper_apply_to, METH_VARARGS,
     "Call to apply the gate on the input matrix."
    },
    {"get_Gate_Kernel", (PyCFunction) qgd_CNOT_Wrapper_get_Gate_Kernel, METH_NOARGS,
     "Call to calculate the gate matrix acting on a single qbit space."
    },
    {"get_Parameter_Num", (PyCFunction) qgd_CNOT_Wrapper_get_Parameter_Num, METH_NOARGS,
     "Call to get the number of free parameters in the gate."
    },
    {"get_Parameter_Start_Index", (PyCFunction) qgd_CNOT_Wrapper_get_Parameter_Start_Index, METH_NOARGS,
     "Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated."
    },
    {"get_Target_Qbit", (PyCFunction) qgd_CNOT_Wrapper_get_Target_Qbit, METH_NOARGS,
     "Call to get the target qbit."
    },
    {"get_Control_Qbit", (PyCFunction) qgd_CNOT_Wrapper_get_Control_Qbit, METH_NOARGS,
     "Call to get the control qbit (returns with -1 if no control qbit is used in the gate)."
    },
    {"set_Target_Qbit", (PyCFunction) qgd_CNOT_Wrapper_set_Target_Qbit, METH_VARARGS,
     "Call to get the target qbit."
    },
    {"set_Control_Qbit", (PyCFunction) qgd_CNOT_Wrapper_set_Control_Qbit, METH_VARARGS,
     "Call to get the control qbit."
    },
    {"Extract_Parameters", (PyCFunction) qgd_CNOT_Wrapper_Extract_Parameters, METH_VARARGS,
     "Call to extract the paramaters corresponding to the gate, from a parameter array associated to the circuit in which the gate is embedded."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class  qgd_CNOT_Wrapper.
*/
static PyTypeObject  qgd_CNOT_Wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_CNOT_Wrapper.qgd_CNOT_Wrapper", /*tp_name*/
  sizeof(qgd_CNOT_Wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor)  qgd_CNOT_Wrapper_dealloc, /*tp_dealloc*/
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
  "Object to represent a CNOT gate of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
   qgd_CNOT_Wrapper_methods, /*tp_methods*/
   qgd_CNOT_Wrapper_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc)  qgd_CNOT_Wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
   qgd_CNOT_Wrapper_new, /*tp_new*/
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
static PyModuleDef  qgd_CNOT_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    "qgd_CNOT_Wrapper",
    "Python binding for QGD CNOT gate",
    -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_CNOT_Wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(& qgd_CNOT_Wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(& qgd_CNOT_Wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(& qgd_CNOT_Wrapper_Type);
    if (PyModule_AddObject(m, "qgd_CNOT_Wrapper", (PyObject *) & qgd_CNOT_Wrapper_Type) < 0) {
        Py_DECREF(& qgd_CNOT_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
