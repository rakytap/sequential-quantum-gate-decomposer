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
\file qgd_CROT_Wrapper.cpp
\brief Python interface for the CROT gate class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "CROT.h"
#include "numpy_interface.h"




/**
@brief Type definition of the qgd_CROT_Wrapper Python class of the qgd_CROT_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the RY gate
    CROT* gate;
} qgd_CROT_Wrapper;


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param qbit_num The number of qubits spanning the operation.
@param target_qbit The 0<=ID<qbit_num of the target qubit.
*/
CROT* 
create_CROT( int qbit_num, int target_qbit, int control_qbit, crot_type subtype_in  ) {

    return new CROT( qbit_num, target_qbit, control_qbit, subtype_in );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_CROT( CROT*  instance ) {
    delete instance;
    return;
}





extern "C"
{


/**
@brief Method called when a python instance of the class qgd_CROT_Wrapper is destroyed
@param self A pointer pointing to an instance of class qgd_CROT_Wrapper.
*/
static void
qgd_CROT_Wrapper_dealloc(qgd_CROT_Wrapper *self)
{

    // release the RY gate
    release_CROT( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class qgd_CROT_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_CROT_Wrapper.
*/
static PyObject *
qgd_CROT_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_CROT_Wrapper *self;
    self = (qgd_CROT_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_CROT_Wrapper is initialized
@param self A pointer pointing to an instance of the class qgd_CROT_Wrapper.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), Theta (bool) , Phi (bool), Lambda (bool)
@param kwds A tuple of keywords
*/
static int
qgd_CROT_Wrapper_init(qgd_CROT_Wrapper *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"qbit_num", (char*)"target_qbit", (char*)"control_qbit", (char*) "subtype",NULL};
    int  qbit_num = -1; 
    int target_qbit = -1;
    int control_qbit = -1;
    PyObject* subtype_arg = NULL;

    if (PyArray_API == NULL) {
        import_array();
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiiO", kwlist,
                                     &qbit_num, &target_qbit, &control_qbit,&subtype_arg))
        return -1;
    PyObject* subtype_string = PyObject_Str(subtype_arg);
    PyObject* subtype_string_unicode = PyUnicode_AsEncodedString(subtype_string, "utf-8", "~E~");
    const char* subtype_C = PyBytes_AS_STRING(subtype_string_unicode);
    crot_type qgd_subtype;
    if ( strcmp("control_r", subtype_C) == 0 || strcmp("CONTROL_R", subtype_C) == 0) {
        qgd_subtype = CONTROL_R;        
    }
    else if ( strcmp("control_opposite", subtype_C)==0 || strcmp("CONTROL_OPPOSITE", subtype_C)==0) {
        qgd_subtype = CONTROL_OPPOSITE;        
    }
    else if ( strcmp("control_independent", subtype_C)==0 || strcmp("CONTROL_INDEPENDENT", subtype_C)==0) {
        qgd_subtype = CONTROL_INDEPENDENT;        
    }
    if (qbit_num != -1 && target_qbit != -1) {
        self->gate = create_CROT( qbit_num, target_qbit, control_qbit,qgd_subtype );
    }
    return 0;
}




/**
@brief Call to apply the gate operation on the inut matrix
*/
static PyObject *
qgd_CROT_Wrapper_apply_to( qgd_CROT_Wrapper *self, PyObject *args ) {

    PyArrayObject * parameters_arr = NULL;
    PyArrayObject * unitary_arg = NULL;



    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|OO", &parameters_arr, &unitary_arg )) 
        return Py_BuildValue("i", -1);
    
    if ( PyArray_IS_C_CONTIGUOUS(parameters_arr) ) {
        Py_INCREF(parameters_arr);
    }
    else {
        parameters_arr = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }

    // get the C++ wrapper around the data
    Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );

    // convert python object array to numpy C API array
    if ( unitary_arg == NULL ) {
        PyErr_SetString(PyExc_Exception, "Input matrix was not given");
        return NULL;
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
    self->gate->apply_to( parameters_mtx, unitary_mtx, parallel );
    
    if (unitary_mtx.data != PyArray_DATA(unitary)) {
        memcpy(PyArray_DATA(unitary), unitary_mtx.data, unitary_mtx.size() * sizeof(QGD_Complex16));
    }

    Py_DECREF(parameters_arr);
    Py_DECREF(unitary);

    return Py_BuildValue("i", 0);
}



/**
@brief Call to get the number of free parameters in the gate
@return Returns with the starting index
*/
static PyObject *
qgd_CROT_Wrapper_get_Parameter_Num( qgd_CROT_Wrapper *self ) {

    int parameter_num = self->gate->get_parameter_num();

    return Py_BuildValue("i", parameter_num);

}

/**
@brief Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated
@return Returns with the starting index
*/
static PyObject *
qgd_CROT_Wrapper_get_Parameter_Start_Index( qgd_CROT_Wrapper *self ) {

    int start_index = self->gate->get_parameter_start_idx();

    return Py_BuildValue("i", start_index);

}


/**
@brief Call to get the target qbit
@return Returns with the target qbit
*/
static PyObject *
qgd_CROT_Wrapper_get_Target_Qbit( qgd_CROT_Wrapper *self ) {

    int target_qbit = self->gate->get_target_qbit();

    return Py_BuildValue("i", target_qbit);

}

/**
@brief Call to get the control qbit (returns with -1 if no control qbit is used in the gate)
@return Returns with the control qbit
*/
static PyObject *
qgd_CROT_Wrapper_get_Control_Qbit( qgd_CROT_Wrapper *self ) {

    int control_qbit = self->gate->get_control_qbit();

    return Py_BuildValue("i", control_qbit);

}

/**
@brief Call to set the target qbit
*/
static PyObject *
qgd_CROT_Wrapper_set_Target_Qbit( qgd_CROT_Wrapper *self, PyObject *args ) {
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
qgd_CROT_Wrapper_set_Control_Qbit( qgd_CROT_Wrapper *self, PyObject *args ) {
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
qgd_CROT_Wrapper_Extract_Parameters( qgd_CROT_Wrapper *self, PyObject *args ) {

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
@brief Structure containing metadata about the members of class qgd_CROT_Wrapper.
*/
static PyMemberDef qgd_CROT_Wrapper_members[] = {
    {NULL}  /* Sentinel */
};


/**
@brief Structure containing metadata about the methods of class qgd_CROT_Wrapper.
*/
static PyMethodDef qgd_CROT_Wrapper_methods[] = {
    {"apply_to", (PyCFunction) qgd_CROT_Wrapper_apply_to, METH_VARARGS,
     "Call to apply the gate on the input matrix."
    },
    {"get_Parameter_Num", (PyCFunction) qgd_CROT_Wrapper_get_Parameter_Num, METH_NOARGS,
     "Call to get the number of free parameters in the gate."
    },
    {"get_Parameter_Start_Index", (PyCFunction) qgd_CROT_Wrapper_get_Parameter_Start_Index, METH_NOARGS,
     "Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated."
    },
    {"get_Target_Qbit", (PyCFunction) qgd_CROT_Wrapper_get_Target_Qbit, METH_NOARGS,
     "Call to get the target qbit."
    },
    {"get_Control_Qbit", (PyCFunction) qgd_CROT_Wrapper_get_Control_Qbit, METH_NOARGS,
     "Call to get the control qbit (returns with -1 if no control qbit is used in the gate)."
    },
    {"set_Target_Qbit", (PyCFunction) qgd_CROT_Wrapper_set_Target_Qbit, METH_VARARGS,
     "Call to set the target qbit."
    },
    {"set_Control_Qbit", (PyCFunction) qgd_CROT_Wrapper_set_Control_Qbit, METH_VARARGS,
     "Call to set the control qbit."
    },
    {"Extract_Parameters", (PyCFunction) qgd_CROT_Wrapper_Extract_Parameters, METH_VARARGS,
     "Call to extract the paramaters corresponding to the gate, from a parameter array associated to the circuit in which the gate is embedded."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class qgd_CROT_Wrapper.
*/
static PyTypeObject  qgd_CROT_Wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_CROT_Wrapper.qgd_CROT_Wrapper", /*tp_name*/
  sizeof(qgd_CROT_Wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_CROT_Wrapper_dealloc, /*tp_dealloc*/
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
  qgd_CROT_Wrapper_methods, /*tp_methods*/
  qgd_CROT_Wrapper_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_CROT_Wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_CROT_Wrapper_new, /*tp_new*/
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
static PyModuleDef  qgd_CROT_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    "qgd_CROT_Wrapper",
    "Python binding for QGD RY gate",
    -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_CROT_Wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(& qgd_CROT_Wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(& qgd_CROT_Wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(& qgd_CROT_Wrapper_Type);
    if (PyModule_AddObject(m, "qgd_CROT_Wrapper", (PyObject *) & qgd_CROT_Wrapper_Type) < 0) {
        Py_DECREF(& qgd_CROT_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}




}
