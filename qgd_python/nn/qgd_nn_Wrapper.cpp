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
\file qgd_nn_Wrapper.cpp
\brief Python interface for the N_Qubit_Decomposition class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <stdio.h>
#include "NN.h"


#include "numpy_interface.h"



/**
@brief Type definition of the qgd_nn_Wrapper Python class of the qgd_nn_Wrapper module
*/
typedef struct qgd_nn_Wrapper {
    PyObject_HEAD
    /// pointer to the C++ side NN component
    NN* nn;

} qgd_nn_Wrapper;




/**
@brief Creates an instance of class NN and return with a pointer pointing to the class instance (C++ linking is needed)
@return Return with a void pointer pointing to an instance of NN class.
*/
NN* 
create_NN() {

    return new NN();
}




/**
@brief Call to deallocate an instance of N_Qubit_Decomposition_adaptive class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_NN( NN*  instance ) {

    if (instance != NULL ) {
        delete instance;
    }
    return;
}


extern "C"
{


/**
@brief Method called when a python instance of the class qgd_nn_Wrapper is destroyed
@param self A pointer pointing to an instance of class qgd_nn_Wrapper.
*/
static void
qgd_nn_Wrapper_dealloc(qgd_nn_Wrapper *self)
{


    if ( self->nn != NULL ) {
        // deallocate the instance of class N_Qubit_Decomposition
        release_NN( self->nn );
        self->nn = NULL;
    }
    
    Py_TYPE(self)->tp_free((PyObject *) self);

}

/**
@brief Method called when a python instance of the class qgd_nn_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_nn_Wrapper.
*/
static PyObject *
qgd_nn_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_nn_Wrapper *self;
    self = (qgd_nn_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {

        self->nn = NULL;

    }


    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_nn_Wrapper is initialized
@param self A pointer pointing to an instance of the class qgd_nn_Wrapper.
@param args A tuple of the input arguments: Umtx (numpy array), qbit_num (integer), optimize_layer_num (bool), initial_guess (string PyObject 
@param kwds A tuple of keywords
*/
static int
qgd_nn_Wrapper_init(qgd_nn_Wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {NULL};
 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|", kwlist ))
        return -1;


    try {
        self->nn = create_NN();
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        std::cout << err << std::endl;
        return 1;
    }
   


    return 0;
}

/**
@brief Wrapper function to call the start_decomposition method of C++ class N_Qubit_Decomposition
@param self A pointer pointing to an instance of the class qgd_nn_Wrapper.
@param args A tuple of the input arguments: finalize_decomp (bool), prepare_export (bool)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_nn_Wrapper_get_nn_chanels(qgd_nn_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"qbit_num", (char*)"levels", (char*)"samples_num", NULL};

    // initiate variables for input arguments
    int qbit_num = -1;
    int levels = -1;    
    int samples_num = -1;    
    


    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist, &qbit_num, &levels, &samples_num) ) {
        std::string err( "Invalid parameters" );
        PyErr_SetString(PyExc_Exception, err.c_str());
        std::cout << err << std::endl;
        return NULL;         
    }



    // preallocate output variables
    Matrix_real chanels(0,0);
    matrix_base<int8_t> nontrivial_adaptive_layers;
    

    // calculate the neural network chanels 
    try {
        if ( qbit_num > 0 && levels >= 0 && samples_num <= 1) {
            std::cout << "qbit_num: " << qbit_num << ", levels: " << levels << std::endl;
            
            // preallocate output variables
            //Matrix_real parameters(0,0);

            
            self->nn->get_nn_chanels(qbit_num, levels, chanels, nontrivial_adaptive_layers);
            
        }
        else if ( qbit_num > 0 && levels >= 0 && samples_num > 1) {
            std::cout << "qbit_num: " << qbit_num << ", levels: " << levels << ", samples num:" << samples_num << std::endl;
            
            // preallocate output variables
            //Matrix_real parameters(0,0);
            
            self->nn->get_nn_chanels(qbit_num, levels, samples_num, chanels, nontrivial_adaptive_layers);
            
        }
        else {
            std::string err( "Not enough input parameters");
            PyErr_SetString(PyExc_Exception, err.c_str());
            return NULL;        
        }
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        std::cout << err << std::endl;
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    
    
    PyObject* chanels_py = matrix_real_to_numpy( chanels );
    chanels.set_owner( false );      

    PyObject* nontrivial_adaptive_layers_py;
    if ( nontrivial_adaptive_layers.size() > 0 ) {
        nontrivial_adaptive_layers_py = matrix_int8_to_numpy( nontrivial_adaptive_layers );   
        nontrivial_adaptive_layers.set_owner( false );  
    }
    else {
        nontrivial_adaptive_layers_py = Py_None;
    }



    return Py_BuildValue("(OO)", chanels_py, nontrivial_adaptive_layers_py);
    //return chanels_py;



}







/**
@brief Structure containing metadata about the members of class qgd_nn_Wrapper.
*/
static PyMemberDef qgd_nn_Wrapper_members[] = {
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the methods of class qgd_nn_Wrapper.
*/
static PyMethodDef qgd_nn_Wrapper_methods[] = {
    {"get_NN_Chanels", (PyCFunction) qgd_nn_Wrapper_get_nn_chanels, METH_VARARGS | METH_KEYWORDS,
     "Method to retrieve the data chanels for the neural network."
    },
    {NULL}  /* Sentinel */
};

/**
@brief A structure describing the type of the class qgd_nn_Wrapper.
*/
static PyTypeObject qgd_nn_Wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_nn_Wrapper.qgd_nn_Wrapper", /*tp_name*/
  sizeof(qgd_nn_Wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_nn_Wrapper_dealloc, /*tp_dealloc*/
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
  "Object to represent a Gates_block class of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  qgd_nn_Wrapper_methods, /*tp_methods*/
  qgd_nn_Wrapper_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_nn_Wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_nn_Wrapper_new, /*tp_new*/
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
static PyModuleDef qgd_nn_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    "qgd_nn_Wrapper",
    "Python binding for the neural network component of SQUANDER",
    -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_nn_Wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&qgd_nn_Wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&qgd_nn_Wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&qgd_nn_Wrapper_Type);
    if (PyModule_AddObject(m, "qgd_nn_Wrapper", (PyObject *) &qgd_nn_Wrapper_Type) < 0) {
        Py_DECREF(&qgd_nn_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}


} //extern C

