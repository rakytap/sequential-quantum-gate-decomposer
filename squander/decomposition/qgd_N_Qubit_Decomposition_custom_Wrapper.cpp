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
\file qgd_N_Qubit_Decomposition_custom_Wrapper.cpp
\brief Python interface for the N_Qubit_Decomposition class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <stdio.h>
#include "N_Qubit_Decomposition_custom.h"
#include "Gates_block.h"

#include "numpy_interface.h"




/**
@brief Type definition of the qgd_Circuit_Wrapper Python class of the qgd_Circuit_Wrapper module
*/
typedef struct qgd_Circuit_Wrapper {
    PyObject_HEAD
    Gates_block* gate;
} qgd_Circuit_Wrapper;


/**
@brief Type definition of the qgd_N_Qubit_Decomposition_custom_Wrapper Python class of the qgd_N_Qubit_Decomposition_custom_Wrapper module
*/
typedef struct qgd_N_Qubit_Decomposition_custom_Wrapper {
    PyObject_HEAD
    /// pointer to the unitary to be decomposed to keep it alive
    PyArrayObject *Umtx;
    /// An object to decompose the unitary
    N_Qubit_Decomposition_custom* decomp;

} qgd_N_Qubit_Decomposition_custom_Wrapper;



/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param Umtx An instance of class Matrix containing the unitary to be decomposed
@param qbit_num Number of qubits spanning the unitary
@param optimize_layer_num Logical value. Set true to optimize the number of decomposing layers during the decomposition procedure, or false otherwise.
@param initial_guess Type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
N_Qubit_Decomposition_custom* 
create_N_Qubit_Decomposition_custom( Matrix& Umtx, int qbit_num, bool optimize_layer_num, guess_type initial_guess, std::map<std::string, Config_Element>& config, int accelerator_num ) {

    return new N_Qubit_Decomposition_custom( Umtx, qbit_num, optimize_layer_num, config, initial_guess, accelerator_num );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition_custom class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition_custom class.
*/
void
release_N_Qubit_Decomposition_custom( N_Qubit_Decomposition_custom*  instance ) {
    if (instance != NULL ) {
        delete instance;
    }
    return;
}



extern "C"
{


/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper is destroyed
@param self A pointer pointing to an instance of class qgd_N_Qubit_Decomposition_custom_Wrapper.
*/
static void
qgd_N_Qubit_Decomposition_custom_Wrapper_dealloc(qgd_N_Qubit_Decomposition_custom_Wrapper *self)
{

    if ( self->decomp != NULL ) {
        // deallocate the instance of class N_Qubit_Decomposition_custom
        release_N_Qubit_Decomposition_custom( self->decomp );
        self->decomp = NULL;
    }

    if ( self->Umtx != NULL ) {
        // release the unitary to be decomposed
        Py_DECREF(self->Umtx);    
        self->Umtx = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *) self);

}

/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_N_Qubit_Decomposition_custom_Wrapper *self;
    self = (qgd_N_Qubit_Decomposition_custom_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->decomp = NULL;
    self->Umtx = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper is initialized
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
@param args A tuple of the input arguments: Umtx (numpy array), qbit_num (integer), optimize_layer_num (bool), initial_guess (string PyObject 
@param kwds A tuple of keywords
*/
static int
qgd_N_Qubit_Decomposition_custom_Wrapper_init(qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"Umtx", (char*)"qbit_num", (char*)"initial_guess", (char*)"config", (char*)"accelerator_num", NULL};
 
    // initiate variables for input arguments
    PyObject *Umtx_arg = NULL;
    PyObject *config_arg = NULL;    
    int  qbit_num = -1; 
    PyObject *initial_guess = NULL;
    int accelerator_num = 0;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OiOOi", kwlist,
                                     &Umtx_arg, &qbit_num, &initial_guess, &config_arg, &accelerator_num))
        return -1;

    // convert python object array to numpy C API array
    if ( Umtx_arg == NULL ) return -1;
    self->Umtx = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)Umtx_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(self->Umtx) ) {
        std::cout << "Umtx is not memory contiguous" << std::endl;
    }


    // create QGD version of the Umtx
    Matrix Umtx_mtx = numpy2matrix(self->Umtx);  


    // determine the initial guess type
    PyObject* initial_guess_string = PyObject_Str(initial_guess);
    PyObject* initial_guess_string_unicode = PyUnicode_AsEncodedString(initial_guess_string, "utf-8", "~E~");
    const char* initial_guess_C = PyBytes_AS_STRING(initial_guess_string_unicode);

    guess_type qgd_initial_guess;
    if ( strcmp("zeros", initial_guess_C) == 0 || strcmp("ZEROS", initial_guess_C) == 0) {
        qgd_initial_guess = ZEROS;        
    }
    else if ( strcmp("random", initial_guess_C)==0 || strcmp("RANDOM", initial_guess_C)==0) {
        qgd_initial_guess = RANDOM;        
    }
    else if ( strcmp("close_to_zero", initial_guess_C)==0 || strcmp("CLOSE_TO_ZERO", initial_guess_C)==0) {
        qgd_initial_guess = CLOSE_TO_ZERO;        
    }
    else {
        std::cout << "Wrong initial guess format. Using default ZEROS." << std::endl; 
        qgd_initial_guess = ZEROS;     
    }


    // parse config and create C++ version of the hyperparameters

    // integer type config metadata utilized during the optimization
    std::map<std::string, Config_Element> config;

    // keys and values of the config dict
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(config_arg, &pos, &key, &value)) {

        // determine the initial guess type
        PyObject* key_string = PyObject_Str(key);
        PyObject* key_string_unicode = PyUnicode_AsEncodedString(key_string, "utf-8", "~E~");
        const char* key_C = PyBytes_AS_STRING(key_string_unicode);

        std::string key_Cpp( key_C );
        Config_Element element;

        if ( PyLong_Check( value ) ) { 
            element.set_property( key_Cpp, PyLong_AsLongLong( value ) );
            config[ key_Cpp ] = element;
        }
        else if ( PyFloat_Check( value ) ) {
            element.set_property( key_Cpp, PyFloat_AsDouble( value ) );
            config[ key_Cpp ] = element;
        }
        else {

        }

    }
  
    // create an instance of the class N_Qubit_Decomposition_custom
    if (qbit_num > 0 ) {
        try {
            self->decomp =  create_N_Qubit_Decomposition_custom( Umtx_mtx, qbit_num, false, qgd_initial_guess, config, accelerator_num);
        }
        catch (std::string err ) {
            PyErr_SetString(PyExc_Exception, err.c_str());
            return -1;
        }
    }
    else {
        std::cout << "The number of qubits should be given as a positive integer, " << qbit_num << "  was given" << std::endl;
        return -1;
    }



    Py_XDECREF(initial_guess_string);
    Py_XDECREF(initial_guess_string_unicode);

    return 0;
}

/**
@brief Wrapper function to call the start_decomposition method of C++ class N_Qubit_Decomposition
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_custom_Wrapper.
@param kwds A tuple of keywords
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_Start_Decomposition(qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {NULL};


    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|", kwlist))
        return Py_BuildValue("i", -1);


    // start the decomposition
    try {
        self->decomp->start_decomposition();
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


    return Py_BuildValue("i", 0);

}





/**
@brief Wrapper function to get the number of decomposing gates.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
@return Returns with the number of gates
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_get_gate_num( qgd_N_Qubit_Decomposition_custom_Wrapper *self ) {

    // get the number of gates
    int ret = self->decomp->get_gate_num();


    return Py_BuildValue("i", ret);

}







/**
@brief Wrapper function to retrieve the circuit (Squander format) incorporated in the instance.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_get_circuit( qgd_N_Qubit_Decomposition_custom_Wrapper *self ) {


    PyObject* qgd_Circuit  = PyImport_ImportModule("squander.gates.qgd_Circuit");

    if ( qgd_Circuit == NULL ) {
        PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_Circuit" );
        return NULL;
    }

    // retrieve the C++ variant of the flat circuit (flat circuit does not conatain any sub-circuits)
    Gates_block* circuit = self->decomp->get_flat_circuit();



    // construct python interfarce for the circuit
    PyObject* qgd_circuit_Dict  = PyModule_GetDict( qgd_Circuit );

    // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
    PyObject* py_circuit_class = PyDict_GetItemString( qgd_circuit_Dict, "qgd_Circuit");

    // create gate parameters
    PyObject* qbit_num     = Py_BuildValue("i",  circuit->get_qbit_num() );
    PyObject* circuit_input = Py_BuildValue("(O)", qbit_num);

    PyObject* py_circuit   = PyObject_CallObject(py_circuit_class, circuit_input);
    qgd_Circuit_Wrapper* py_circuit_C = reinterpret_cast<qgd_Circuit_Wrapper*>( py_circuit );

    
    // replace the empty circuit with the extracted one
    
    delete( py_circuit_C->gate );
    py_circuit_C->gate = circuit;


    return py_circuit;

}


/**
@brief Lists the gates decomposing the initial unitary. (These gates are the inverse gates of the gates bringing the intial matrix into unity.)
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_List_Gates( qgd_N_Qubit_Decomposition_custom_Wrapper *self ) {

    self->decomp->list_gates( 0 );

    return Py_None;
}




/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_get_Optimized_Parameters( qgd_N_Qubit_Decomposition_custom_Wrapper *self ) {

    int parameter_num = self->decomp->get_parameter_num();
    Matrix_real parameters_mtx(1, parameter_num);
    double* parameters = parameters_mtx.get_data();
    self->decomp->get_optimized_parameters(parameters);

    // convert to numpy array
    parameters_mtx.set_owner(false);
    PyObject * parameter_arr = matrix_real_to_numpy( parameters_mtx );

    return parameter_arr;
}





/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_set_Optimized_Parameters( qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args ) {

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


    Matrix_real parameters_mtx = numpy2matrix_real( parameters_arr );


    try {
        self->decomp->set_optimized_parameters(parameters_mtx.get_data(), parameters_mtx.size());
    }
    catch (std::string err ) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    Py_DECREF(parameters_arr);

    return Py_BuildValue("i", 0);
}





/**
@brief Set the maximal number of layers used in the subdecomposition of the qbit-th qubit.
@param max_layer_num A dictionary {'n': max_layer_num} labeling the maximal number of the gate layers used in the subdecomposition.
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_set_Max_Layer_Num(qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    PyObject* max_layer_num; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &max_layer_num )) return Py_BuildValue("i", -1);

    // Check whether input is dictionary
    if (!PyDict_Check(max_layer_num)) {
        printf("Input must be dictionary!\n");
        return Py_BuildValue("i", -1);
    }


    PyObject* key = NULL;
    PyObject* value = NULL;
    Py_ssize_t pos = 0;


    while (PyDict_Next(max_layer_num, &pos, &key, &value)) {

        // convert value fron PyObject to int
        assert(PyLong_Check(value) == 1);
        int value_int = (int) PyLong_AsLong(value);

        // convert keylue fron PyObject to int
        assert(PyLong_Check(key) == 1);
        int key_int = (int) PyLong_AsLong(key);

        // set maximal layer nums on the C++ side
        self->decomp->set_max_layer_num( key_int, value_int );
    }

    return Py_BuildValue("i", 0);
}






/**
@brief Set the number of iteration loops during the subdecomposition of the qbit-th qubit.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
@param args A tuple of the input arguments: identical_blocks (PyDict)
identical_blocks: A dictionary {'n': iteration_loops} labeling the number of successive identical layers used in the subdecomposition at the disentangling of the n-th qubit.
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_set_Iteration_Loops(qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    PyObject* iteration_loops; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &iteration_loops )) return Py_BuildValue("i", -1);

    // Check whether input is dictionary
    if (!PyDict_Check(iteration_loops)) {
        printf("Input must be dictionary!\n");
        return Py_BuildValue("i", -1);
    }


    PyObject* key = NULL;
    PyObject* value = NULL;
    Py_ssize_t pos = 0;


    while (PyDict_Next(iteration_loops, &pos, &key, &value)) {

        // convert value fron PyObject to int
        assert(PyLong_Check(value) == 1);
        int value_int = (int) PyLong_AsLong(value);

        // convert keylue fron PyObject to int
        assert(PyLong_Check(key) == 1);
        int key_int = (int) PyLong_AsLong(key);

        // set maximal layer nums on the C++ side
        self->decomp->set_iteration_loops( key_int, value_int );
    }

    return Py_BuildValue("i", 0);
}



/**
@brief Set the verbosity of the N_Qubit_Decomposition class
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
@param args A tuple of the input arguments: verbose (bool)
verbose: Set False to suppress the output messages of the decompostion, or True (deafult) otherwise.
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_set_Verbose(qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    int verbose; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|i", &verbose )) return Py_BuildValue("i", -1);


    // set maximal layer nums on the C++ side
    self->decomp->set_verbose( verbose );

    return Py_BuildValue("i", 0);
}

/**
@brief Set the debugfile name of the N_Qubit_Decomposition class
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param args A tuple of the input arguments: debugfile_name (string)
debug: Set True to suppress the output messages of the decompostion into a file named debugfile_name, or False (deafult) otherwise.
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_set_Debugfile(qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args ) {


    PyObject *debugfile = NULL;
 
    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &debugfile )) return Py_BuildValue("s", -1);

    // determine the debugfile name type
    PyObject* debugfile_string = PyObject_Str(debugfile);
    PyObject* debugfile_string_unicode = PyUnicode_AsEncodedString(debugfile_string, "utf-8", "~E~");
    const char* debugfile_C = PyBytes_AS_STRING(debugfile_string_unicode);

    
    Py_XDECREF(debugfile_string);
    Py_XDECREF(debugfile_string_unicode);

    // determine the length of the filename and initialize C++ variant of the string
    Py_ssize_t string_length = PyBytes_Size(debugfile_string_unicode);
    std::string debugfile_Cpp(debugfile_C, string_length);

    // set the name of the debugfile on the C++ side
    self->decomp->set_debugfile( debugfile_Cpp );


    return Py_BuildValue("s", NULL);
}



/**
@brief Wrapper method to set the optimization tolerance of the optimization process during the decomposition. 
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
@param args A tuple of the input arguments: tolerance (double)
tolerance: The maximal allowed error of the optimization problem
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_set_Optimization_Tolerance(qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args ) {
 
    // initiate variables for input arguments
    double tolerance; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|d", &tolerance )) return Py_BuildValue("i", -1);


    // set maximal layer nums on the C++ side
    self->decomp->set_optimization_tolerance( tolerance );

    return Py_BuildValue("i", 0);
}




/**
@brief Wrapper method to set the threshold of convergence in the optimization processes.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper.
@param args A tuple of the input arguments: tolerance (double)
tolerance: The maximal allowed error of the optimization problem
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_set_Convergence_Threshold(qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    double threshold; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|d", &threshold )) return Py_BuildValue("i", -1);


    // set maximal layer nums on the C++ side
    self->decomp->set_convergence_threshold( threshold );

    return Py_BuildValue("i", 0);
}


/**
@brief Wrapper method to to set the number of gate blocks to be optimized in one shot
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
@param args A tuple of the input arguments: tolerance (double)
optimization_block: number of operators in one sub-layer of the optimization process
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_set_Optimization_Blocks(qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    double optimization_block; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|d", &optimization_block )) return Py_BuildValue("i", -1);


    // set maximal layer nums on the C++ side
    self->decomp->set_optimization_blocks( optimization_block );

    return Py_BuildValue("i", 0);
}



/**
@brief Wrapper function to set custom gate structure for the decomposition.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
@param args A tuple of the input arguments: gate_structure_dict (PyDict)
gate_structure_dict: ?????????????????????????????
@return Returns with zero on success.
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_set_Gate_Structure( qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    PyObject* gate_structure_py; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &gate_structure_py )) return Py_BuildValue("i", -1);

    // convert gate structure from PyObject to qgd_Circuit_Wrapper
    qgd_Circuit_Wrapper* qgd_op_block = (qgd_Circuit_Wrapper*) gate_structure_py;

    self->decomp->set_custom_gate_structure( qgd_op_block->gate );

    return Py_BuildValue("i", 0);


}


/**
@brief Wrapper method to reorder the qubits in the decomposition class.
@param 
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_Reorder_Qubits(qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    PyObject* qbit_list; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &qbit_list )) return Py_BuildValue("i", -1);

    bool is_tuple = PyTuple_Check(qbit_list);
    bool is_list = PyList_Check(qbit_list);

    // Check whether input is dictionary
    if (!is_list && !is_tuple) {
        printf("Input must be tuple or list!\n");
        return Py_BuildValue("i", -1);
    }

    // get the number of qbubits
    Py_ssize_t element_num;

    if (is_tuple) {
        element_num = PyTuple_GET_SIZE(qbit_list);
    }
    else {
        element_num = PyList_GET_SIZE(qbit_list);
    }


    // create C++ variant of the tuple/list
    std::vector<int> qbit_list_C( (int) element_num);
    for ( Py_ssize_t idx=0; idx<element_num; idx++ ) {
        if (is_tuple) {        
            qbit_list_C[(int) idx] = (int) PyLong_AsLong( PyTuple_GetItem(qbit_list, idx ) );
        }
        else {
            qbit_list_C[(int) idx] = (int) PyLong_AsLong( PyList_GetItem(qbit_list, idx ) );
        }

    }


    // reorder the qubits in the decomposition class
    self->decomp->reorder_qubits( qbit_list_C );

    

    return Py_BuildValue("i", 0);
}



/**
@brief Wrapper function to set custom gate structure for the decomposition.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
@return Returns with zero on success.
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_set_Optimizer( qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"optimizer", NULL};

    PyObject* optimizer_arg = NULL;


    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &optimizer_arg)) {

        std::string err( "Unsuccessful argument parsing not ");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;       
 
    }


    if ( optimizer_arg == NULL ) {
        std::string err( "optimizer argument not set");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;        
    }

   

    PyObject* optimizer_string = PyObject_Str(optimizer_arg);
    PyObject* optimizer_string_unicode = PyUnicode_AsEncodedString(optimizer_string, "utf-8", "~E~");
    const char* optimizer_C = PyBytes_AS_STRING(optimizer_string_unicode);


    optimization_aglorithms qgd_optimizer;
    if ( strcmp("bfgs", optimizer_C) == 0 || strcmp("BFGS", optimizer_C) == 0) {
        qgd_optimizer = BFGS;        
    }
    else if ( strcmp("adam", optimizer_C)==0 || strcmp("ADAM", optimizer_C)==0) {
        qgd_optimizer = ADAM;        
    }
    else if ( strcmp("grad_descend", optimizer_C)==0 || strcmp("GRAD_DESCEND", optimizer_C)==0) {
        qgd_optimizer = GRAD_DESCEND;        
    }
    else if ( strcmp("adam_batched", optimizer_C)==0 || strcmp("ADAM_BATCHED", optimizer_C)==0) {
        qgd_optimizer = ADAM_BATCHED;        
    }
    else if ( strcmp("bfgs2", optimizer_C)==0 || strcmp("BFGS2", optimizer_C)==0) {
        qgd_optimizer = BFGS2;        
    }
    else if ( strcmp("agents", optimizer_C)==0 || strcmp("AGENTS", optimizer_C)==0) {
        qgd_optimizer = AGENTS;        
    }
    else if ( strcmp("cosine", optimizer_C)==0 || strcmp("COSINE", optimizer_C)==0) {
        qgd_optimizer = COSINE;        
    }
    else if ( strcmp("grad_descend_phase_shift_rule", optimizer_C)==0 || strcmp("GRAD_DESCEND_PARAMETER_SHIFT_RULE", optimizer_C)==0) {
        qgd_optimizer = GRAD_DESCEND_PARAMETER_SHIFT_RULE;        
    }  
    else if ( strcmp("agents_combined", optimizer_C)==0 || strcmp("AGENTS_COMBINED", optimizer_C)==0) {
        qgd_optimizer = AGENTS_COMBINED;        
    }
    else {
        std::cout << "Wrong optimizer: " << optimizer_C << ". Using default: BFGS" << std::endl; 
        qgd_optimizer = BFGS;     
    }


    try {
        self->decomp->set_optimizer(qgd_optimizer);
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


    return Py_BuildValue("i", 0);

}



/**
@brief Wrapper function to set a variant for the cost function. Input argument 0 stands for FROBENIUS_NORM, 1 for FROBENIUS_NORM_CORRECTION1, 2 for FROBENIUS_NORM_CORRECTION2, 3 for FROBENIUS_NORM_CORRECTION2_EXACT_DERIVATE
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@return Returns with zero on success.
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_set_Cost_Function_Variant( qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"costfnc", NULL};

    int costfnc_arg = 0;


    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &costfnc_arg)) {

        std::string err( "Unsuccessful argument parsing");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;       
 
    }
   
    cost_function_type qgd_costfnc = (cost_function_type)costfnc_arg;


    try {
        self->decomp->set_cost_function_variant(qgd_costfnc);
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


    return Py_BuildValue("i", 0);

}

//////////////////////////////////////////////////////

/**
@brief Wrapper function to evaluate the cost function.
@return teh value of the cost function
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_Optimization_Problem( qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args)
{


    PyArrayObject* parameters_arg = NULL;


    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arg )) {

        std::string err( "Unsuccessful argument parsing not ");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;      

    } 

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(parameters_arg) && PyArray_TYPE(parameters_arg) == NPY_FLOAT64 ){
        Py_INCREF(parameters_arg);
    }
    else if (PyArray_TYPE(parameters_arg) == NPY_FLOAT64 ) {
        parameters_arg = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)parameters_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    }
    else {
        std::string err( "Parameters should be should be real (given in float64 format)");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    Matrix_real parameters_mtx = numpy2matrix_real( parameters_arg );
    double f0;

    try {
        f0 = self->decomp->optimization_problem(parameters_mtx );
    }
    catch (std::string err ) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch (...) {
        std::string err( "Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    Py_DECREF(parameters_arg);


    return Py_BuildValue("d", f0);
}


/**
@brief Wrapper function to evaluate the cost function an dthe gradient components.
@return Unitarty numpy matrix
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_Optimization_Problem_Grad( qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args)
{


    PyArrayObject* parameters_arg = NULL;


    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arg )) {

        std::string err( "Unsuccessful argument parsing not ");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;      

    } 

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(parameters_arg) && PyArray_TYPE(parameters_arg) == NPY_FLOAT64 ){
        Py_INCREF(parameters_arg);
    }
    else if (PyArray_TYPE(parameters_arg) == NPY_FLOAT64 ) {
        parameters_arg = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)parameters_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    }
    else {
        std::string err( "Parameters should be should be real (given in float64 format)");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    Matrix_real parameters_mtx = numpy2matrix_real( parameters_arg );
    Matrix_real grad_mtx(parameters_mtx.size(), 1);

    try {
        self->decomp->optimization_problem_grad(parameters_mtx, self->decomp, grad_mtx );
    }
    catch (std::string err ) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch (...) {
        std::string err( "Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // convert to numpy array
    grad_mtx.set_owner(false);
    PyObject *grad_py = matrix_real_to_numpy( grad_mtx );

    Py_DECREF(parameters_arg);


    return grad_py;
}

/**
@brief Wrapper function to evaluate the cost function an dthe gradient components.
@return Unitarty numpy matrix
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_Optimization_Problem_Combined( qgd_N_Qubit_Decomposition_custom_Wrapper *self, PyObject *args)
{


    PyArrayObject* parameters_arg = NULL;


    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arg )) {

        std::string err( "Unsuccessful argument parsing not ");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;      

    } 

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(parameters_arg) && PyArray_TYPE(parameters_arg) == NPY_FLOAT64 ){
        Py_INCREF(parameters_arg);
    }
    else if (PyArray_TYPE(parameters_arg) == NPY_FLOAT64 ) {
        parameters_arg = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)parameters_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    }
    else {
        std::string err( "Parameters should be should be real (given in float64 format)");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    Matrix_real parameters_mtx = numpy2matrix_real( parameters_arg );
    Matrix_real grad_mtx(parameters_mtx.size(), 1);
    double f0;

    try {
        self->decomp->optimization_problem_combined(parameters_mtx, &f0, grad_mtx );
    }
    catch (std::string err ) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch (...) {
        std::string err( "Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // convert to numpy array
    grad_mtx.set_owner(false);
    PyObject *grad_py = matrix_real_to_numpy( grad_mtx );

    Py_DECREF(parameters_arg);


    PyObject* p = Py_BuildValue("(dO)", f0, grad_py);
    Py_DECREF(grad_py);
    return p;
}



/**
@brief Call to upload the unitary to the DFE. (Has no effect for non-DFE builds)
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_Upload_Umtx_to_DFE(qgd_N_Qubit_Decomposition_custom_Wrapper *self ) {

#ifdef __DFE__

    try {
        self->decomp->upload_Umtx_to_DFE();
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

#endif

    return Py_BuildValue("i", 0);
    
}



/**
@brief Structure containing metadata about the members of class qgd_N_Qubit_Decomposition_custom_Wrapper.
*/
static PyMemberDef qgd_N_Qubit_Decomposition_custom_Wrapper_members[] = {
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the methods of class qgd_N_Qubit_Decomposition_custom_Wrapper.
*/
static PyMethodDef qgd_N_Qubit_Decomposition_custom_Wrapper_methods[] = {
    {"Start_Decomposition", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_Start_Decomposition, METH_VARARGS | METH_KEYWORDS,
     "Method to start the decomposition."
    },
    {"get_Gate_Num", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_get_gate_num, METH_NOARGS,
     "Method to get the number of decomposing gates."
    },
    {"get_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_get_Optimized_Parameters, METH_NOARGS,
     "Method to get the array of optimized parameters."
    },
    {"set_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_set_Optimized_Parameters, METH_VARARGS,
     "Method to set the initial array of optimized parameters."
    },
    {"get_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_get_circuit, METH_NOARGS,
     "Method to get the incorporated circuit."
    },
    {"List_Gates", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_List_Gates, METH_NOARGS,
     "Call to print the decomposing nitaries on standard output"
    },
    {"set_Max_Layer_Num", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_set_Max_Layer_Num, METH_VARARGS,
     "Call to set the maximal number of layers used in the subdecomposition of the qbit-th qubit."
    },
    {"set_Iteration_Loops", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_set_Iteration_Loops, METH_VARARGS,
     "Call to set the number of iteration loops during the subdecomposition of the qbit-th qubit."
    },
    {"set_Verbose", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_set_Verbose, METH_VARARGS,
     "Call to set the verbosity of the qgd_N_Qubit_Decomposition class."
    },
    {"set_Debugfile", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_set_Debugfile, METH_VARARGS,
     "Set the debugfile name of the N_Qubit_Decomposition class."
    },
    {"set_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_set_Gate_Structure, METH_VARARGS,
     "Call to set custom gate structure in the decomposition."
    },
    {"Reorder_Qubits", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_Reorder_Qubits, METH_VARARGS,
     "Wrapper method to reorder the qubits in the decomposition class."
    },
    {"set_Optimization_Tolerance", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_set_Optimization_Tolerance, METH_VARARGS,
     "Wrapper method to set the optimization tolerance of the optimization process during the decomposition."
    },
    {"set_Convergence_Threshold", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_set_Convergence_Threshold, METH_VARARGS,
     "Wrapper method to set the threshold of convergence in the optimization processes."
    },
    {"set_Optimization_Blocks", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_set_Optimization_Blocks, METH_VARARGS,
     "Wrapper method to to set the number of gate blocks to be optimized in one shot."
    },
    {"set_Optimizer", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_set_Optimizer, METH_VARARGS | METH_KEYWORDS,
     "Wrapper method to to set the optimizer method for the gate synthesis."
    },
    {"Upload_Umtx_to_DFE", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_Upload_Umtx_to_DFE, METH_NOARGS,
     "Call to upload the unitary to the DFE. (Has no effect for non-DFE builds)"
    },
    {"set_Cost_Function_Variant", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_set_Cost_Function_Variant, METH_VARARGS | METH_KEYWORDS,
     "Wrapper method to to set the variant of the cost function. Input argument 0 stands for FROBENIUS_NORM, 1 for FROBENIUS_NORM_CORRECTION1, 2 for FROBENIUS_NORM_CORRECTION2, 3 for HILBERT_SCHMIDT_TEST, 4 for HILBERT_SCHMIDT_TEST_CORRECTION1, 5 for HILBERT_SCHMIDT_TEST_CORRECTION2."
    },
    {"Optimization_Problem", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_Optimization_Problem, METH_VARARGS,
     "Wrapper function to evaluate the cost function."
    },
    {"Optimization_Problem_Grad", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_Optimization_Problem_Grad, METH_VARARGS,
     "Wrapper function to evaluate the gradient components."
    },
    {"Optimization_Problem_Combined", (PyCFunction) qgd_N_Qubit_Decomposition_custom_Wrapper_Optimization_Problem_Combined, METH_VARARGS,
     "Wrapper function to evaluate the cost function and the gradient components."
    },
    {NULL}  /* Sentinel */
};

/**
@brief A structure describing the type of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
*/
static PyTypeObject qgd_N_Qubit_Decomposition_custom_Wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_N_Qubit_Decomposition_custom_Wrapper.qgd_N_Qubit_Decomposition_custom_Wrapper", /*tp_name*/
  sizeof(qgd_N_Qubit_Decomposition_custom_Wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_N_Qubit_Decomposition_custom_Wrapper_dealloc, /*tp_dealloc*/
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
  "Object to represent a Circuit class of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  qgd_N_Qubit_Decomposition_custom_Wrapper_methods, /*tp_methods*/
  qgd_N_Qubit_Decomposition_custom_Wrapper_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_N_Qubit_Decomposition_custom_Wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_N_Qubit_Decomposition_custom_Wrapper_new, /*tp_new*/
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
static PyModuleDef qgd_N_Qubit_Decomposition_custom_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    "qgd_N_Qubit_Decomposition_custom_Wrapper",
    "Python binding for QGD N_Qubit_Decomposition class",
    -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_N_Qubit_Decomposition_custom_Wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&qgd_N_Qubit_Decomposition_custom_Wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&qgd_N_Qubit_Decomposition_custom_Wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&qgd_N_Qubit_Decomposition_custom_Wrapper_Type);
    if (PyModule_AddObject(m, "qgd_N_Qubit_Decomposition_custom_Wrapper", (PyObject *) &qgd_N_Qubit_Decomposition_custom_Wrapper_Type) < 0) {
        Py_DECREF(&qgd_N_Qubit_Decomposition_custom_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}


} //extern C


