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
\file qgd_N_Qubit_Decomposition_adaptive_Wrapper.cpp
\brief Python interface for the N_Qubit_Decomposition class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <stdio.h>
#include "N_Qubit_Decomposition_adaptive.h"
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
@brief Type definition of the qgd_N_Qubit_Decomposition_adaptive_Wrapper Python class of the qgd_N_Qubit_Decomposition_adaptive_Wrapper module
*/
typedef struct qgd_N_Qubit_Decomposition_adaptive_Wrapper {
    PyObject_HEAD
    /// pointer to the unitary to be decomposed to keep it alive
    PyArrayObject* Umtx;
    /// An object to decompose the unitary
    N_Qubit_Decomposition_adaptive* decomp;
    /// An object to decompose the unitary

} qgd_N_Qubit_Decomposition_adaptive_Wrapper;



/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param Umtx An instance of class Matrix containing the unitary to be decomposed
@param qbit_num Number of qubits spanning the unitary
@param level_limit The maximal number of layers used in the decomposition
@param initial_guess Type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
N_Qubit_Decomposition_adaptive* 
create_N_Qubit_Decomposition_adaptive( Matrix& Umtx, int qbit_num, int level_limit, int level_limit_min, std::vector<matrix_base<int>> topology_in, std::map<std::string, Config_Element>& config, int accelerator_num ) {

    return new N_Qubit_Decomposition_adaptive( Umtx, qbit_num, level_limit, level_limit_min, topology_in, config, accelerator_num );
}




/**
@brief Call to deallocate an instance of N_Qubit_Decomposition_adaptive class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_N_Qubit_Decomposition_adaptive( N_Qubit_Decomposition_adaptive*  instance ) {

    if (instance != NULL ) {
        delete instance;
    }
    return;
}






extern "C"
{


/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper is destroyed
@param self A pointer pointing to an instance of class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
*/
static void
qgd_N_Qubit_Decomposition_adaptive_Wrapper_dealloc(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self)
{

    if ( self->decomp != NULL ) {
        // deallocate the instance of class N_Qubit_Decomposition
        release_N_Qubit_Decomposition_adaptive( self->decomp );
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
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_N_Qubit_Decomposition_adaptive_Wrapper *self;
    self = (qgd_N_Qubit_Decomposition_adaptive_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {

        self->decomp = NULL;
        self->Umtx = NULL;

    }

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper is initialized
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param args A tuple of the input arguments: Umtx (numpy array), qbit_num (integer), optimize_layer_num (bool), initial_guess (string PyObject 
@param kwds A tuple of keywords
*/
static int
qgd_N_Qubit_Decomposition_adaptive_Wrapper_init(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"Umtx", (char*)"qbit_num", (char*)"level_limit_min", (char*)"method", (char*)"topology", (char*)"config", (char*)"accelerator_num", NULL};
 
    // initiate variables for input arguments
    PyArrayObject *Umtx_arg = NULL;
    PyObject *config_arg = NULL;
    int  qbit_num = -1; 
    int level_limit = 0;
    int level_limit_min = 0;
    PyObject *topology = NULL;
    int accelerator_num = 0;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OiiiOOi", kwlist,
                                     &Umtx_arg, &qbit_num, &level_limit, &level_limit_min, &topology, &config_arg, &accelerator_num))
        return -1;

    // convert python object array to numpy C API array
    if ( Umtx_arg == NULL ) return -1;
    self->Umtx = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*) Umtx_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(self->Umtx) ) {
        std::cout << "Umtx is not memory contiguous" << std::endl;
    }


    // create QGD version of the Umtx
    Matrix Umtx_mtx = numpy2matrix(self->Umtx);

    // elaborate connectivity topology
    bool is_None = topology == Py_None;
    bool is_list = PyList_Check(topology);

    // Check whether input is a list
    if (!is_list && !is_None) {
        printf("Input topology must be a list!\n");
        return -1;
    }

    // create C++ variant of the list
    std::vector<matrix_base<int>> topology_Cpp;

    if ( !is_None ) {

        // get the number of qbubits
        Py_ssize_t element_num = PyList_GET_SIZE(topology);

        for ( Py_ssize_t idx=0; idx<element_num; idx++ ) {
            PyObject *item = PyList_GetItem(topology, idx );

            // Check whether input is a list
            if (!PyTuple_Check(item)) {
                printf("Elements of topology must be a tuple!\n");
                return -1;
            }

            matrix_base<int> item_Cpp(1,2);  
            item_Cpp[0] = (int) PyLong_AsLong( PyTuple_GetItem(item, 0 ) );
            item_Cpp[1] = (int) PyLong_AsLong( PyTuple_GetItem(item, 1 ) );

            topology_Cpp.push_back( item_Cpp );        
        }
    }


    // parse config and create C++ version of the hyperparameters

    bool is_dict = PyDict_Check( config_arg );
    if (!is_dict) {
        printf("Config object must be a python dictionary!\n");
        return -1;
    }

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


    // create an instance of the class N_Qubit_Decomposition
    if (qbit_num > 0 ) {
        try {
            self->decomp = create_N_Qubit_Decomposition_adaptive( Umtx_mtx, qbit_num, level_limit, level_limit_min, topology_Cpp, config, accelerator_num);
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



    return 0;
}

/**
@brief Wrapper function to call the start_decomposition method of C++ class N_Qubit_Decomposition
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param kwds A tuple of keywords
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_Start_Decomposition(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {NULL};


    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|", kwlist))
        return Py_BuildValue("i", -1);

    // starting the decomposition
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

static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Initial_Circuit(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self)
{

    // starting the decomposition
    try {
        self->decomp->get_initial_circuit();
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

static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_Compress_Circuit(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self)
{

    // starting the decomposition
    try {
        self->decomp->compress_circuit();
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

static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_Finalize_Circuit(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {NULL};


    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|", kwlist))
        return Py_BuildValue("i", -1);

    // starting the decomposition
    try {
        self->decomp->finalize_circuit();
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
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@return Returns with the number of gates
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_gate_num( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

    // get the number of gates
    int ret = self->decomp->get_gate_num();


    return Py_BuildValue("i", ret);

}







/**
@brief returns the angle of the global phase (the radius us always sqrt(2))
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Global_Phase(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

    QGD_Complex16 global_phase_factor_C = self->decomp->get_global_phase_factor();
    PyObject* global_phase = PyFloat_FromDouble( std::atan2(global_phase_factor_C.imag,global_phase_factor_C.real));

    return global_phase;
    
}

/**
@brief sets the global phase to the new angle given
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param arg global_phase_factor_new_angle the angle to be set
*/
static PyObject * qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Global_Phase(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args) {

    double new_global_phase;
    if (!PyArg_ParseTuple(args, "|d", &new_global_phase )) return Py_BuildValue("i", -1);
    self->decomp->set_global_phase(new_global_phase);

    return Py_BuildValue("i", 0);
    
}

/**
@brief applies the global phase to the Unitary matrix
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
*/
static PyObject * qgd_N_Qubit_Decomposition_adaptive_Wrapper_apply_Global_Phase_Factor(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

    // get the number of gates
    self->decomp->apply_global_phase_factor();

    return Py_BuildValue("i", 0);
    
}


/**
@brief Wrapper function to retrieve the circuit (Squander format) incorporated in the instance.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_circuit( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {


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
qgd_N_Qubit_Decomposition_adaptive_Wrapper_List_Gates( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

    self->decomp->list_gates( 0 );

    return Py_None;
}




/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Optimized_Parameters( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

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
@brief Get the number of free parameters in the gate structure used for the decomposition
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Parameter_Num( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

    int parameter_num = self->decomp->get_parameter_num();

    return Py_BuildValue("i", parameter_num);
}

/**
@brief Get the number of free parameters in the gate structure used for the decomposition
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Num_of_Iters( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

    int number_of_iters = self->decomp->get_num_iters();
    
    return Py_BuildValue("i", number_of_iters);
}


/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Optimized_Parameters( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

    PyArrayObject* parameters_arr = NULL;


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
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Max_Layer_Num(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

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
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param args A tuple of the input arguments: identical_blocks (PyDict)
identical_blocks: A dictionary {'n': iteration_loops} labeling the number of successive identical layers used in the subdecomposition at the disentangling of the n-th qubit.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Iteration_Loops(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

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
@brief Set the number of maximum iterations.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param args  (int) number of max iters.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Max_Iterations(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    int max_iters_input; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|i", &max_iters_input )) return Py_BuildValue("i", -1);


    //set the maximum number of iterations
    self->decomp->set_max_inner_iterations(max_iters_input);


    return Py_BuildValue("i", 0);
}

/**
@brief Set the verbosity of the N_Qubit_Decomposition class
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param args A tuple of the input arguments: verbose (int)
verbose: Set False to suppress the output messages of the decompostion, or True (deafult) otherwise.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Verbose(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

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
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Debugfile(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {
  

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
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param args A tuple of the input arguments: tolerance (double)
tolerance: The maximal allowed error of the optimization problem
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Optimization_Tolerance(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

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
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Convergence_Threshold(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

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
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param args A tuple of the input arguments: tolerance (double)
optimization_block: number of operators in one sub-layer of the optimization process
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Optimization_Blocks(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

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
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@return Returns with zero on success.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Gate_Structure( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    PyObject* gate_structure_py; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &gate_structure_py )) return Py_BuildValue("i", -1);


    // convert gate structure from PyObject to qgd_Circuit_Wrapper
    qgd_Circuit_Wrapper* qgd_op_block = (qgd_Circuit_Wrapper*) gate_structure_py;

    try {
        self->decomp->set_custom_gate_structure( qgd_op_block->gate );
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
    

    return Py_BuildValue("i", 0);


}



/**
@brief Wrapper function to append custom layers to the gate structure that are intended to be used in the decomposition.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_add_Gate_Structure_From_Binary( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {



    // initiate variables for input arguments
    PyObject* filename_py=NULL; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &filename_py )) return Py_BuildValue("i", -1);

    // determine the optimizaton method
    PyObject* filename_string = PyObject_Str(filename_py);
    PyObject* filename_string_unicode = PyUnicode_AsEncodedString(filename_string, "utf-8", "~E~");
    const char* filename_C = PyBytes_AS_STRING(filename_string_unicode);
    std::string filename_str( filename_C );


    try {
        self->decomp->add_adaptive_gate_structure( filename_str );
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


    return Py_BuildValue("i", 0);

}



/**
@brief Wrapper function to set custom layers to the gate structure that are intended to be used in the decomposition.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Gate_Structure_From_Binary( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {



    // initiate variables for input arguments
    PyObject* filename_py=NULL; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &filename_py )) return Py_BuildValue("i", -1);

    // determine the optimizaton method
    PyObject* filename_string = PyObject_Str(filename_py);
    PyObject* filename_string_unicode = PyUnicode_AsEncodedString(filename_string, "utf-8", "~E~");
    const char* filename_C = PyBytes_AS_STRING(filename_string_unicode);
    std::string filename_str( filename_C );


    try {
        self->decomp->set_adaptive_gate_structure( filename_str );
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



    return Py_BuildValue("i", 0);

}

static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Unitary_From_Binary(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ){
    // initiate variables for input arguments
    PyObject* filename_py=NULL; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &filename_py )) return Py_BuildValue("i", -1);
    
    PyObject* filename_string = PyObject_Str(filename_py);
    PyObject* filename_string_unicode = PyUnicode_AsEncodedString(filename_string, "utf-8", "~E~");
    const char* filename_C = PyBytes_AS_STRING(filename_string_unicode);
    std::string filename_str( filename_C );


    try {
        self->decomp->set_unitary_from_file( filename_str );
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


    return Py_BuildValue("i", 0);
}

/**
@brief Wrapper function to add finalyzing layer (single qubit rotations on all of the qubits) to the gate structure.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_add_Finalyzing_Layer_To_Gate_Structure( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {


    try {
        self->decomp->add_finalyzing_layer();
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



    return Py_BuildValue("i", 0);

}



/**
@brief Wrapper function to apply the imported gate structure on the unitary. The transformed unitary is to be decomposed in the calculations, and the imported gate structure is released.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_apply_Imported_Gate_Structure( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {



    try {
        self->decomp->apply_imported_gate_structure();
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




    return Py_BuildValue("i", 0);

}

/**
@brief get project name 
@return string name of the project
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Project_Name( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self) {


    std::string project_name = self->decomp->get_project_name();
    
    // convert to python string
    PyObject* project_name_pyhton = PyUnicode_FromString(project_name.c_str());

    return project_name_pyhton;
}

/**
@brief set project name 
@param project_name_new new string to be set as project name
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Project_Name( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {
    // initiate variables for input arguments
    PyObject* project_name_new=NULL; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &project_name_new)) return Py_BuildValue("i", -1);
    
    
    PyObject* project_name_new_string = PyObject_Str(project_name_new);
    PyObject* project_name_new_unicode = PyUnicode_AsEncodedString(project_name_new_string, "utf-8", "~E~");
    const char* project_name_new_C = PyBytes_AS_STRING(project_name_new_unicode);
    std::string project_name_new_str = ( project_name_new_C );
    
    // convert to python string
    self->decomp->set_project_name(project_name_new_str);

   return Py_BuildValue("i", 0);
}


/**
@brief Cll to get the error of the decomposition (i.e. the final value of the cost function)
@return The error of the decomposition
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Decomposition_Error( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self) {


    double decomposition_error = self->decomp->get_decomposition_error();
    

    return Py_BuildValue("d", decomposition_error);
}

/**
@brief export unitary to binary file
@param filename file to be exported to
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_export_Unitary( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {
    // initiate variables for input arguments
    PyObject* filename=NULL; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &filename)) return Py_BuildValue("i", -1);
    
    
    PyObject* filename_string = PyObject_Str(filename);
    PyObject* filename_unicode = PyUnicode_AsEncodedString(filename_string, "utf-8", "~E~");
    const char* filename_C = PyBytes_AS_STRING(filename_unicode);
    std::string filename_str = ( filename_C );

    // convert to python string
    self->decomp->export_unitary(filename_str);

   return Py_BuildValue("i", 0);
}

/**
@brief get Unitary
@return Unitarty numpy matrix
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Unitary( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self) {


    Matrix Unitary_mtx;

    try {
        Unitary_mtx = self->decomp->get_Umtx().copy();
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
    Unitary_mtx.set_owner(false);
    PyObject *Unitary_py = matrix_to_numpy( Unitary_mtx );

    return Unitary_py;
}


/**
@brief Wrapper function to evaluate the cost function.
@return teh value of the cost function
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_Optimization_Problem( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_adaptive_Wrapper_Optimization_Problem_Grad( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_adaptive_Wrapper_Optimization_Problem_Combined( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args)
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
@brief Wrapper function to evaluate the unitary function and the unitary derivates.
@return Unitarty numpy matrix
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_Optimization_Problem_Combined_Unitary( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args)
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
    Matrix Umtx;
    std::vector<Matrix> Umtx_deriv;

    try {
        self->decomp->optimization_problem_combined_unitary(parameters_mtx, Umtx, Umtx_deriv );
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
    Umtx.set_owner(false);
    PyObject *unitary_py = matrix_to_numpy( Umtx );
    PyObject* graduni_py = PyList_New(Umtx_deriv.size());
    for (size_t i = 0; i < Umtx_deriv.size(); i++) {
        Umtx_deriv[i].set_owner(false);
        PyList_SetItem(graduni_py, i, matrix_to_numpy(Umtx_deriv[i]));
    }

    Py_DECREF(parameters_arg);


    PyObject* p = Py_BuildValue("(OO)", unitary_py, graduni_py);
    Py_DECREF(unitary_py); Py_DECREF(graduni_py);
    return p;
}

/**
@brief Wrapper function to evaluate the cost function an dthe gradient components.
@return Unitarty numpy matrix
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_Optimization_Problem_Batch( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args)
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
    Matrix_real result_mtx;

    try {
        std::vector<Matrix_real> parameters_vec;
        parameters_vec.resize(parameters_mtx.rows);
        for( int row_idx=0; row_idx<parameters_mtx.rows; row_idx++ ) {
            parameters_vec[row_idx] = Matrix_real( parameters_mtx.get_data() + row_idx*parameters_mtx.stride, 1, parameters_mtx.cols, parameters_mtx.stride );
        }
        result_mtx = self->decomp->optimization_problem_batched( parameters_vec );
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
    result_mtx.set_owner(false);
    PyObject *result_py = matrix_real_to_numpy( result_mtx );

    Py_DECREF(parameters_arg);

    return result_py;
}

static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Unitary( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

       if ( self->Umtx != NULL ) {
           // release the unitary to be decomposed
           Py_DECREF(self->Umtx);    
           self->Umtx = NULL;
       }

       PyArrayObject *Umtx_arg = NULL;
       //Parse arguments 
       if (!PyArg_ParseTuple(args, "|O", &Umtx_arg )) return Py_BuildValue("i", -1);
	   
       // convert python object array to numpy C API array
       if ( Umtx_arg == NULL ) {
           PyErr_SetString(PyExc_Exception, "Umtx argument in empty");
           return NULL;
       }
	
	self->Umtx = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)Umtx_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

	// test C-style contiguous memory allocation of the array
	if ( !PyArray_IS_C_CONTIGUOUS(self->Umtx) ) {
	    std::cout << "Umtx is not memory contiguous" << std::endl;
	}


	// create QGD version of the Umtx
	Matrix Umtx_mtx = numpy2matrix(self->Umtx);
    self->decomp->set_unitary(Umtx_mtx);

    return Py_BuildValue("i", 0);
}

/**
@brief Wrapper method to reorder the qubits in the decomposition class.
@param 
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_Reorder_Qubits(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

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
@brief Wrapper method to add adaptive layers to the gate structure stored by the class.
@param 
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_add_Adaptive_Layers(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {


    self->decomp->add_adaptive_layers();
    

    return Py_BuildValue("i", 0);
}



/**
@brief Wrapper method to reorder the qubits in the decomposition class.
@param 
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_add_Layer_To_Imported_Gate_Structure(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {


    self->decomp->add_layer_to_imported_gate_structure();
    

    return Py_BuildValue("i", 0);
}





/**
@brief Retrieve the unified unitary operation of the circuit.
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Matrix( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

    PyArrayObject* parameters_arr = NULL;


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


    Matrix unitary_mtx;

    unitary_mtx = self->decomp->get_matrix( parameters_mtx );
    
    
    // convert to numpy array
    unitary_mtx.set_owner(false);
    PyObject *unitary_py = matrix_to_numpy( unitary_mtx );


    Py_DECREF(parameters_arr);

    return unitary_py;
}



/**
@brief Wrapper function to set custom gate structure for the decomposition.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@return Returns with zero on success.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Optimizer( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args, PyObject *kwds)
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
    else if ( strcmp("bayes_opt", optimizer_C)==0 || strcmp("BAYES_OPT", optimizer_C)==0) {
        qgd_optimizer = BAYES_OPT;        
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
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Cost_Function_Variant( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args, PyObject *kwds)
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






/**
@brief Wrapper function to set the trace offset used in the cost function. In this case Tr(A) = sum_(i-offset=j) A_{ij}
@return Returns with zero on success.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Trace_Offset( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"trace_offset", NULL};

    int trace_offset_arg = 0;


    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &trace_offset_arg)) {

        std::string err( "Unsuccessful argument parsing");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;       
 
    }
   

    try {
        self->decomp->set_trace_offset(trace_offset_arg);
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
@brief Wrapper function to get the trace offset used in the cost function. In this case Tr(A) = sum_(i-offset=j) A_{ij}
@return Returns with the trace offset
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Trace_Offset( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self )
{
   
    int trace_offset = 0;

    try {
        trace_offset = self->decomp->get_trace_offset();
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


    return Py_BuildValue("i", trace_offset);

}




/**
@brief Call to upload the unitary to the DFE. (Has no effect for non-DFE builds)
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_Upload_Umtx_to_DFE(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

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
@brief Wrapper function to evaluate the second Rényi entropy of a quantum circuit at a specific parameter set.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Second_Renyi_Entropy( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args)
{


    PyArrayObject * parameters_arr = NULL;
    PyArrayObject * input_state_arg = NULL;
    PyObject * qubit_list_arg = NULL;


    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|OOO", &parameters_arr, &input_state_arg, &qubit_list_arg )) 
        return Py_BuildValue("i", -1);

    
    if ( PyArray_IS_C_CONTIGUOUS(parameters_arr) ) {
        Py_INCREF(parameters_arr);
    }
    else {
        parameters_arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }

    // get the C++ wrapper around the data
    Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );


    // convert python object array to numpy C API array
    if ( input_state_arg == NULL ) {
        PyErr_SetString(PyExc_Exception, "Input matrix was not given");
        return NULL;
    }

    PyArrayObject* input_state = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)input_state_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(input_state) ) {
        PyErr_SetString(PyExc_Exception, "input mtrix is not memory contiguous");
        return NULL;
    }


    // create QGD version of the input matrix
    Matrix input_state_mtx = numpy2matrix(input_state);


    // check input argument qbit_list
    if ( qubit_list_arg == NULL || (!PyList_Check( qubit_list_arg )) ) {
        PyErr_SetString(PyExc_Exception, "qubit_list should be a list");
        return NULL;
    }

    Py_ssize_t reduced_qbit_num = PyList_Size( qubit_list_arg );

    matrix_base<int> qbit_list_mtx( (int)reduced_qbit_num, 1);
    for ( int idx=0; idx<reduced_qbit_num; idx++ ) {

        PyObject* item = PyList_GET_ITEM( qubit_list_arg, idx );
        qbit_list_mtx[idx] = (int) PyLong_AsLong( item );

    }


    double entropy = -1;


    try {
        entropy = self->decomp->get_second_Renyi_entropy( parameters_mtx, input_state_mtx, qbit_list_mtx );
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


    Py_DECREF(parameters_arr);
    Py_DECREF(input_state);



    PyObject* p = Py_BuildValue("d", entropy);

    return p;
}


/**
@brief Call to retrieve the number of qubits in the circuit
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Qbit_Num(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

    int qbit_num = 0;

    try {
        qbit_num = self->decomp->get_qbit_num();
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


    return Py_BuildValue("i", qbit_num );
    
}



/**
@brief Structure containing metadata about the members of class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
*/
static PyMemberDef qgd_N_Qubit_Decomposition_adaptive_Wrapper_members[] = {
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the methods of class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
*/
static PyMethodDef qgd_N_Qubit_Decomposition_adaptive_Wrapper_methods[] = {
    {"Start_Decomposition", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_Start_Decomposition, METH_VARARGS | METH_KEYWORDS,
     "Method to start the decomposition."
    },
    {"get_Initial_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Initial_Circuit, METH_NOARGS,
     "Method to get initial circuit in decomposition."
    },
    {"Compress_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_Compress_Circuit, METH_NOARGS,
     "Method to compress gate structure."
    },
    {"Finalize_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_Finalize_Circuit, METH_VARARGS | METH_KEYWORDS,
     "Method to finalize the decomposition."
    },
    {"get_Gate_Num", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_gate_num, METH_NOARGS,
     "Method to get the number of decomposing gates."
    },
    {"get_Parameter_Num", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Parameter_Num, METH_NOARGS,
     "Call to get the number of free parameters in the gate structure used for the decomposition"
    },
    {"get_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Optimized_Parameters, METH_NOARGS,
     "Method to get the array of optimized parameters."
    },
    {"set_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Optimized_Parameters, METH_VARARGS,
     "Method to set the initial array of optimized parameters."
    },
    {"get_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_circuit, METH_NOARGS,
     "Method to get the incorporated circuit."
    },
    {"List_Gates", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_List_Gates, METH_NOARGS,
     "Call to print the decomposing nitaries on standard output"
    },
    {"get_Num_of_Iters", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Num_of_Iters, METH_NOARGS,
     "Method to get the number of iterations."
    },
    {"set_Max_Layer_Num", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Max_Layer_Num, METH_VARARGS,
     "Call to set the maximal number of layers used in the subdecomposition of the qbit-th qubit."
    },
    {"set_Iteration_Loops", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Iteration_Loops, METH_VARARGS,
     "Call to set the number of iteration loops during the subdecomposition of the qbit-th qubit."
    },
    {"set_Verbose", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Verbose, METH_VARARGS,
     "Call to set the verbosity of the qgd_N_Qubit_Decomposition class."
    },
    {"set_Debugfile", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Debugfile, METH_VARARGS,
     "Set the debugfile name of the N_Qubit_Decomposition class."
    },
    {"set_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Gate_Structure, METH_VARARGS,
     "Call to set adaptive custom gate structure in the decomposition."
    },
    {"Reorder_Qubits", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_Reorder_Qubits, METH_VARARGS,
     "Wrapper method to reorder the qubits in the decomposition class."
    },
    {"set_Optimization_Tolerance", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Optimization_Tolerance, METH_VARARGS,
     "Wrapper method to set the optimization tolerance of the optimization process during the decomposition."
    },
    {"set_Convergence_Threshold", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Convergence_Threshold, METH_VARARGS,
     "Wrapper method to set the threshold of convergence in the optimization processes."
    },
    {"set_Optimization_Blocks", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Optimization_Blocks, METH_VARARGS,
     "Wrapper method to to set the number of gate blocks to be optimized in one shot."
    },
    {"set_Gate_Structure_From_Binary", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Gate_Structure_From_Binary, METH_VARARGS,
     "Call to set custom layers to the gate structure that are intended to be used in the decomposition from a binary file created from SQUANDER"
    },
    {"add_Gate_Structure_From_Binary", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_add_Gate_Structure_From_Binary, METH_VARARGS,
     "Call to append custom layers to the gate structure that are intended to be used in the decomposition from a binary file created from SQUANDER"
    },
    {"set_Unitary_From_Binary", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Unitary_From_Binary, METH_VARARGS,
     "Call to set unitary matrix from a binary file created from SQUANDER"
    },
    {"set_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Unitary, METH_VARARGS,
     "Call to set unitary matrix to a numpy matrix"
    },
    {"export_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_export_Unitary, METH_VARARGS,
     "Call to export unitary matrix to a binary file"
    },
    {"get_Project_Name", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Project_Name, METH_NOARGS,
     "Call to get the name of SQUANDER project"
    },
    {"set_Project_Name", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Project_Name, METH_VARARGS,
     "Call to set the name of SQUANDER project"
    },
    {"get_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Global_Phase, METH_NOARGS,
     "Call to get global phase"
    },
    {"set_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Global_Phase, METH_VARARGS,
     "Call to set global phase"
    },
    {"apply_Global_Phase_Factor", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_apply_Global_Phase_Factor, METH_NOARGS, 
     "Call to apply global phase on Unitary matrix"
    },
    {"add_Finalyzing_Layer_To_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_add_Finalyzing_Layer_To_Gate_Structure, METH_NOARGS,
     "Call to add finalyzing layer (single qubit rotations on all of the qubits) to the gate structure."
    },
    {"get_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Unitary, METH_NOARGS,
     "Call to get Unitary Matrix"
    },
    {"add_Adaptive_Layers", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_add_Adaptive_Layers, METH_NOARGS,
     "Call to add adaptive layers to the gate structure stored by the class."
    },
    {"add_Layer_To_Imported_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_add_Layer_To_Imported_Gate_Structure, METH_NOARGS,
     "Call to add an adaptive layer to the gate structure previously imported gate structure"
    },
    {"apply_Imported_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_apply_Imported_Gate_Structure, METH_NOARGS,
     "Call to apply the imported gate structure on the unitary. The transformed unitary is to be decomposed in the calculations, and the imported gate structure is released."
    },
    {"set_Optimizer", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Optimizer, METH_VARARGS | METH_KEYWORDS,
     "Wrapper method to to set the optimizer method for the gate synthesis."
    },
    {"set_Max_Iterations", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Max_Iterations, METH_VARARGS | METH_VARARGS,
     "Wrapper method to to set the maximum number of iterations for the gate synthesis."
    },
    {"get_Matrix", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Matrix, METH_VARARGS,
     "Method to retrieve the unitary of the circuit."
    },
    {"set_Cost_Function_Variant", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Cost_Function_Variant, METH_VARARGS | METH_KEYWORDS,
     "Wrapper method to to set the variant of the cost function. Input argument 0 stands for FROBENIUS_NORM, 1 for FROBENIUS_NORM_CORRECTION1, 2 for FROBENIUS_NORM_CORRECTION2, 3 for HILBERT_SCHMIDT_TEST, 4 for HILBERT_SCHMIDT_TEST_CORRECTION1, 5 for HILBERT_SCHMIDT_TEST_CORRECTION2."
    },
    {"Optimization_Problem", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_Optimization_Problem, METH_VARARGS,
     "Wrapper function to evaluate the cost function."
    },
    {"Optimization_Problem_Combined_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_Optimization_Problem_Combined_Unitary, METH_VARARGS,
     "Wrapper function to evaluate the unitary function and the gradient components."
    },	
    {"Optimization_Problem_Grad", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_Optimization_Problem_Grad, METH_VARARGS,
     "Wrapper function to evaluate the gradient components."
    },
    {"Optimization_Problem_Combined", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_Optimization_Problem_Combined, METH_VARARGS,
     "Wrapper function to evaluate the cost function and the gradient components."
    },
    {"Optimization_Problem_Batch", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_Optimization_Problem_Batch, METH_VARARGS,
     "Wrapper function to evaluate the cost function of batch."
    },
    {"Upload_Umtx_to_DFE", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_Upload_Umtx_to_DFE, METH_NOARGS,
     "Call to upload the unitary to the DFE. (Has no effect for non-DFE builds)"
    },
    {"get_Trace_Offset", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Trace_Offset, METH_NOARGS,
     "Call to get the trace offset used in the cost function. In this case Tr(A) = sum_(i-offset=j) A_{ij}"
    },
    {"set_Trace_Offset", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Trace_Offset, METH_VARARGS | METH_KEYWORDS,
     "Call to set the trace offset used in the cost function. In this case Tr(A) = sum_(i-offset=j) A_{ij}"
    },
    {"get_Decomposition_Error", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Decomposition_Error, METH_NOARGS,
     "Call to get the error of the decomposition. (i.e. the final value of the cost function)"
    },
    {"get_Second_Renyi_Entropy", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Second_Renyi_Entropy, METH_VARARGS,
     "Wrapper function to evaluate the second Rényi entropy of a quantum circuit at a specific parameter set."
    },
    {"get_Qbit_Num", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Qbit_Num, METH_NOARGS,
     "Call to get the number of qubits in the circuit"
    },
    {NULL}  /* Sentinel */
};

/**
@brief A structure describing the type of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
*/
static PyTypeObject qgd_N_Qubit_Decomposition_adaptive_Wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_N_Qubit_Decomposition_adaptive_Wrapper.qgd_N_Qubit_Decomposition_adaptive_Wrapper", /*tp_name*/
  sizeof(qgd_N_Qubit_Decomposition_adaptive_Wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_N_Qubit_Decomposition_adaptive_Wrapper_dealloc, /*tp_dealloc*/
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
  qgd_N_Qubit_Decomposition_adaptive_Wrapper_methods, /*tp_methods*/
  qgd_N_Qubit_Decomposition_adaptive_Wrapper_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_N_Qubit_Decomposition_adaptive_Wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_N_Qubit_Decomposition_adaptive_Wrapper_new, /*tp_new*/
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
static PyModuleDef qgd_N_Qubit_Decomposition_adaptive_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    "qgd_N_Qubit_Decomposition_adaptive_Wrapper",
    "Python binding for QGD N_Qubit_Decomposition class",
    -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_N_Qubit_Decomposition_adaptive_Wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&qgd_N_Qubit_Decomposition_adaptive_Wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&qgd_N_Qubit_Decomposition_adaptive_Wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&qgd_N_Qubit_Decomposition_adaptive_Wrapper_Type);
    if (PyModule_AddObject(m, "qgd_N_Qubit_Decomposition_adaptive_Wrapper", (PyObject *) &qgd_N_Qubit_Decomposition_adaptive_Wrapper_Type) < 0) {
        Py_DECREF(&qgd_N_Qubit_Decomposition_adaptive_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}


} //extern C

