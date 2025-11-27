/*
\file qgd_N_Qubit_Decompositions_Wrapper.cpp
\brief Python interface for N-Qubit Decomposition classes
*/
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <stdio.h>
#include <complex>
#include <cmath>
#include <cstring>
#include <cctype>

// Cross-platform case-insensitive string comparison
#ifdef _WIN32
    #define strcasecmp _stricmp
#else
    #include <strings.h>
#endif

#include "numpy_interface.h"
#include "N_Qubit_Decomposition.h"
#include "N_Qubit_Decomposition_adaptive.h"
#include "N_Qubit_Decomposition_Tree_Search.h"
#include "N_Qubit_Decomposition_Tabu_Search.h"
#include "Gates_block.h"

/**
@brief Type definition for qgd_Circuit_Wrapper
*/
typedef struct qgd_Circuit_Wrapper {
    PyObject_HEAD
    Gates_block* gate;
} qgd_Circuit_Wrapper;

/**
@brief Type definition of the unified N-Qubit Decomposition wrapper
*/
typedef struct qgd_N_Qubit_Decomposition_Wrapper {
    PyObject_HEAD
    /// pointer to the unitary to be decomposed to keep it alive
    PyArrayObject* Umtx;
    /// An object to decompose the unitary
    Optimization_Interface* decomp;
} qgd_N_Qubit_Decomposition_Wrapper;

////////////////////////////////////////////////////////////////// HELPER FUNCTIONS

/**
 * @brief Extract and validate Matrix from numpy array
 */
Matrix extract_matrix(PyObject* Umtx_arg, PyArrayObject** store_ref) {
    if (!Umtx_arg) {
        throw std::runtime_error("Umtx is NULL");
    }
    *store_ref = (PyArrayObject*)PyArray_FROM_OTF(Umtx_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (!*store_ref) {
        throw std::runtime_error("Failed to convert Umtx");
    }
    if (!PyArray_IS_C_CONTIGUOUS(*store_ref)) {
        std::cout << "Warning: Umtx is not memory contiguous" << std::endl;
    }
    return numpy2matrix(*store_ref);
}

/**
 * @brief Extract guess_type from Python string/object
 */
guess_type extract_guess_type(PyObject* initial_guess) {
    if (!initial_guess || initial_guess == Py_None) {
        return RANDOM;
    }

    PyObject* guess_str_obj = PyObject_Str(initial_guess);
    if (!guess_str_obj) {
        throw std::runtime_error("Failed to convert initial guess to string");
    }
    const char* guess_str = PyUnicode_AsUTF8(guess_str_obj);
    if (!guess_str) {
        throw std::runtime_error("Failed to convert initial guess to string");
    }
    
    if (strcasecmp("zeros", guess_str) == 0) return ZEROS;
    if (strcasecmp("random", guess_str) == 0) return RANDOM;
    if (strcasecmp("close_to_zero", guess_str) == 0) return CLOSE_TO_ZERO;
    std::cout << "Warning: Unknown guess '" << guess_str << "', using RANDOM" << std::endl;

    Py_XDECREF(guess_str_obj);
    return RANDOM;
}

/**
 * @brief Extract topology list from Python
 */
std::vector<matrix_base<int>> extract_topology(PyObject* topology) {
    std::vector<matrix_base<int>> result;
    if (!topology || topology == Py_None) {
        return result;
    }
    if (!PyList_Check(topology)) {
        throw std::runtime_error("Topology must be a list");
    }
    Py_ssize_t n = PyList_Size(topology);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PyList_GetItem(topology, i);
        if (!PyTuple_Check(item)) {
            throw std::runtime_error("Topology elements must be tuples");
        }
        matrix_base<int> pair(1, 2);
        pair[0] = PyLong_AsLong(PyTuple_GetItem(item, 0));
        pair[1] = PyLong_AsLong(PyTuple_GetItem(item, 1));
        result.push_back(pair);
    }
    return result;
}

/**
 * @brief Extract config dictionary
 */
std::map<std::string, Config_Element> extract_config(PyObject* config_arg) {
    std::map<std::string, Config_Element> config;
    if (!config_arg || config_arg == Py_None) {
        return config;
    }
    if (!PyDict_Check(config_arg)) {
        throw std::runtime_error("Config must be a dictionary");
    }
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(config_arg, &pos, &key, &value)) {
        std::string key_str = PyUnicode_AsUTF8(key);
        Config_Element element;
        if (PyLong_Check(value)) {
            element.set_property(key_str, PyLong_AsLongLong(value));
        } else if (PyFloat_Check(value)) {
            element.set_property(key_str, PyFloat_AsDouble(value));
        }
        config[key_str] = element;
    }
    return config;
}

//////////////////////////////////////////////////////////////////

static int 
qgd_N_Qubit_Decomposition_Wrapper_init(qgd_N_Qubit_Decomposition_Wrapper* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {
        (char*)"Umtx", (char*)"qbit_num", (char*)"optimize_layer_num", 
        (char*)"initial_guess", NULL
    };
    
    PyObject *Umtx_arg = NULL, *initial_guess = NULL; 
    int qbit_num = -1;
    bool optimize_layer_num = false;
    
    if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O|ibO", kwlist,
        &Umtx_arg, &qbit_num, &optimize_layer_num, &initial_guess)
    ) {
        return -1;
    }

    try {
        Matrix Umtx_mtx = extract_matrix(Umtx_arg, &self->Umtx);
        // calculate qbit_num from matrix size if not provided
        if (qbit_num == -1) {
            qbit_num = (int)std::round(std::log2(Umtx_mtx.rows));
        }
        
        guess_type guess = extract_guess_type(initial_guess);
        std::map<std::string, Config_Element> config; // empty for base

        self->decomp = new N_Qubit_Decomposition(Umtx_mtx, qbit_num, optimize_layer_num, config, guess);

        return 0;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return -1;
    }
}

static int 
qgd_N_Qubit_Decomposition_adaptive_Wrapper_init(qgd_N_Qubit_Decomposition_Wrapper* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {
        (char*)"Umtx", (char*)"qbit_num", (char*)"level_limit_max", 
        (char*)"level_limit_min", (char*)"topology", (char*)"config", 
        (char*)"accelerator_num", NULL
    };
    PyObject *Umtx_arg = NULL, *topology = NULL, *config_arg = NULL;
    int qbit_num = -1, level_limit = 8, level_limit_min = 0, accelerator_num = 0;
    
    if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O|iiiOOi", kwlist,
        &Umtx_arg, &qbit_num, &level_limit, &level_limit_min, &topology, &config_arg, &accelerator_num)
    ) {
        return -1;
    }

    try {
        Matrix Umtx_mtx = extract_matrix(Umtx_arg, &self->Umtx);
        
        // For state vector input: State Preparation passes (State, level_limit_max, level_limit_min, ...)
        // without qbit_num, so we calculate qbit_num from state size and interpret the qbit_num 
        // position as level_limit_max. Example: (State_16x1, 5, 0) -> qbit_num=4, level_limit_max=5
        if (Umtx_mtx.cols == 1 && qbit_num > 0) {
            int level_limit_max_in = qbit_num;
            qbit_num = (int)std::round(std::log2(Umtx_mtx.rows));
            level_limit = level_limit_max_in;
        }
        else {
            // For Unitary decomposition, calculate qbit_num from matrix size if not provided
            if (qbit_num == -1) {
                qbit_num = (int)std::round(std::log2(Umtx_mtx.rows));
            }
        }
        
        auto topology_cpp = extract_topology(topology);
        auto config = extract_config(config_arg);
        
        self->decomp = new N_Qubit_Decomposition_adaptive(
            Umtx_mtx, qbit_num, level_limit, level_limit_min, 
            topology_cpp, config, accelerator_num
        );
        
        return 0;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return -1;
    }
}

static int 
qgd_N_Qubit_Decomposition_custom_Wrapper_init(qgd_N_Qubit_Decomposition_Wrapper* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {
        (char*)"Umtx", (char*)"qbit_num", (char*)"initial_guess", 
        (char*)"config", (char*)"accelerator_num", NULL
    };
    
    PyObject *Umtx_arg = NULL, *initial_guess = NULL, *config_arg = NULL;
    int qbit_num = -1, accelerator_num = 0;
    
    if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O|iOOi", kwlist,
        &Umtx_arg, &qbit_num, &initial_guess, &config_arg, &accelerator_num)
    ) {
        return -1;
    }

    try {
        Matrix Umtx_mtx = extract_matrix(Umtx_arg, &self->Umtx);
        // calculate qbit_num from matrix size if not provided
        if (qbit_num == -1) {
            qbit_num = (int)std::round(std::log2(Umtx_mtx.rows));
        }
        
        guess_type guess = extract_guess_type(initial_guess);
        auto config = extract_config(config_arg);
        
        self->decomp = new N_Qubit_Decomposition_custom(Umtx_mtx, qbit_num, false, config, guess, accelerator_num);
        
        return 0;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return -1;
    }
}

template<typename DecompT>
static int search_wrapper_init(qgd_N_Qubit_Decomposition_Wrapper* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {
        (char*)"Umtx", (char*)"qbit_num", (char*)"topology", 
        (char*)"config", (char*)"accelerator_num", NULL
    };
    
    PyObject *Umtx_arg = NULL, *topology = NULL, *config_arg = NULL;
    int qbit_num = -1, accelerator_num = 0;
    
    if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O|iOOi", kwlist,
        &Umtx_arg, &qbit_num, &topology, &config_arg, &accelerator_num)
    ) {
        return -1;
    }
    
    try {
        Matrix Umtx_mtx = extract_matrix(Umtx_arg, &self->Umtx);
        // calculate qbit_num from matrix size if not provided
        if (qbit_num == -1) {
            qbit_num = (int)std::round(std::log2(Umtx_mtx.rows));
        }
        
        auto topology_cpp = extract_topology(topology);
        auto config = extract_config(config_arg);
        
        self->decomp = new DecompT(Umtx_mtx, qbit_num, topology_cpp, config, accelerator_num);
        
        return 0;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return -1;
    }
}

static int 
qgd_N_Qubit_Decomposition_Tree_Search_Wrapper_init(qgd_N_Qubit_Decomposition_Wrapper* self, PyObject* args, PyObject* kwds) {
    return search_wrapper_init<N_Qubit_Decomposition_Tree_Search>(self, args, kwds);
}

static int
qgd_N_Qubit_Decomposition_Tabu_Search_Wrapper_init(qgd_N_Qubit_Decomposition_Wrapper* self, PyObject* args, PyObject* kwds) {
    return search_wrapper_init<N_Qubit_Decomposition_Tabu_Search>(self, args, kwds);
}

/**
 * @brief Deallocate decomposition instance
 */
template<typename DecompT>
void release_decomposition(DecompT* instance) {
    if (instance != NULL) {
        delete instance;
    }
}

/**
 * @brief Called when Python object is destroyed
 */
static void
qgd_N_Qubit_Decomposition_Wrapper_dealloc(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    if (self->decomp != NULL) {
        // deallocate the instance of class N_Qubit_Decomposition
        release_decomposition(self->decomp);
        self->decomp = NULL;
    }
    if (self->Umtx != NULL) {
        // release the unitary to be decomposed
        Py_DECREF(self->Umtx);
        self->Umtx = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
 * @brief Allocate memory for new Python object
 */
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_N_Qubit_Decomposition_Wrapper *self;
    self = (qgd_N_Qubit_Decomposition_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->Umtx = NULL;
        self->decomp = NULL;
    }
    return (PyObject *) self;
}

////////////////////////////////////////////////////////////////// COMMON METHODs

/**
@brief Wrapper function to call the start_decomposition method of C++ class N_Qubit_Decomposition
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decompositions_Wrapper
@param kwds A tuple of keywords
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_Start_Decomposition(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {NULL};

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|", kwlist))
        return Py_BuildValue("i", -1);

    // Try each decomposition type and call start_decomposition
    if (N_Qubit_Decomposition_adaptive* p = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp)) {
        p->start_decomposition();
        return Py_BuildValue("i", 0);
    }
    if (N_Qubit_Decomposition_custom* p = dynamic_cast<N_Qubit_Decomposition_custom*>(self->decomp)) {
        p->start_decomposition();
        return Py_BuildValue("i", 0);
    }
    if (N_Qubit_Decomposition_Tree_Search* p = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp)) {
        p->start_decomposition();
        return Py_BuildValue("i", 0);
    }
    if (N_Qubit_Decomposition_Tabu_Search* p = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp)) {
        p->start_decomposition();
        return Py_BuildValue("i", 0);
    }
    if (N_Qubit_Decomposition* p = dynamic_cast<N_Qubit_Decomposition*>(self->decomp)) {
        p->start_decomposition();
        return Py_BuildValue("i", 0);
    }

    PyErr_SetString(PyExc_TypeError, "Unknown decomposition type");
    return NULL;
}

/**
@brief Call to get the number of gates
@return PyLong representing the number of gates
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Gate_Num(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    // get the number of gates
    int ret = self->decomp->get_gate_num();
    return Py_BuildValue("i", ret);
}

/**
@brief Call to get the optimized parameters
@return Numpy array of optimized parameters
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Optimized_Parameters(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    Matrix_real parameters_mtx = self->decomp->get_optimized_parameters();

    // convert to numpy array
    parameters_mtx.set_owner(false);
    PyObject* parameter_arr = matrix_real_to_numpy( parameters_mtx );

    return parameter_arr;
}

/**
@brief Call to get the incorporated circuit
@return Python wrapper object containing the circuit
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Circuit(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    PyObject* qgd_Circuit  = PyImport_ImportModule("squander.gates.qgd_Circuit");
    if ( qgd_Circuit == NULL ) {
        PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_Circuit" );
        return NULL;
    }

    // retrieve the C++ variant of the flat circuit (flat circuit does not conatain any sub-circuits)
    Gates_block* circuit = self->decomp->get_flat_circuit();

    // construct python interfarce for the circuit
    PyObject* qgd_circuit_Dict = PyModule_GetDict( qgd_Circuit );

    // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
    PyObject* py_circuit_class = PyDict_GetItemString( qgd_circuit_Dict, "qgd_Circuit");

    // create gate parameters
    PyObject* qbit_num = Py_BuildValue("i",  circuit->get_qbit_num() );
    PyObject* circuit_input = Py_BuildValue("(O)", qbit_num);

    PyObject* py_circuit = PyObject_CallObject(py_circuit_class, circuit_input);
    qgd_Circuit_Wrapper* py_circuit_C = reinterpret_cast<qgd_Circuit_Wrapper*>( py_circuit );
    
    // replace the empty circuit with the extracted one
    delete( py_circuit_C->gate );
    py_circuit_C->gate = circuit;

    return py_circuit;
}

/**
@brief Call to list the gates decomposing the unitary
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_List_Gates(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    // list gates with start_index = 0
    self->decomp->list_gates(0);
    return Py_BuildValue("");
}

/**
@brief Set the maximal number of layers used in the subdecomposition of the qbit-th qubit
@param args Dictionary {'n': max_layer_num} labeling the maximal number of the gate layers used in the subdecomposition
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Max_Layer_Num(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    // initiate variables for input arguments
    PyObject* max_layer_num;
    // parsing input arguments
    if (!PyArg_ParseTuple(args, "O", &max_layer_num)) {
        return NULL;
    }
    // Check whether input is dictionary
    if (!PyDict_Check(max_layer_num)) {
        PyErr_SetString(PyExc_TypeError, "Input must be dictionary");
        return NULL;
    }

    PyObject *key = NULL, *value = NULL;
    Py_ssize_t pos = 0;

    try {
        while (PyDict_Next(max_layer_num, &pos, &key, &value)) {
            // convert value from PyObject to int
            if (!PyLong_Check(value)) {
                PyErr_SetString(PyExc_TypeError, "Dictionary values must be integers");
                return NULL;
            }
            int value_int = (int)PyLong_AsLong(value);

            // convert key from PyObject to int
            if (!PyLong_Check(key)) {
                PyErr_SetString(PyExc_TypeError, "Dictionary keys must be integers");
                return NULL;
            }
            int key_int = (int)PyLong_AsLong(key);

            // set maximal layer nums on the C++ side (base class method)
            self->decomp->set_max_layer_num(key_int, value_int);
        }
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Set the number of iteration loops during the subdecomposition of the qbit-th qubit
@param args Dictionary {'n': iteration_loops} labeling the number of successive identical layers used in the subdecomposition at the disentangling of the n-th qubit
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Iteration_Loops(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    // initiate variables for input arguments
    PyObject* iteration_loops;
    // parsing input arguments
    if (!PyArg_ParseTuple(args, "O", &iteration_loops)) {
        return NULL;
    }
    // Check whether input is dictionary
    if (!PyDict_Check(iteration_loops)) {
        PyErr_SetString(PyExc_TypeError, "Input must be dictionary");
        return NULL;
    }

    PyObject *key = NULL, *value = NULL;
    Py_ssize_t pos = 0;

    try {
        while (PyDict_Next(iteration_loops, &pos, &key, &value)) {
            // convert value from PyObject to int
            if (!PyLong_Check(value)) {
                PyErr_SetString(PyExc_TypeError, "Dictionary values must be integers");
                return NULL;
            }
            int value_int = (int)PyLong_AsLong(value);

            // convert key from PyObject to int
            if (!PyLong_Check(key)) {
                PyErr_SetString(PyExc_TypeError, "Dictionary keys must be integers");
                return NULL;
            }
            int key_int = (int)PyLong_AsLong(key);

            self->decomp->set_iteration_loops(key_int, value_int);
        }
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Set the verbosity of the decomposition class
@param args Integer (0=False, non-zero=True)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Verbose(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    int verbose;
    if (!PyArg_ParseTuple(args, "i", &verbose)) {
        return NULL;
    }
    try {
        self->decomp->set_verbose(verbose);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Set the debugfile name of the decomposition class
@param args PyObject string for debug filename
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Debugfile(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    PyObject* debugfile = NULL;
    if (!PyArg_ParseTuple(args, "O", &debugfile)) {
        return NULL;
    }
    // determine the debugfile name type
    PyObject* debugfile_string = PyObject_Str(debugfile);
    PyObject* debugfile_string_unicode = PyUnicode_AsEncodedString(debugfile_string, "utf-8", "~E~");
    const char* debugfile_C = PyBytes_AS_STRING(debugfile_string_unicode);
    Py_XDECREF(debugfile_string);
    Py_XDECREF(debugfile_string_unicode);
    // determine the length of the filename and initialize C++ variant of the string
    Py_ssize_t string_length = PyBytes_Size(debugfile_string_unicode);
    std::string debugfile_Cpp(debugfile_C, string_length);
    try {
        // set the name of the debugfile on the C++ side
        self->decomp->set_debugfile(debugfile_Cpp);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Method to reorder the qubits in the decomposition class
@param args Tuple or list of integers representing the new qubit ordering
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_Reorder_Qubits(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    PyObject* qbit_list;
    if (!PyArg_ParseTuple(args, "O", &qbit_list)) {
        return NULL;
    }
    bool is_list = PyList_Check(qbit_list), is_tuple = PyTuple_Check(qbit_list);
    if (!is_list && !is_tuple) {
        PyErr_SetString(PyExc_TypeError, "Input must be tuple or list");
        return NULL;
    }
    Py_ssize_t element_num;
    if (is_tuple) {
        element_num = PyTuple_GET_SIZE(qbit_list);
    } else {
        element_num = PyList_GET_SIZE(qbit_list);
    }
    // create C++ variant of the tuple/list
    std::vector<int> qbit_list_C((int)element_num);
    for (Py_ssize_t idx = 0; idx < element_num; idx++) {
        if (is_tuple) {        
            qbit_list_C[(int) idx] = (int) PyLong_AsLong( PyTuple_GetItem(qbit_list, idx) );
        }
        else {
            qbit_list_C[(int) idx] = (int) PyLong_AsLong( PyList_GetItem(qbit_list, idx) );
        }
    }
    try {
        // reorder the qubits in the decomposition class
        self->decomp->reorder_qubits(qbit_list_C);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Wrapper method to set the optimization tolerance
@param args Double representing the tolerance value
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Optimization_Tolerance(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    double tolerance;
    if (!PyArg_ParseTuple(args, "d", &tolerance)) {
        return NULL;
    }
    try {
        self->decomp->set_optimization_tolerance(tolerance);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Wrapper method to set the threshold of convergence
@param args Double representing the threshold value
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Convergence_Threshold(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    double threshold;
    if (!PyArg_ParseTuple(args, "d", &threshold)) {
        return NULL;
    }
    try {
        self->decomp->set_convergence_threshold(threshold);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Wrapper method to set the number of gate blocks to be optimized
@param args Integer representing the number of optimization blocks
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Optimization_Blocks(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    int optimization_blocks;
    if (!PyArg_ParseTuple(args, "i", &optimization_blocks)) {
        return NULL;
    }
    try {
        self->decomp->set_optimization_blocks(optimization_blocks);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Wrapper function to add finalyzing layer (single qubit rotations on all qubits) to the gate structure.
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_add_Finalyzing_Layer_To_Gate_Structure(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    try {
        self->decomp->add_finalyzing_layer();
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    return Py_BuildValue("i", 0);
}

/**
@brief Wrapper function to set custom gate structure for the decomposition.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decompositions_Wrapper
@param args PyObject containing either a dictionary {int: Gates_block} (Decomposition only) or a single gate structure (all types)
@return Returns with zero on success.
@note applicable to: Decomposition (both map and single), Adaptive, Custom, Tree Search, Tabu Search (single only)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Gate_Structure(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    // initiate variables for input arguments
    PyObject* gate_structure_py; 
    
    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &gate_structure_py)) {
        return Py_BuildValue("i", -1);
    }
    
    // Check if input is a dictionary (map<int, Gates_block*> version: N_Qubit_Decomposition ONLY)
    if (PyDict_Check(gate_structure_py)) {
        PyObject *key = NULL, *value = NULL;
        Py_ssize_t pos = 0;
        std::map<int, Gates_block*> gate_structure;
        
        while (PyDict_Next(gate_structure_py, &pos, &key, &value)) {
            // convert key from PyObject to int
            if (!PyLong_Check(key)) {
                PyErr_SetString(PyExc_TypeError, "Dictionary keys must be integers");
                return NULL;
            }
            int key_int = (int)PyLong_AsLong(key);
            // convert value from PyObject to qgd_Circuit_Wrapper
            qgd_Circuit_Wrapper* qgd_op_block = (qgd_Circuit_Wrapper*)value;
            gate_structure.insert(std::pair<int, Gates_block*>(key_int, qgd_op_block->gate));
        }
        
        // The map version is only available in base N_Qubit_Decomposition class
        N_Qubit_Decomposition* base_decomp = dynamic_cast<N_Qubit_Decomposition*>(self->decomp);
        if (base_decomp != NULL) {
            try {
                base_decomp->set_custom_gate_structure(gate_structure);
                return Py_BuildValue("i", 0);
            } catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            } catch (std::exception& e) {
                PyErr_SetString(PyExc_Exception, e.what());
                return NULL;
            } catch (...) {
                std::string err("Invalid pointer to decomposition class");
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
        PyErr_SetString(PyExc_AttributeError, "Dictionary-based set_Gate_Structure is only available for N_Qubit_Decomposition");
        return NULL;
    }
    
    qgd_Circuit_Wrapper* qgd_op_block = (qgd_Circuit_Wrapper*)gate_structure_py;
    try {
        self->decomp->set_custom_gate_structure(qgd_op_block->gate);
        return Py_BuildValue("i", 0);
    } catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    } catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
}

/**
@brief  Get the number of free parameters in the gate structure used for the decomposition
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Parameter_Num(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    int parameter_num = self->decomp->get_parameter_num();
    return Py_BuildValue("i", parameter_num);
}

/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Optimized_Parameters(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    PyArrayObject* parameters_arr = NULL;
    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arr )) {
        return Py_BuildValue("i", -1);
    }
    
    if ( PyArray_IS_C_CONTIGUOUS(parameters_arr) ) {
        Py_INCREF(parameters_arr);
    } else {
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
@brief Get the number of free parameters in the gate structure used for the decomposition
@return The number of iterations
@note applicable to: Adaptive, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Num_of_Iters(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    int number_of_iters = self->decomp->get_num_iters();   
    return Py_BuildValue("i", number_of_iters);
}

/**
@brief Export unitary matrix to binary file
@param args Tuple containing filename string
@return 0 on success, -1 on error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_export_Unitary(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    // initiate variables for input arguments
    PyObject* filename = NULL;
    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &filename)) {
        return Py_BuildValue("i", -1);
    }
    PyObject* filename_string = PyObject_Str(filename);
    PyObject* filename_unicode = PyUnicode_AsEncodedString(filename_string, "utf-8", "~E~");
    const char* filename_C = PyBytes_AS_STRING(filename_unicode);
    std::string filename_str(filename_C);
    // export unitary to file
    self->decomp->export_unitary(filename_str);
    return Py_BuildValue("i", 0);
}

/**
@brief Call to get the project name
@return PyUnicode string with project name
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Project_Name(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    try {
        std::string project_name = self->decomp->get_project_name();
        return PyUnicode_FromString(project_name.c_str());
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to set the project name
@param args Tuple containing project name string
@return 0 on success, -1 on error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Project_Name(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    // initiate variables for input arguments
    PyObject* project_name_new = NULL;
    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &project_name_new)) {
        return Py_BuildValue("i", -1);
    }
    PyObject* project_name_new_string = PyObject_Str(project_name_new);
    PyObject* project_name_new_unicode = PyUnicode_AsEncodedString(project_name_new_string, "utf-8", "~E~");
    const char* project_name_new_C = PyBytes_AS_STRING(project_name_new_unicode);
    std::string project_name_new_str(project_name_new_C);
    // set the project name
    self->decomp->set_project_name(project_name_new_str);
    return Py_BuildValue("i", 0);
}

/**
@brief Call to get the global phase factor (returns the angle of the global phase)
@return PyFloat representing the angle of the global phase (the radius is always sqrt(2))
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Global_Phase(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    QGD_Complex16 global_phase_factor_C = self->decomp->get_global_phase_factor();
    PyObject* global_phase = PyFloat_FromDouble(std::atan2(global_phase_factor_C.imag, global_phase_factor_C.real));
    return global_phase;
}

/**
@brief Call to set the global phase
@param args Tuple containing phase angle (double)
@return Py_None on success, NULL on error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Global_Phase(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    double phase_angle;
    if (!PyArg_ParseTuple(args, "d", &phase_angle)) {
        return Py_BuildValue("i", -1);
    }
    try {
        self->decomp->set_global_phase(phase_angle);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Apply global phase factor to the unitary matrix
@return Py_None on success, NULL on error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_apply_Global_Phase_Factor(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    try {
        self->decomp->apply_global_phase_factor();
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to get the unitary matrix
@return Numpy array representing the unitary matrix
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Unitary(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    Matrix Unitary_mtx;
    try {
        Unitary_mtx = self->decomp->get_Umtx().copy();
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    // convert to numpy array
    Unitary_mtx.set_owner(false);
    PyObject *Unitary_py = matrix_to_numpy(Unitary_mtx);
    return Unitary_py;
}

/**
@brief Call to set the optimizer algorithm
@param args Positional arguments
@param kwds Keyword arguments (optimizer: string)
@return Py_BuildValue("i", 0) on success, NULL on error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Optimizer(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"optimizer", NULL};

    PyObject* optimizer_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &optimizer_arg)) {
        std::string err("Unsuccessful argument parsing");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    if (optimizer_arg == NULL) {
        std::string err("optimizer argument not set");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    PyObject* optimizer_string = PyObject_Str(optimizer_arg);
    PyObject* optimizer_string_unicode = PyUnicode_AsEncodedString(optimizer_string, "utf-8", "~E~");
    const char* optimizer_C = PyBytes_AS_STRING(optimizer_string_unicode);

    optimization_aglorithms qgd_optimizer;
    if (strcmp("bfgs", optimizer_C) == 0 || strcmp("BFGS", optimizer_C) == 0) {
        qgd_optimizer = BFGS;
    }
    else if (strcmp("adam", optimizer_C) == 0 || strcmp("ADAM", optimizer_C) == 0) {
        qgd_optimizer = ADAM;
    }
    else if (strcmp("grad_descend", optimizer_C) == 0 || strcmp("GRAD_DESCEND", optimizer_C) == 0) {
        qgd_optimizer = GRAD_DESCEND;
    }
    else if (strcmp("adam_batched", optimizer_C) == 0 || strcmp("ADAM_BATCHED", optimizer_C) == 0) {
        qgd_optimizer = ADAM_BATCHED;
    }
    else if (strcmp("bfgs2", optimizer_C) == 0 || strcmp("BFGS2", optimizer_C) == 0) {
        qgd_optimizer = BFGS2;
    }
    else if (strcmp("agents", optimizer_C) == 0 || strcmp("AGENTS", optimizer_C) == 0) {
        qgd_optimizer = AGENTS;
    }
    else if (strcmp("cosine", optimizer_C) == 0 || strcmp("COSINE", optimizer_C) == 0) {
        qgd_optimizer = COSINE;
    }
    else if (strcmp("grad_descend_phase_shift_rule", optimizer_C) == 0 || strcmp("GRAD_DESCEND_PARAMETER_SHIFT_RULE", optimizer_C) == 0) {
        qgd_optimizer = GRAD_DESCEND_PARAMETER_SHIFT_RULE;
    }
    else if (strcmp("agents_combined", optimizer_C) == 0 || strcmp("AGENTS_COMBINED", optimizer_C) == 0) {
        qgd_optimizer = AGENTS_COMBINED;
    }
    else if (strcmp("bayes_opt", optimizer_C) == 0 || strcmp("BAYES_OPT", optimizer_C) == 0) {
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
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    return Py_BuildValue("i", 0);
}

/**
@brief Set the number of maximum iterations for optimization
@param args Tuple containing max_iterations integer
@return Py_None on success, NULL on error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Max_Iterations(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    int max_iterations;
    if (!PyArg_ParseTuple(args, "i", &max_iterations)) {
        return Py_BuildValue("i", -1);
    }
    try {
        self->decomp->set_max_inner_iterations(max_iterations);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to get the matrix representation of the circuit with given parameters
@param args Tuple containing parameters array (numpy array of doubles)
@param kwds Optional keyword arguments (currently unused)
@return Numpy array representing the unitary matrix of the parameterized circuit
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Matrix(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args, PyObject *kwds)
{
    PyArrayObject* parameters_arr = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arr))
        return Py_BuildValue("i", -1);

    if (PyArray_IS_C_CONTIGUOUS(parameters_arr)) {
        Py_INCREF(parameters_arr);
    }
    else {
        parameters_arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }

    // get the C++ wrapper around the data
    Matrix_real&& parameters_mtx = numpy2matrix_real(parameters_arr);

    Matrix unitary_mtx;

    unitary_mtx = self->decomp->get_matrix(parameters_mtx);

    // convert to numpy array
    unitary_mtx.set_owner(false);
    PyObject *unitary_py = matrix_to_numpy(unitary_mtx);

    Py_DECREF(parameters_arr);

    return unitary_py;
}

/**
@brief Call to set the cost function variant
@param args Positional arguments
@param kwds Keyword arguments (costfnc: integer)
@return Py_BuildValue("i", 0) on success, NULL on error
@note applicable to: Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Cost_Function_Variant(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"costfnc", NULL};

    int costfnc_arg = 0;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &costfnc_arg)) {
        std::string err("Unsuccessful argument parsing");
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
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    return Py_BuildValue("i", 0);
}

/**
@brief Call to evaluate the optimization problem (cost function)
@param args Tuple containing parameters array (numpy array of doubles)
@return PyFloat with the cost function value
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_Optimization_Problem(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    PyArrayObject* parameters_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arg)) {
        std::string err("Unsuccessful argument parsing not ");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // establish memory contiguous arrays for C calculations
    if (PyArray_IS_C_CONTIGUOUS(parameters_arg) && PyArray_TYPE(parameters_arg) == NPY_FLOAT64) {
        Py_INCREF(parameters_arg);
    }
    else if (PyArray_TYPE(parameters_arg) == NPY_FLOAT64) {
        parameters_arg = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    }
    else {
        std::string err("Parameters should be should be real (given in float64 format)");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    Matrix_real parameters_mtx = numpy2matrix_real(parameters_arg);
    double f0;

    try {
        f0 = self->decomp->optimization_problem(parameters_mtx);
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    Py_DECREF(parameters_arg);

    return Py_BuildValue("d", f0);
}

/**
@brief Call to evaluate the optimization problem with unitary and derivatives
@param args Tuple containing parameters array (numpy array of doubles)
@return Tuple (unitary matrix, list of unitary derivatives)
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_Optimization_Problem_Combined_Unitary(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    PyArrayObject* parameters_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arg)) {
        std::string err("Unsuccessful argument parsing not ");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // establish memory contiguous arrays for C calculations
    if (PyArray_IS_C_CONTIGUOUS(parameters_arg) && PyArray_TYPE(parameters_arg) == NPY_FLOAT64) {
        Py_INCREF(parameters_arg);
    }
    else if (PyArray_TYPE(parameters_arg) == NPY_FLOAT64) {
        parameters_arg = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    }
    else {
        std::string err("Parameters should be should be real (given in float64 format)");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    Matrix_real parameters_mtx = numpy2matrix_real(parameters_arg);
    Matrix Umtx;
    std::vector<Matrix> Umtx_deriv;

    try {
        self->decomp->optimization_problem_combined_unitary(parameters_mtx, Umtx, Umtx_deriv);
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // convert to numpy array
    Umtx.set_owner(false);
    PyObject *unitary_py = matrix_to_numpy(Umtx);
    PyObject* graduni_py = PyList_New(Umtx_deriv.size());
    for (size_t i = 0; i < Umtx_deriv.size(); i++) {
        Umtx_deriv[i].set_owner(false);
        PyList_SetItem(graduni_py, i, matrix_to_numpy(Umtx_deriv[i]));
    }

    Py_DECREF(parameters_arg);

    PyObject* p = Py_BuildValue("(OO)", unitary_py, graduni_py);
    Py_DECREF(unitary_py);
    Py_DECREF(graduni_py);
    return p;
}

/**
@brief Call to evaluate the gradient of the optimization problem
@param args Tuple containing parameters array (numpy array of doubles)
@return Numpy array representing the gradient
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_Optimization_Problem_Grad(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    PyArrayObject* parameters_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arg)) {
        std::string err("Unsuccessful argument parsing not ");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // establish memory contiguous arrays for C calculations
    if (PyArray_IS_C_CONTIGUOUS(parameters_arg) && PyArray_TYPE(parameters_arg) == NPY_FLOAT64) {
        Py_INCREF(parameters_arg);
    }
    else if (PyArray_TYPE(parameters_arg) == NPY_FLOAT64) {
        parameters_arg = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    }
    else {
        std::string err("Parameters should be should be real (given in float64 format)");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    Matrix_real parameters_mtx = numpy2matrix_real(parameters_arg);
    Matrix_real grad_mtx(parameters_mtx.size(), 1);

    try {
        self->decomp->optimization_problem_grad(parameters_mtx, self->decomp, grad_mtx);
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // convert to numpy array
    grad_mtx.set_owner(false);
    PyObject *grad_py = matrix_real_to_numpy(grad_mtx);

    Py_DECREF(parameters_arg);

    return grad_py;
}

/**
@brief Call to evaluate the optimization problem with cost and gradient combined
@param args Tuple containing parameters array (numpy array of doubles)
@return Tuple (cost value, gradient array)
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_Optimization_Problem_Combined(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    PyArrayObject* parameters_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arg)) {
        std::string err("Unsuccessful argument parsing not ");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // establish memory contiguous arrays for C calculations
    if (PyArray_IS_C_CONTIGUOUS(parameters_arg) && PyArray_TYPE(parameters_arg) == NPY_FLOAT64) {
        Py_INCREF(parameters_arg);
    }
    else if (PyArray_TYPE(parameters_arg) == NPY_FLOAT64) {
        parameters_arg = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    }
    else {
        std::string err("Parameters should be should be real (given in float64 format)");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    Matrix_real parameters_mtx = numpy2matrix_real(parameters_arg);
    Matrix_real grad_mtx(parameters_mtx.size(), 1);
    double f0;

    try {
        self->decomp->optimization_problem_combined(parameters_mtx, &f0, grad_mtx);
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // convert to numpy array
    grad_mtx.set_owner(false);
    PyObject *grad_py = matrix_real_to_numpy(grad_mtx);

    Py_DECREF(parameters_arg);

    PyObject* p = Py_BuildValue("(dO)", f0, grad_py);
    Py_DECREF(grad_py);
    return p;
}

/**
@brief Call to evaluate the optimization problem for batched parameters
@param args Tuple containing parameters matrix (numpy array of doubles, each row is a parameter set)
@return Numpy array of cost function values
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_Optimization_Problem_Batch(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    PyArrayObject* parameters_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arg)) {
        std::string err("Unsuccessful argument parsing not ");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // establish memory contiguous arrays for C calculations
    if (PyArray_IS_C_CONTIGUOUS(parameters_arg) && PyArray_TYPE(parameters_arg) == NPY_FLOAT64) {
        Py_INCREF(parameters_arg);
    }
    else if (PyArray_TYPE(parameters_arg) == NPY_FLOAT64) {
        parameters_arg = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    }
    else {
        std::string err("Parameters should be should be real (given in float64 format)");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    Matrix_real parameters_mtx = numpy2matrix_real(parameters_arg);
    Matrix_real result_mtx;

    try {
        std::vector<Matrix_real> parameters_vec;
        parameters_vec.resize(parameters_mtx.rows);
        for (int row_idx = 0; row_idx < parameters_mtx.rows; row_idx++) {
            parameters_vec[row_idx] = Matrix_real(parameters_mtx.get_data() + row_idx * parameters_mtx.stride, 1, parameters_mtx.cols, parameters_mtx.stride);
        }
        result_mtx = self->decomp->optimization_problem_batched(parameters_vec);
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // convert to numpy array
    result_mtx.set_owner(false);
    PyObject *result_py = matrix_real_to_numpy(result_mtx);

    Py_DECREF(parameters_arg);

    return result_py;
}

/**
@brief Upload unitary matrix to DFE
@return Py_None on success, NULL on error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search (only when compiled with __DFE__)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_Upload_Umtx_to_DFE(qgd_N_Qubit_Decomposition_Wrapper *self)
{
#ifdef __DFE__
    try {
        self->decomp->upload_Umtx_to_DFE();
        Py_RETURN_NONE;
    } catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    } catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
#else
    PyErr_SetString(PyExc_NotImplementedError, "upload_Umtx_to_DFE is only available when compiled with DFE support");
    return NULL;
#endif
}

/**
@brief Get trace offset of the compression
@return Integer value of trace offset
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Trace_Offset(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    try {
        int trace_offset = self->decomp->get_trace_offset();
        return Py_BuildValue("i", trace_offset);
    } catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    } catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
}

/**
@brief Set trace offset for the compression
@param args Python tuple of arguments (trace_offset: int)
@return Py_None on success, NULL on error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Trace_Offset(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"trace_offset", NULL};
    
    int trace_offset = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &trace_offset)) {
        std::string err("Invalid arguments: expected (trace_offset: int)");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    try {
        self->decomp->set_trace_offset(trace_offset);
        Py_RETURN_NONE;
    } catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    } catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
}

/**
@brief Get the error of the decomposition
@return Double value of the decomposition error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Decomposition_Error(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    try {
        double error = self->decomp->get_decomposition_error();
        return Py_BuildValue("d", error);
    } catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    } catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
}

/**
@brief Get second Renyi entropy
@param args Python tuple of arguments (parameters_arr, input_state_arg, qubit_list_arg)
@return Double value of the second Renyi entropy
@note applicable to: Adaptive, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Second_Renyi_Entropy(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    PyArrayObject *parameters_arr = NULL, *input_state_arg = NULL;
    PyObject *qubit_list_arg = NULL;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "|OOO", &parameters_arr, &input_state_arg, &qubit_list_arg)) {
        return Py_BuildValue("i", -1);
    }

    // Ensure parameters array is C-contiguous
    if (PyArray_IS_C_CONTIGUOUS(parameters_arr)) {
        Py_INCREF(parameters_arr);
    } else {
        parameters_arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }

    // Get C++ wrapper around the parameters data
    Matrix_real&& parameters_mtx = numpy2matrix_real(parameters_arr);

    // Convert input state array
    if (input_state_arg == NULL) {
        PyErr_SetString(PyExc_Exception, "Input matrix was not given");
        return NULL;
    }

    PyArrayObject* input_state = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)input_state_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    // Test C-style contiguous memory allocation
    if (!PyArray_IS_C_CONTIGUOUS(input_state)) {
        PyErr_SetString(PyExc_Exception, "Input matrix is not memory contiguous");
        return NULL;
    }

    // Create QGD version of the input matrix
    Matrix input_state_mtx = numpy2matrix(input_state);

    // Check qubit list argument
    if (qubit_list_arg == NULL || !PyList_Check(qubit_list_arg)) {
        PyErr_SetString(PyExc_Exception, "qubit_list should be a list");
        return NULL;
    }

    Py_ssize_t reduced_qbit_num = PyList_Size(qubit_list_arg);
    matrix_base<int> qbit_list_mtx((int)reduced_qbit_num, 1);
    
    for (int idx = 0; idx < reduced_qbit_num; idx++) {
        PyObject* item = PyList_GET_ITEM(qubit_list_arg, idx);
        qbit_list_mtx[idx] = (int)PyLong_AsLong(item);
    }

    double entropy = -1;

    try {
        entropy = self->decomp->get_second_Renyi_entropy(parameters_mtx, input_state_mtx, qbit_list_mtx);
    } catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    } catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // Clean up references
    Py_DECREF(parameters_arr);
    Py_DECREF(input_state);

    PyObject* p = Py_BuildValue("d", entropy);
    return p;
}

/**
@brief Get the number of qubits
@return Integer value of the number of qubits
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Qbit_Num(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    try {
        int qbit_num = self->decomp->get_qbit_num();
        return Py_BuildValue("i", qbit_num);
    } catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    } catch (...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
}

////////////////////////////////////////////////////////////////// DECOMP SPECIFIC METHODs

/**
@brief Set the number of identical successive blocks (N_Qubit_Decomposition only)
@note applicable to: Decomposition
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Identical_Blocks(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    PyObject* identical_blocks_dict;
    if (!PyArg_ParseTuple(args, "O", &identical_blocks_dict)) {
        return NULL;
    }
    if (!PyDict_Check(identical_blocks_dict)) {
        PyErr_SetString(PyExc_TypeError, "Expected dictionary argument");
        return NULL;
    }
    try {
        N_Qubit_Decomposition* base_decomp = dynamic_cast<N_Qubit_Decomposition*>(self->decomp);
        if (base_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "set_identical_blocks is only available for N_Qubit_Decomposition");
            return NULL;
        }
        std::map<int, int> identical_blocks_map;
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(identical_blocks_dict, &pos, &key, &value)) {
            if (!PyLong_Check(key) || !PyLong_Check(value)) {
                PyErr_SetString(PyExc_TypeError, "Dictionary keys and values must be integers");
                return NULL;
            }
            int qubit_idx = PyLong_AsLong(key);
            int blocks = PyLong_AsLong(value);
            identical_blocks_map[qubit_idx] = blocks;
        }
        base_decomp->set_identical_blocks(identical_blocks_map);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to get initial circuit
@note applicable to: Adaptive
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_Initial_Circuit(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "get_initial_circuit is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }
        adaptive_decomp->get_initial_circuit();
        return Py_BuildValue("i", 0);
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to compress circuit
@note applicable to: Adaptive
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_Compress_Circuit(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "compress_circuit is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }
        adaptive_decomp->compress_circuit();
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to finalize circuit
@note applicable to: Adaptive
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_Finalize_Circuit(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args, PyObject *kwds)
{
    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "finalize_circuit is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }
        adaptive_decomp->finalize_circuit();
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Wrapper function to set custom layers to the gate structure that are intended to be used in the decomposition.
@note applicable to: Adaptive
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Gate_Structure_From_Binary(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    // initiate variables for input arguments
    PyObject* filename_py=NULL; 
    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &filename_py )) {
        return Py_BuildValue("i", -1);
    }
    // determine the optimizaton method
    PyObject* filename_string = PyObject_Str(filename_py);
    PyObject* filename_string_unicode = PyUnicode_AsEncodedString(filename_string, "utf-8", "~E~");
    const char* filename_C = PyBytes_AS_STRING(filename_string_unicode);
    std::string filename_str( filename_C );
    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            std::string err("set_Gate_Structure_From_Binary is only available for adaptive decomposition");
            PyErr_SetString(PyExc_Exception, err.c_str());
            return NULL;
        }
        adaptive_decomp->set_adaptive_gate_structure( filename_str );
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
@brief Wrapper function to append custom layers to the gate structure from binary file.
@note applicable to: Adaptive
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_add_Gate_Structure_From_Binary(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    // initiate variables for input arguments
    PyObject* filename_py = NULL;
    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &filename_py)) {
        return Py_BuildValue("i", -1);
    } 
    // Convert PyObject to UTF-8 encoded string
    PyObject* filename_string = PyObject_Str(filename_py);
    PyObject* filename_string_unicode = PyUnicode_AsEncodedString(filename_string, "utf-8", "~E~");
    const char* filename_C = PyBytes_AS_STRING(filename_string_unicode);
    std::string filename_str(filename_C);
    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "add_Gate_Structure_From_Binary is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }
        adaptive_decomp->add_adaptive_gate_structure(filename_str);
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    return Py_BuildValue("i", 0);
}

/**
@brief Wrapper function to set unitary from binary file.
@note applicable to: Adaptive
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Unitary_From_Binary(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    // initiate variables for input arguments
    PyObject* filename_py = NULL;
    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &filename_py)) {
        return Py_BuildValue("i", -1);
    } 
    // Convert PyObject to UTF-8 encoded string
    PyObject* filename_string = PyObject_Str(filename_py);
    PyObject* filename_string_unicode = PyUnicode_AsEncodedString(filename_string, "utf-8", "~E~");
    const char* filename_C = PyBytes_AS_STRING(filename_string_unicode);
    std::string filename_str(filename_C);
    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "set_Unitary_From_Binary is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }
        adaptive_decomp->set_unitary_from_file(filename_str);
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    return Py_BuildValue("i", 0);
}

/**
@brief Wrapper method to add adaptive layers to the gate structure stored by the class.
@note applicable to: Adaptive
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_add_Adaptive_Layers(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
    if (adaptive_decomp == NULL) {
        PyErr_SetString(PyExc_AttributeError, "add_Adaptive_Layers is only available for N_Qubit_Decomposition_adaptive");
        return NULL;
    }
    adaptive_decomp->add_adaptive_layers();
    return Py_BuildValue("i", 0);
}

/**
@brief Wrapper method to add layer to imported gate structure.
@note applicable to: Adaptive
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_add_Layer_To_Imported_Gate_Structure(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
    if (adaptive_decomp == NULL) {
        PyErr_SetString(PyExc_AttributeError, "add_Layer_To_Imported_Gate_Structure is only available for N_Qubit_Decomposition_adaptive");
        return NULL;
    }
    adaptive_decomp->add_layer_to_imported_gate_structure();
    return Py_BuildValue("i", 0);
}

/**
@brief Wrapper function to apply the imported gate structure on the unitary. The transformed unitary is to be decomposed in the calculations, and the imported gate structure is released.
@note applicable to: Adaptive
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_apply_Imported_Gate_Structure(qgd_N_Qubit_Decomposition_Wrapper *self)
{
    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "apply_Imported_Gate_Structure is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }
        adaptive_decomp->apply_imported_gate_structure();
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err("Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    return Py_BuildValue("i", 0);
}

// ========================================================================= METHODS SHARED ACROSS DECOMP CLASSES

/**
@brief Call to set unitary matrix
@param args Tuple containing numpy array (unitary matrix)
@return Py_BuildValue("i", 0) on success, NULL on error
@note applicable to: Adaptive, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Unitary(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args)
{
    if ( self->Umtx != NULL ) {
        // release the unitary to be decomposed
        Py_DECREF(self->Umtx);    
        self->Umtx = NULL;
    }

    PyArrayObject *Umtx_arg = NULL;
    //Parse arguments 
    if (!PyArg_ParseTuple(args, "|O", &Umtx_arg )) {
        return Py_BuildValue("i", -1);
    }
    
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
    
    // Try each decomposition type that supports set_unitary
    if (N_Qubit_Decomposition_adaptive* p = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp)) {
        p->set_unitary(Umtx_mtx);
        return Py_BuildValue("i", 0);
    }
    if (N_Qubit_Decomposition_Tree_Search* p = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp)) {
        p->set_unitary(Umtx_mtx);
        return Py_BuildValue("i", 0);
    }
    if (N_Qubit_Decomposition_Tabu_Search* p = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp)) {
        p->set_unitary(Umtx_mtx);
        return Py_BuildValue("i", 0);
    }

    PyErr_SetString(PyExc_TypeError, "set_unitary not available for this decomposition type");
    return NULL;
}


////////////////////////////////////////////////////////////////// EXTRA METHODS

/**
@brief Method to get gates as a list of dictionaries (with parameters from optimized_parameters array)
@return Returns with a Python list of gate dictionaries
*/
static PyObject*
qgd_N_Qubit_Decomposition_Wrapper_get_Gates(qgd_N_Qubit_Decomposition_Wrapper* self)
{
    std::vector<Gate*>&& gates = self->decomp->get_gates();
    Matrix_real&& params = self->decomp->get_optimized_parameters();

    PyObject* gates_list = PyList_New(0);
    if (!gates_list) return NULL;

    for (size_t idx = 0; idx < gates.size(); idx++) {
        Gate* gate = gates[idx];
        if (!gate) continue;

        PyObject* gate_dict = PyDict_New();
        if (!gate_dict) {
            Py_DECREF(gates_list);
            return NULL;
        }

        // Map gate type to string
        const char* type_str = nullptr;
        switch(gate->get_type()) {
            case CNOT_OPERATION: type_str = "CNOT"; break;
            case CZ_OPERATION: type_str = "CZ"; break;
            case CH_OPERATION: type_str = "CH"; break;
            case SYC_OPERATION: type_str = "SYC"; break;
            case U3_OPERATION: type_str = "U3"; break;
            case RX_OPERATION: type_str = "RX"; break;
            case RY_OPERATION: type_str = "RY"; break;
            case RZ_OPERATION: type_str = "RZ"; break;
            case X_OPERATION: type_str = "X"; break;
            case Y_OPERATION: type_str = "Y"; break;
            case Z_OPERATION: type_str = "Z"; break;
            case SX_OPERATION: type_str = "SX"; break;
            case CRY_OPERATION: type_str = "CRY"; break;
            default: type_str = "UNKNOWN"; break;
        }
        PyDict_SetItemString(gate_dict, "type", PyUnicode_FromString(type_str));
        
        PyDict_SetItemString(gate_dict, "target_qbit", PyLong_FromLong(gate->get_target_qbit()));

        int control_qbit = gate->get_control_qbit();
        if (control_qbit >= 0) {
            PyDict_SetItemString(gate_dict, "control_qbit", PyLong_FromLong(control_qbit));
        }
        
        // Add parameters
        int pnum = gate->get_parameter_num();
        int pstart = gate->get_parameter_start_idx();
        if (pnum > 0 && pstart >= 0 && (pstart + pnum) <= (int)params.size()) {
            if (gate->get_type() == U3_OPERATION && pnum >= 3) {
                PyDict_SetItemString(gate_dict, "Theta", PyFloat_FromDouble(params[pstart]));
                PyDict_SetItemString(gate_dict, "Phi", PyFloat_FromDouble(params[pstart + 1]));
                PyDict_SetItemString(gate_dict, "Lambda", PyFloat_FromDouble(params[pstart + 2]));
            } else if (gate->get_type() == RX_OPERATION || gate->get_type() == RY_OPERATION || gate->get_type() == CRY_OPERATION) {
                PyDict_SetItemString(gate_dict, "Theta", PyFloat_FromDouble(params[pstart]));
            } else if (gate->get_type() == RZ_OPERATION) {
                PyDict_SetItemString(gate_dict, "Phi", PyFloat_FromDouble(params[pstart]));
            }
        }

        PyList_Append(gates_list, gate_dict);
        Py_DECREF(gate_dict);
    }
    return gates_list;
}

/**
@brief Method to get Qiskit circuit representation
@return Returns with a Qiskit QuantumCircuit object
*/
static PyObject*
qgd_N_Qubit_Decomposition_Wrapper_get_Qiskit_Circuit(qgd_N_Qubit_Decomposition_Wrapper* self)
{
    // Import Qiskit_IO module
    PyObject* qiskit_io_module = PyImport_ImportModule("squander.IO_interfaces.Qiskit_IO");
    if (!qiskit_io_module) {
        PyErr_SetString(PyExc_ImportError, "Failed to import squander.IO_interfaces.Qiskit_IO");
        return NULL;
    }
    
    // Get the get_Qiskit_Circuit function
    PyObject* get_qiskit_func = PyObject_GetAttrString(qiskit_io_module, "get_Qiskit_Circuit");
    Py_DECREF(qiskit_io_module);
    if (!get_qiskit_func) {
        PyErr_SetString(PyExc_AttributeError, "get_Qiskit_Circuit not found in Qiskit_IO");
        return NULL;
    }

    // Get circuit and parameters
    PyObject* circuit = qgd_N_Qubit_Decomposition_Wrapper_get_Circuit(self);
    if (!circuit) {
        Py_DECREF(get_qiskit_func);
        return NULL;
    }
    PyObject* parameters = qgd_N_Qubit_Decomposition_Wrapper_get_Optimized_Parameters(self);
    if (!parameters) {
        Py_DECREF(get_qiskit_func);
        Py_DECREF(circuit);
        return NULL;
    }

    // Call get_Qiskit_Circuit(circuit, parameters)
    PyObject* args = PyTuple_Pack(2, circuit, parameters);
    PyObject* result = PyObject_CallObject(get_qiskit_func, args);
    
    Py_DECREF(args);
    Py_DECREF(parameters);
    Py_DECREF(circuit);
    Py_DECREF(get_qiskit_func);

    return result;
}

/**
@brief Method to get Cirq circuit representation
@return Returns with a Cirq Circuit object
*/

#define CIRQ_ADD_SINGLE_QUBIT_GATE(name) do { \
    PyObject* gate_func = PyObject_GetAttrString(cirq_module, #name); \
    PyObject* gate_args = PyTuple_Pack(1, target_qubit); \
    PyObject* cirq_gate = PyObject_CallObject(gate_func, gate_args); \
    Py_DECREF(gate_args); Py_DECREF(gate_func); \
    if (cirq_gate) { \
        PyObject* append_args = PyTuple_Pack(1, cirq_gate); \
        PyObject_CallObject(append_func, append_args); \
        Py_DECREF(append_args); Py_DECREF(cirq_gate); \
    } \
} while(0)

// Helper macros for Cirq gate creation
#define CIRQ_ADD_TWO_QUBIT_GATE(name) do { \
    PyObject* control_qbit_obj = PyDict_GetItemString(gate, "control_qbit"); \
    if (!control_qbit_obj) continue; \
    long control_idx = qbit_num - 1 - PyLong_AsLong(control_qbit_obj); \
    PyObject* control_qubit = PyList_GetItem(qubits, control_idx); \
    PyObject* gate_func = PyObject_GetAttrString(cirq_module, #name); \
    PyObject* gate_args = PyTuple_Pack(2, control_qubit, target_qubit); \
    PyObject* cirq_gate = PyObject_CallObject(gate_func, gate_args); \
    Py_DECREF(gate_args); Py_DECREF(gate_func); \
    if (cirq_gate) { \
        PyObject* append_args = PyTuple_Pack(1, cirq_gate); \
        PyObject_CallObject(append_func, append_args); \
        Py_DECREF(append_args); Py_DECREF(cirq_gate); \
    } \
} while(0)

#define CIRQ_ADD_ROTATION_GATE(name, param) do { \
    PyObject* param_obj = PyDict_GetItemString(gate, param); \
    if (!param_obj) continue; \
    PyObject* gate_func = PyObject_GetAttrString(cirq_module, #name); \
    PyObject* gate_args = PyTuple_Pack(1, param_obj); \
    PyObject* cirq_gate = PyObject_CallObject(gate_func, gate_args); \
    Py_DECREF(gate_args); Py_DECREF(gate_func); \
    if (cirq_gate) { \
        PyObject* on_method = PyObject_GetAttrString(cirq_gate, "on"); \
        PyObject* on_args = PyTuple_Pack(1, target_qubit); \
        PyObject* gate_op = PyObject_CallObject(on_method, on_args); \
        Py_DECREF(on_args); Py_DECREF(on_method); Py_DECREF(cirq_gate); \
        if (gate_op) { \
            PyObject* append_args = PyTuple_Pack(1, gate_op); \
            PyObject_CallObject(append_func, append_args); \
            Py_DECREF(append_args); Py_DECREF(gate_op); \
        } \
    } \
} while(0)

static PyObject*
qgd_N_Qubit_Decomposition_Wrapper_get_Cirq_Circuit(qgd_N_Qubit_Decomposition_Wrapper* self)
{
    PyObject* cirq_module = PyImport_ImportModule("cirq");
    if (!cirq_module) {
        PyErr_SetString(PyExc_ImportError, "Failed to import cirq. Please install cirq package.");
        return NULL;
    }

    PyObject* cirq_circuit_class = PyObject_GetAttrString(cirq_module, "Circuit");
    if (!cirq_circuit_class) { 
        Py_DECREF(cirq_module); 
        return NULL; 
    }
    
    PyObject* cirq_circuit_obj = PyObject_CallObject(cirq_circuit_class, NULL);
    Py_DECREF(cirq_circuit_class);
    if (!cirq_circuit_obj) { 
        Py_DECREF(cirq_module); 
        return NULL; 
    }

    // Create qubit register
    PyObject* cirq_line_qubit_class = PyObject_GetAttrString(cirq_module, "LineQubit");
    if (!cirq_line_qubit_class) { 
        Py_DECREF(cirq_circuit_obj); 
        Py_DECREF(cirq_module); 
        return NULL; 
    }
    PyObject* range_func = PyObject_GetAttrString(cirq_line_qubit_class, "range");
    Py_DECREF(cirq_line_qubit_class);
    if (!range_func) { 
        Py_DECREF(cirq_circuit_obj); 
        Py_DECREF(cirq_module); 
        return NULL; 
    }

    int qbit_num = self->decomp->get_qbit_num();
    PyObject* range_args = PyTuple_Pack(1, PyLong_FromLong(qbit_num));
    PyObject* qubits = PyObject_CallObject(range_func, range_args);
    Py_DECREF(range_args); Py_DECREF(range_func);
    if (!qubits) { 
        Py_DECREF(cirq_circuit_obj); 
        Py_DECREF(cirq_module); 
        return NULL; 
    }

    PyObject* gates_list = qgd_N_Qubit_Decomposition_Wrapper_get_Gates(self);
    if (!gates_list) { 
        Py_DECREF(qubits); 
        Py_DECREF(cirq_circuit_obj); 
        Py_DECREF(cirq_module); 
        return NULL; 
    }

    PyObject* append_func = PyObject_GetAttrString(cirq_circuit_obj, "append");
    if (!append_func) {
        Py_DECREF(gates_list); Py_DECREF(qubits); Py_DECREF(cirq_circuit_obj); Py_DECREF(cirq_module);
        return NULL;
    }

    PyObject* cirq_google_module = PyObject_GetAttrString(cirq_module, "google");

    // Process gates in reverse order
    Py_ssize_t num_gates = PyList_Size(gates_list);
    for (Py_ssize_t idx = num_gates - 1; idx >= 0; idx--) {
        PyObject* gate = PyList_GetItem(gates_list, idx);
        if (!gate) continue;

        PyObject* gate_type = PyDict_GetItemString(gate, "type");
        if (!gate_type) continue;
        const char* gate_type_str = PyUnicode_AsUTF8(gate_type);
        if (!gate_type_str) continue;

        PyObject* target_qbit_obj = PyDict_GetItemString(gate, "target_qbit");
        if (!target_qbit_obj) continue;

        long target_idx = qbit_num - 1 - PyLong_AsLong(target_qbit_obj);
        PyObject* target_qubit = PyList_GetItem(qubits, target_idx);
        if (!target_qubit) continue;

        if (strcmp(gate_type_str, "CNOT") == 0) { CIRQ_ADD_TWO_QUBIT_GATE(CNOT); }
        else if (strcmp(gate_type_str, "CZ") == 0) { CIRQ_ADD_TWO_QUBIT_GATE(CZ); }
        else if (strcmp(gate_type_str, "CH") == 0) { CIRQ_ADD_TWO_QUBIT_GATE(CH); }
        else if (strcmp(gate_type_str, "SYC") == 0 && cirq_google_module) {
            PyObject* control_qbit_obj = PyDict_GetItemString(gate, "control_qbit");
            if (control_qbit_obj) {
                long control_idx = qbit_num - 1 - PyLong_AsLong(control_qbit_obj);
                PyObject* control_qubit = PyList_GetItem(qubits, control_idx);
                
                PyObject* syc_func = PyObject_GetAttrString(cirq_google_module, "SYC");
                PyObject* syc_args = PyTuple_Pack(2, control_qubit, target_qubit);
                PyObject* cirq_gate = PyObject_CallObject(syc_func, syc_args);
                Py_DECREF(syc_args); 
                Py_DECREF(syc_func);
                if (cirq_gate) {
                    PyObject* append_args = PyTuple_Pack(1, cirq_gate);
                    PyObject_CallObject(append_func, append_args);
                    Py_DECREF(append_args); 
                    Py_DECREF(cirq_gate);
                }
            }
        }
        else if (strcmp(gate_type_str, "CRY") == 0) { 
            printf("CRY gate needs to be implemented\n"); 
        }
        else if (strcmp(gate_type_str, "U3") == 0) {
            printf("Unsupported gate in the Cirq export: U3 gate\n");
            Py_XDECREF(cirq_google_module); 
            Py_DECREF(append_func); 
            Py_DECREF(gates_list);
            Py_DECREF(qubits); 
            Py_DECREF(cirq_circuit_obj); 
            Py_DECREF(cirq_module);
            Py_RETURN_NONE;
        }
        else if (strcmp(gate_type_str, "RX") == 0) { CIRQ_ADD_ROTATION_GATE(rx, "Theta"); }
        else if (strcmp(gate_type_str, "RY") == 0) { CIRQ_ADD_ROTATION_GATE(ry, "Theta"); }
        else if (strcmp(gate_type_str, "RZ") == 0) { CIRQ_ADD_ROTATION_GATE(rz, "Phi"); }
        else if (strcmp(gate_type_str, "X") == 0) { CIRQ_ADD_SINGLE_QUBIT_GATE(x); }
        else if (strcmp(gate_type_str, "Y") == 0) { CIRQ_ADD_SINGLE_QUBIT_GATE(y); }
        else if (strcmp(gate_type_str, "Z") == 0) { CIRQ_ADD_SINGLE_QUBIT_GATE(z); }
        else if (strcmp(gate_type_str, "SX") == 0) { CIRQ_ADD_SINGLE_QUBIT_GATE(sx); }
    }

    Py_XDECREF(cirq_google_module);
    Py_DECREF(append_func);
    Py_DECREF(gates_list);
    Py_DECREF(qubits);
    Py_DECREF(cirq_module);

    return cirq_circuit_obj;
}

#undef CIRQ_ADD_SINGLE_QUBIT_GATE
#undef CIRQ_ADD_TWO_QUBIT_GATE
#undef CIRQ_ADD_ROTATION_GATE

/**
@brief Method to import Qiskit circuit (standard version for non-adaptive decompositions)
@param qc_in Qiskit QuantumCircuit to import
@return Returns Py_None on success
*/
static PyObject*
qgd_N_Qubit_Decomposition_Wrapper_import_Qiskit_Circuit_standard(qgd_N_Qubit_Decomposition_Wrapper* self, PyObject* qc_in)
{
    // Import Qiskit_IO module
    PyObject* qiskit_io_module = PyImport_ImportModule("squander.IO_interfaces.Qiskit_IO");
    if (!qiskit_io_module) {
        PyErr_SetString(PyExc_ImportError, "Failed to import squander.IO_interfaces.Qiskit_IO");
        return NULL;
    }
    // Get the convert_Qiskit_to_Squander function
    PyObject* convert_func = PyObject_GetAttrString(qiskit_io_module, "convert_Qiskit_to_Squander");
    Py_DECREF(qiskit_io_module);
    if (!convert_func) {
        PyErr_SetString(PyExc_AttributeError, "convert_Qiskit_to_Squander not found in Qiskit_IO");
        return NULL;
    }
    // Call convert_Qiskit_to_Squander(qc_in) -> returns (circuit, parameters)
    PyObject* convert_args = PyTuple_Pack(1, qc_in);
    PyObject* convert_result = PyObject_CallObject(convert_func, convert_args);
    Py_DECREF(convert_args);
    Py_DECREF(convert_func);
    if (!convert_result || !PyTuple_Check(convert_result) || PyTuple_Size(convert_result) != 2) {
        Py_XDECREF(convert_result);
        PyErr_SetString(PyExc_ValueError, "convert_Qiskit_to_Squander should return (circuit, parameters)");
        return NULL;
    }

    PyObject *circuit_squander = PyTuple_GetItem(convert_result, 0), *parameters = PyTuple_GetItem(convert_result, 1);

    // Set gate structure
    PyObject* set_gate_args = PyTuple_Pack(1, circuit_squander);
    PyObject* set_gate_result = qgd_N_Qubit_Decomposition_Wrapper_set_Gate_Structure(self, set_gate_args);
    Py_DECREF(set_gate_args);
    if (!set_gate_result) {
        Py_DECREF(convert_result);
        return NULL;
    }
    Py_DECREF(set_gate_result);

    // Set optimized parameters
    PyObject* set_params_args = PyTuple_Pack(1, parameters);
    PyObject* set_params_result = qgd_N_Qubit_Decomposition_Wrapper_set_Optimized_Parameters(self, set_params_args);
    Py_DECREF(set_params_args);
    Py_DECREF(convert_result);
    if (!set_params_result) {
        return NULL;
    }
    Py_DECREF(set_params_result);

    Py_RETURN_NONE;
}

/**
@brief Method to import Qiskit circuit (adaptive-specific version with custom CZ decomposition)
@param qc_in Qiskit QuantumCircuit to import
@return Returns Py_None on success
*/
static PyObject*
qgd_N_Qubit_Decomposition_Wrapper_import_Qiskit_Circuit_adaptive(qgd_N_Qubit_Decomposition_Wrapper* self, PyObject* qc_in)
{
    // Import qiskit module
    PyObject* qiskit_module = PyImport_ImportModule("qiskit");
    if (!qiskit_module) {
        PyErr_SetString(PyExc_ImportError, "Failed to import qiskit");
        return NULL;
    }
    // Get transpile function
    PyObject* transpile_func = PyObject_GetAttrString(qiskit_module, "transpile");
    Py_DECREF(qiskit_module);
    if (!transpile_func) {
        PyErr_SetString(PyExc_AttributeError, "transpile not found in qiskit");
        return NULL;
    }

    // Transpile: transpile(qc_in, optimization_level=0, basis_gates=['cz', 'u3'], layout_method='sabre')
    PyObject* basis_gates = PyList_New(2);
    PyList_SetItem(basis_gates, 0, PyUnicode_FromString("cz"));
    PyList_SetItem(basis_gates, 1, PyUnicode_FromString("u3"));
    
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "optimization_level", PyLong_FromLong(0));
    PyDict_SetItemString(kwargs, "basis_gates", basis_gates);
    PyDict_SetItemString(kwargs, "layout_method", PyUnicode_FromString("sabre"));
    
    PyObject* transpile_args = PyTuple_Pack(1, qc_in);
    PyObject* qc = PyObject_Call(transpile_func, transpile_args, kwargs);
    
    Py_DECREF(transpile_args);
    Py_DECREF(kwargs);
    Py_DECREF(basis_gates);
    Py_DECREF(transpile_func);
    if (!qc) {
        return NULL;
    }

    // Print gate counts
    PyObject* count_ops_func = PyObject_GetAttrString(qc, "count_ops");
    if (count_ops_func) {
        PyObject* count_ops_result = PyObject_CallObject(count_ops_func, NULL);
        Py_DECREF(count_ops_func);
        if (count_ops_result) {
            printf("Gate counts in the imported Qiskit transpiled quantum circuit: ");
            PyObject_Print(count_ops_result, stdout, 0);
            printf("\n");
            Py_DECREF(count_ops_result);
        }
    }

    // Get circuit data
    PyObject* qc_data_attr = PyObject_GetAttrString(qc, "data");
    PyObject* qc_qubits_attr = PyObject_GetAttrString(qc, "qubits");
    PyObject* qc_num_qubits_attr = PyObject_GetAttrString(qc, "num_qubits");
    if (!qc_data_attr || !qc_qubits_attr || !qc_num_qubits_attr) {
        Py_XDECREF(qc_data_attr);
        Py_XDECREF(qc_qubits_attr);
        Py_XDECREF(qc_num_qubits_attr);
        Py_DECREF(qc);
        return NULL;
    }

    int register_size = PyLong_AsLong(qc_num_qubits_attr);
    Py_DECREF(qc_num_qubits_attr);

    // Import Circuit_Wrapper
    PyObject* circuit_wrapper_module = PyImport_ImportModule("squander.gates.qgd_Circuit_Wrapper");
    if (!circuit_wrapper_module) {
        Py_DECREF(qc_data_attr);
        Py_DECREF(qc_qubits_attr);
        Py_DECREF(qc);
        return NULL;
    }

    PyObject* circuit_wrapper_class = PyObject_GetAttrString(circuit_wrapper_module, "qgd_Circuit_Wrapper");
    Py_DECREF(circuit_wrapper_module);
    if (!circuit_wrapper_class) {
        Py_DECREF(qc_data_attr);
        Py_DECREF(qc_qubits_attr);
        Py_DECREF(qc);
        return NULL;
    }

    // Create main circuit: Circuit_ret = qgd_Circuit_Wrapper(register_size)
    PyObject* circuit_ret_args = PyTuple_Pack(1, PyLong_FromLong(register_size));
    PyObject* Circuit_ret_result = PyObject_CallObject(circuit_wrapper_class, circuit_ret_args);
    Py_DECREF(circuit_ret_args);
    Py_DECREF(circuit_wrapper_class);
    if (!Circuit_ret_result) {
        Py_DECREF(qc_data_attr);
        Py_DECREF(qc_qubits_attr);
        Py_DECREF(qc);
        return NULL;
    }

    // Create dictionary for single qubit gates: single_qubit_gates[qubit] = []
    PyObject* single_qubit_gates = PyDict_New();
    for (int idx = 0; idx < register_size; idx++) {
        PyObject* key = PyLong_FromLong(idx);
        PyObject* value = PyList_New(0);
        PyDict_SetItem(single_qubit_gates, key, value);
        Py_DECREF(key);
        Py_DECREF(value);
    }

    PyObject* optimized_parameters = PyList_New(0);

    // Process gates from qc.data
    Py_ssize_t qc_data_attr_size = PyList_Size(qc_data_attr);
    for (Py_ssize_t i = 0; i < qc_data_attr_size; i++) {
        PyObject* gate = PyList_GetItem(qc_data_attr, i);
        PyObject* gate_operation = PyObject_GetAttrString(gate, "operation");
        PyObject* gate_qubits = PyObject_GetAttrString(gate, "qubits");
        if (!gate_operation || !gate_qubits) {
            Py_XDECREF(gate_operation);
            Py_XDECREF(gate_qubits);
            continue;
        }

        PyObject* gate_operation_name_attr = PyObject_GetAttrString(gate_operation, "name");
        const char* name = PyUnicode_AsUTF8(gate_operation_name_attr);
        
        if (strcmp(name, "u3") == 0) {
            // Get qubit index
            PyObject* index_func = PyObject_GetAttrString(qc_qubits_attr, "index");

            PyObject* index_args = PyTuple_Pack(1, PyList_GetItem(gate_qubits, 0));
            PyObject* index_result = PyObject_CallObject(index_func, index_args);
            Py_DECREF(index_func);
            Py_DECREF(index_args);
            
            long qubit = PyLong_AsLong(index_result);
            Py_DECREF(index_result);
            
            // Store u3 gate info
            PyObject* gate_info_dict = PyDict_New();
            PyObject* gate_operation_params_attr = PyObject_GetAttrString(gate_operation, "params");
            PyDict_SetItemString(gate_info_dict, "params", gate_operation_params_attr);
            PyDict_SetItemString(gate_info_dict, "type", PyUnicode_FromString("u3"));
            Py_DECREF(gate_operation_params_attr);
            
            PyObject* qubit_list = PyDict_GetItem(single_qubit_gates, PyLong_FromLong(qubit));
            PyList_Append(qubit_list, gate_info_dict);
            Py_DECREF(gate_info_dict);
        } else if (strcmp(name, "cz") == 0) {
            // Get qubit indices
            PyObject* index_func = PyObject_GetAttrString(qc_qubits_attr, "index");

            PyObject* index_args0 = PyTuple_Pack(1, PyList_GetItem(gate_qubits, 0));
            PyObject* index_args0_result = PyObject_CallObject(index_func, index_args0);
            Py_DECREF(index_args0);
            
            PyObject* index_args1 = PyTuple_Pack(1, PyList_GetItem(gate_qubits, 1));
            PyObject* index_args1_result = PyObject_CallObject(index_func, index_args1);
            Py_DECREF(index_args1);
            Py_DECREF(index_func);
            
            long qubit0 = PyLong_AsLong(index_args0_result);
            long qubit1 = PyLong_AsLong(index_args1_result);
            Py_DECREF(index_args0_result);
            Py_DECREF(index_args1_result);
            
            // Create layer
            PyObject* layer_args = PyTuple_Pack(1, PyLong_FromLong(register_size));
            PyObject* circuit_wrapper_module2 = PyImport_ImportModule("squander.gates.qgd_Circuit_Wrapper");
            PyObject* circuit_wrapper_class2 = PyObject_GetAttrString(circuit_wrapper_module2, "qgd_Circuit_Wrapper");
            Py_DECREF(circuit_wrapper_module2);
            
            PyObject* Layer = PyObject_CallObject(circuit_wrapper_class2, layer_args);
            Py_DECREF(layer_args);
            Py_DECREF(circuit_wrapper_class2);
            
            // Add u3 gates for qubit0
            PyObject* qubit0_list = PyDict_GetItem(single_qubit_gates, PyLong_FromLong(qubit0));
            if (qubit0_list && PyList_Size(qubit0_list) > 0) {
                PyObject* gate0 = PyList_GetItem(qubit0_list, 0);
                PyList_SetSlice(qubit0_list, 0, 1, NULL); // pop first element
                
                PyObject* add_u3_func = PyObject_GetAttrString(Layer, "add_U3");
                PyObject* add_u3_args = Py_BuildValue("(iOOO)", qubit0, Py_True, Py_True, Py_True);
                PyObject_CallObject(add_u3_func, add_u3_args);
                Py_DECREF(add_u3_func);
                Py_DECREF(add_u3_args);
                
                // Add parameters (reversed)
                PyObject* params = PyDict_GetItemString(gate0, "params");
                PyObject* reversed_params = PyList_New(0);
                for (Py_ssize_t j = PyList_Size(params) - 1; j >= 0; j--) {
                    PyList_Append(reversed_params, PyList_GetItem(params, j));
                }
                for (Py_ssize_t j = 0; j < PyList_Size(reversed_params); j++) {
                    PyList_Append(optimized_parameters, PyList_GetItem(reversed_params, j));
                }
                Py_DECREF(reversed_params);
                
                // Divide last parameter by 2
                Py_ssize_t last_idx = PyList_Size(optimized_parameters) - 1;
                PyObject* last_param = PyList_GetItem(optimized_parameters, last_idx);
                double val = PyFloat_AsDouble(last_param) / 2.0;
                PyList_SetItem(optimized_parameters, last_idx, PyFloat_FromDouble(val));
            }
            
            // Add u3 gates for qubit1
            PyObject* qubit1_list = PyDict_GetItem(single_qubit_gates, PyLong_FromLong(qubit1));
            if (qubit1_list && PyList_Size(qubit1_list) > 0) {
                PyObject* gate1 = PyList_GetItem(qubit1_list, 0);
                PyList_SetSlice(qubit1_list, 0, 1, NULL);
                
                PyObject* add_u3_func = PyObject_GetAttrString(Layer, "add_U3");
                PyObject* u3_args = Py_BuildValue("(iOOO)", qubit1, Py_True, Py_True, Py_True);
                PyObject_CallObject(add_u3_func, u3_args);
                Py_DECREF(add_u3_func);
                Py_DECREF(u3_args);
                
                PyObject* params = PyDict_GetItemString(gate1, "params");
                PyObject* reversed_params = PyList_New(0);
                for (Py_ssize_t j = PyList_Size(params) - 1; j >= 0; j--) {
                    PyList_Append(reversed_params, PyList_GetItem(params, j));
                }
                for (Py_ssize_t j = 0; j < PyList_Size(reversed_params); j++) {
                    PyList_Append(optimized_parameters, PyList_GetItem(reversed_params, j));
                }
                Py_DECREF(reversed_params);
                
                Py_ssize_t last_idx = PyList_Size(optimized_parameters) - 1;
                PyObject* last_param = PyList_GetItem(optimized_parameters, last_idx);
                double val = PyFloat_AsDouble(last_param) / 2.0;
                PyList_SetItem(optimized_parameters, last_idx, PyFloat_FromDouble(val));
            }
            
            // Add RX, adaptive, RZ, RX sequence
            PyObject* qubit0_obj = PyLong_FromLong(qubit0);
            PyObject* qubit1_obj = PyLong_FromLong(qubit1);
            
            PyObject* add_rx_func = PyObject_GetAttrString(Layer, "add_RX");
            PyObject* add_rx_arg = PyTuple_Pack(1, qubit0_obj);
            PyObject_CallObject(add_rx_func, add_rx_arg);
            Py_DECREF(add_rx_func);
            Py_DECREF(add_rx_arg);
            
            PyObject* add_adaptive_func = PyObject_GetAttrString(Layer, "add_adaptive");
            PyObject* add_adaptive_args = PyTuple_Pack(2, qubit0_obj, qubit1_obj);
            PyObject_CallObject(add_adaptive_func, add_adaptive_args);
            Py_DECREF(add_adaptive_func);
            Py_DECREF(add_adaptive_args);
            
            PyObject* add_rz_func = PyObject_GetAttrString(Layer, "add_RZ");
            PyObject* add_rz_arg = PyTuple_Pack(1, qubit1_obj);
            PyObject_CallObject(add_rz_func, add_rz_arg);
            Py_DECREF(add_rz_func);
            Py_DECREF(add_rz_arg);
            
            PyObject* add_rx_func_2 = PyObject_GetAttrString(Layer, "add_RX");
            PyObject* add_rx_arg_2 = PyTuple_Pack(1, qubit0_obj);
            PyObject_CallObject(add_rx_func_2, add_rx_arg_2);
            Py_DECREF(add_rx_func_2);
            Py_DECREF(add_rx_arg_2);
            
            Py_DECREF(qubit0_obj);
            Py_DECREF(qubit1_obj);
            
            // Add hardcoded parameters
            PyList_Append(optimized_parameters, PyFloat_FromDouble(M_PI / 4.0));
            PyList_Append(optimized_parameters, PyFloat_FromDouble(M_PI / 2.0));
            PyList_Append(optimized_parameters, PyFloat_FromDouble(-M_PI / 2.0));
            PyList_Append(optimized_parameters, PyFloat_FromDouble(-M_PI / 4.0));
            
            // Add layer to circuit
            PyObject* add_circuit_func = PyObject_GetAttrString(Circuit_ret_result, "add_Circuit");
            PyObject* add_circuit_args = PyTuple_Pack(1, Layer);
            PyObject_CallObject(add_circuit_func, add_circuit_args);
            Py_DECREF(add_circuit_func);
            Py_DECREF(add_circuit_args);
            Py_DECREF(Layer);
        }
        Py_DECREF(gate_operation_name_attr);
        Py_DECREF(gate_operation);
        Py_DECREF(gate_qubits);
    }

    // Add remaining single qubit gates
    PyObject* circuit_module = PyImport_ImportModule("squander.gates.qgd_Circuit");
    PyObject* circuit_class = PyObject_GetAttrString(circuit_module, "qgd_Circuit");
    Py_DECREF(circuit_module);
    
    PyObject* final_layer_args = PyTuple_Pack(1, PyLong_FromLong(register_size));
    PyObject* final_layer_result = PyObject_CallObject(circuit_class, final_layer_args);
    Py_DECREF(circuit_class);
    Py_DECREF(final_layer_args);
    
    for (int qubit = 0; qubit < register_size; qubit++) {
        PyObject* gates_list = PyDict_GetItem(single_qubit_gates, PyLong_FromLong(qubit));
        Py_ssize_t gates_list_size = PyList_Size(gates_list);
        
        for (Py_ssize_t j = 0; j < gates_list_size; j++) {
            PyObject* gate_obj = PyList_GetItem(gates_list, j);
            PyObject* gate_obj_type = PyDict_GetItemString(gate_obj, "type");
            const char* gate_obj_type_str = PyUnicode_AsUTF8(gate_obj_type);
            
            if (strcmp(gate_obj_type_str, "u3") == 0) {
                PyObject* add_u3_func = PyObject_GetAttrString(final_layer_result, "add_U3");
                PyObject* add_u3_args = Py_BuildValue("(iOOO)", qubit, Py_True, Py_True, Py_True);
                PyObject_CallObject(add_u3_func, add_u3_args);
                Py_DECREF(add_u3_func);
                Py_DECREF(add_u3_args);
                
                PyObject* gate_obj_params = PyDict_GetItemString(gate_obj, "params");
                PyObject* reversed_params = PyList_New(0);
                for (Py_ssize_t k = PyList_Size(gate_obj_params) - 1; k >= 0; k--) {
                    PyList_Append(reversed_params, PyList_GetItem(gate_obj_params, k));
                }
                
                // Convert parameters to float and append
                for (Py_ssize_t k = 0; k < PyList_Size(reversed_params); k++) {
                    PyObject* param = PyList_GetItem(reversed_params, k);
                    PyObject* param_float = PyFloat_FromDouble(PyFloat_AsDouble(param));
                    PyList_Append(optimized_parameters, param_float);
                    Py_DECREF(param_float);
                }
                Py_DECREF(reversed_params);
                
                // Divide last parameter by 2
                Py_ssize_t optimized_parameters_last_idx = PyList_Size(optimized_parameters) - 1;
                PyObject* optimized_parameters_last_param = PyList_GetItem(optimized_parameters, optimized_parameters_last_idx);
                double val = PyFloat_AsDouble(optimized_parameters_last_param) / 2.0;
                PyList_SetItem(optimized_parameters, optimized_parameters_last_idx, PyFloat_FromDouble(val));
            }
        }
    }
    
    PyObject* add_final_circuit_func = PyObject_GetAttrString(Circuit_ret_result, "add_Circuit");
    PyObject* add_final_circuit_args = PyTuple_Pack(1, final_layer_result);
    PyObject_CallObject(add_final_circuit_func, add_final_circuit_args);
    Py_DECREF(add_final_circuit_func);
    Py_DECREF(add_final_circuit_args);
    Py_DECREF(final_layer_result);

    // Convert parameters to numpy array and flip
    PyObject* numpy_module = PyImport_ImportModule("numpy");
    PyObject* numpy_asarray_func = PyObject_GetAttrString(numpy_module, "asarray");
    PyObject* numpy_flip_func = PyObject_GetAttrString(numpy_module, "flip");
    Py_DECREF(numpy_module);
    
    PyObject* dtype_dict = PyDict_New();
    PyDict_SetItemString(dtype_dict, "dtype", (PyObject*)&PyFloat_Type);
    PyObject* numpy_asarray_args = PyTuple_Pack(1, optimized_parameters);
    PyObject* numpy_asarray_result = PyObject_Call(numpy_asarray_func, numpy_asarray_args, dtype_dict);
    Py_DECREF(numpy_asarray_func);
    Py_DECREF(numpy_asarray_args);
    Py_DECREF(dtype_dict);
    
    PyObject* numpt_flip_args = PyTuple_Pack(2, numpy_asarray_result, PyLong_FromLong(0));
    PyObject* numpt_flip_result = PyObject_CallObject(numpy_flip_func, numpt_flip_args);
    Py_DECREF(numpy_flip_func);
    Py_DECREF(numpt_flip_args);
    Py_DECREF(numpy_asarray_result);

    // Set gate structure and parameters
    PyObject* set_gate_structure_args = PyTuple_Pack(1, Circuit_ret_result);
    PyObject* set_gate_structure_result = qgd_N_Qubit_Decomposition_Wrapper_set_Gate_Structure(self, set_gate_structure_args);
    Py_DECREF(set_gate_structure_args);
    Py_DECREF(Circuit_ret_result);
    if (!set_gate_structure_result) {
        Py_DECREF(numpt_flip_result);
        Py_DECREF(optimized_parameters);
        Py_DECREF(single_qubit_gates);
        Py_DECREF(qc_data_attr);
        Py_DECREF(qc_qubits_attr);
        Py_DECREF(qc);
        return NULL;
    }
    Py_DECREF(set_gate_structure_result);

    PyObject* set_optimized_params_args = PyTuple_Pack(1, numpt_flip_result);
    PyObject* set_optimized_params_result = qgd_N_Qubit_Decomposition_Wrapper_set_Optimized_Parameters(self, set_optimized_params_args);
    Py_DECREF(set_optimized_params_args);
    Py_DECREF(numpt_flip_result);
    Py_DECREF(optimized_parameters);
    Py_DECREF(single_qubit_gates);
    Py_DECREF(qc_data_attr);
    Py_DECREF(qc_qubits_attr);
    Py_DECREF(qc);
    if (!set_optimized_params_result) {
        return NULL;
    }
    Py_DECREF(set_optimized_params_result);

    Py_RETURN_NONE;
}

/**
@brief Method to import Qiskit circuit
@param qc_in Qiskit QuantumCircuit to import
@return Returns Py_None on success
*/
static PyObject*
qgd_N_Qubit_Decomposition_Wrapper_import_Qiskit_Circuit(qgd_N_Qubit_Decomposition_Wrapper* self, PyObject* args)
{
    PyObject* qc_in = NULL;
    if (!PyArg_ParseTuple(args, "O", &qc_in)) {
        return NULL;
    }
    bool is_adaptive = (dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp) != nullptr);
    if (is_adaptive) {
        return qgd_N_Qubit_Decomposition_Wrapper_import_Qiskit_Circuit_adaptive(self, qc_in);
    } else {
        return qgd_N_Qubit_Decomposition_Wrapper_import_Qiskit_Circuit_standard(self, qc_in);
    }
}





//////////////////////////////////////////////////////////////////

extern "C"
{

/**
@brief Base methods shared by all decomposition types
These methods are available for all decomposition classes
*/
#define DECOMPOSITION_WRAPPER_BASE_METHODS \
    {"Start_Decomposition", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_Start_Decomposition, METH_VARARGS | METH_KEYWORDS, \
     "Method to start the decomposition"}, \
    {"get_Gate_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Gate_Num, METH_NOARGS, \
     "Method to get the number of decomposing gates"}, \
    {"get_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Optimized_Parameters, METH_NOARGS, \
     "Method to get the array of optimized parameters"}, \
    {"get_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Circuit, METH_NOARGS, \
     "Method to get the incorporated circuit"}, \
    {"List_Gates", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_List_Gates, METH_NOARGS, \
     "Call to print the decomposing unitaries on standard output"}, \
    {"set_Max_Layer_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Max_Layer_Num, METH_VARARGS, \
     "Set the maximal number of layers used in the subdecomposition"}, \
    {"set_Iteration_Loops", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Iteration_Loops, METH_VARARGS, \
     "Set the number of iteration loops during the subdecomposition"}, \
    {"set_Verbose", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Verbose, METH_VARARGS, \
     "Set the verbosity of the decomposition class"}, \
    {"set_Debugfile", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Debugfile, METH_VARARGS, \
     "Set the debugfile name of the decomposition class"}, \
    {"Reorder_Qubits", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_Reorder_Qubits, METH_VARARGS, \
     "Method to reorder the qubits in the decomposition class"}, \
    {"set_Optimization_Tolerance", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Optimization_Tolerance, METH_VARARGS, \
     "Wrapper method to set the optimization tolerance"}, \
    {"set_Convergence_Threshold", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Convergence_Threshold, METH_VARARGS, \
     "Wrapper method to set the threshold of convergence"}, \
    {"set_Optimization_Blocks", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Optimization_Blocks, METH_VARARGS, \
     "Wrapper method to set the number of gate blocks to be optimized"}, \
    {"get_Parameter_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Parameter_Num, METH_NOARGS, \
     "Get the number of free parameters"}, \
    {"set_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Optimized_Parameters, METH_VARARGS, \
     "Set the optimized parameters"}, \
    {"get_Num_of_Iters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Num_of_Iters, METH_NOARGS, \
     "Get the number of iterations"}, \
    {"export_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_export_Unitary, METH_VARARGS, \
     "Export unitary matrix"}, \
    {"get_Project_Name", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Project_Name, METH_NOARGS, \
     "Get the name of SQUANDER project"}, \
    {"set_Project_Name", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Project_Name, METH_VARARGS, \
     "Set the name of SQUANDER project"}, \
    {"get_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Global_Phase, METH_NOARGS, \
     "Call to get global phase"}, \
    {"set_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Global_Phase, METH_VARARGS, \
     "Set global phase"}, \
    {"apply_Global_Phase_Factor", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_apply_Global_Phase_Factor, METH_NOARGS, \
     "Apply global phase factor"}, \
    {"get_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Unitary, METH_NOARGS, \
     "Get Unitary Matrix"}, \
    {"set_Optimizer", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Optimizer, METH_VARARGS | METH_KEYWORDS, \
     "Set the optimizer method"}, \
    {"set_Max_Iterations", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Max_Iterations, METH_VARARGS | METH_KEYWORDS, \
     "Set the number of maximum iterations"}, \
    {"get_Matrix", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Matrix, METH_VARARGS | METH_KEYWORDS, \
     "Method to retrieve the unitary of the circuit"}, \
    {"set_Cost_Function_Variant", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Cost_Function_Variant, METH_VARARGS | METH_KEYWORDS, \
     "Set the cost function variant"}, \
    {"Optimization_Problem", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_Optimization_Problem, METH_VARARGS, \
     "Optimization problem method"}, \
    {"Optimization_Problem_Combined_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_Optimization_Problem_Combined_Unitary, METH_VARARGS, \
     "Optimization problem combined unitary method"}, \
    {"Optimization_Problem_Grad", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_Optimization_Problem_Grad, METH_VARARGS, \
     "Optimization problem gradient method"}, \
    {"Optimization_Problem_Combined", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_Optimization_Problem_Combined, METH_VARARGS, \
     "Optimization problem combined method"}, \
    {"Optimization_Problem_Batch", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_Optimization_Problem_Batch, METH_VARARGS, \
     "Optimization problem batch method"}, \
    {"Upload_Umtx_to_DFE", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_Upload_Umtx_to_DFE, METH_NOARGS, \
     "Upload unitary matrix to DFE"}, \
    {"get_Trace_Offset", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Trace_Offset, METH_NOARGS, \
     "Get trace offset"}, \
    {"set_Trace_Offset", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Trace_Offset, METH_VARARGS | METH_KEYWORDS, \
     "Set trace offset"}, \
    {"get_Decomposition_Error", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Decomposition_Error, METH_NOARGS, \
     "Get decomposition error"}, \
    {"get_Second_Renyi_Entropy", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Second_Renyi_Entropy, METH_VARARGS, \
     "Get second Renyi entropy"}, \
    {"get_Qbit_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Qbit_Num, METH_NOARGS, \
     "Get the number of qubits"}, \
    {"set_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Gate_Structure, METH_VARARGS, \
     "Set custom gate structure for decomposition"}, \
    {"add_Finalyzing_Layer_To_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_add_Finalyzing_Layer_To_Gate_Structure, METH_NOARGS, \
     "Add finalizing layer to gate structure"}, \
    {"get_Gates", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Gates, METH_NOARGS, \
     "Get gates as a list of dictionaries"}, \
    {"get_Qiskit_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Qiskit_Circuit, METH_NOARGS, \
     "Export decomposition to Qiskit QuantumCircuit format"}, \
    {"get_Cirq_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Cirq_Circuit, METH_NOARGS, \
     "Export decomposition to Cirq Circuit format"}, \
    {"import_Qiskit_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_import_Qiskit_Circuit, METH_VARARGS, \
     "Import Qiskit QuantumCircuit"}, \


/**
@brief Method table for base N_Qubit_Decomposition 
*/
static PyMethodDef qgd_N_Qubit_Decomposition_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS
    {"set_Identical_Blocks", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Identical_Blocks, METH_VARARGS,
     "Set the number of identical successive blocks during subdecomposition"},
    {NULL}
};

/**
@brief Method table for N_Qubit_Decomposition_adaptive
*/
static PyMethodDef qgd_N_Qubit_Decomposition_adaptive_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS
    {"get_Initial_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_Initial_Circuit, METH_NOARGS,
     "Method to get initial circuit in decomposition"},
    {"Compress_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_Compress_Circuit, METH_NOARGS,
     "Method to compress gate structure"},
    {"Finalize_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_Finalize_Circuit, METH_VARARGS | METH_KEYWORDS,
     "Method to finalize the decomposition"},
    {"set_Gate_Structure_From_Binary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Gate_Structure_From_Binary, METH_VARARGS,
     "Set gate structure from binary"},
    {"add_Gate_Structure_From_Binary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_add_Gate_Structure_From_Binary, METH_VARARGS,
     "Add gate structure from binary"},
    {"set_Unitary_From_Binary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Unitary_From_Binary, METH_VARARGS,
     "Set unitary from binary"},
    {"add_Adaptive_Layers", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_add_Adaptive_Layers, METH_NOARGS,
     "Call to add adaptive layers to the gate structure"},
    {"add_Layer_To_Imported_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_add_Layer_To_Imported_Gate_Structure, METH_VARARGS,
     "Add layer to imported gate structure"},
    {"apply_Imported_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_apply_Imported_Gate_Structure, METH_NOARGS,
     "Apply imported gate structure"},
    {"set_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Unitary, METH_VARARGS,
     "Call to set unitary matrix"},
    {NULL}
};

/**
@brief Method table for N_Qubit_Decomposition_custom
*/
static PyMethodDef qgd_N_Qubit_Decomposition_custom_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS
    {NULL}
};

/**
@brief Method table for N_Qubit_Decomposition_Tabu_Search
*/
static PyMethodDef qgd_N_Qubit_Decomposition_Tabu_Search_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS
    {"set_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Unitary, METH_VARARGS,
     "Call to set unitary matrix"},
    {NULL}
};

/**
@brief Method table for N_Qubit_Decomposition_Tree_Search
*/
static PyMethodDef qgd_N_Qubit_Decomposition_Tree_Search_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS
    {"set_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Unitary, METH_VARARGS,
     "Call to set unitary matrix"},
    {NULL}
};

#define decomposition_wrapper_type_template(decomp_class) \
static PyTypeObject qgd_##decomp_class##_Wrapper_Type = { \
    PyVarObject_HEAD_INIT(NULL, 0) \
    "qgd_N_Qubit_Decomposition_Wrapper." #decomp_class, /* tp_name */ \
    sizeof(qgd_N_Qubit_Decomposition_Wrapper), /* tp_basicsize */ \
    0, /* tp_itemsize */ \
    (destructor) qgd_N_Qubit_Decomposition_Wrapper_dealloc, /* tp_dealloc */ \
    0, /* tp_vectorcall_offset */ \
    0, /* tp_getattr */ \
    0, /* tp_setattr */ \
    0, /* tp_as_async */ \
    0, /* tp_repr */ \
    0, /* tp_as_number */ \
    0, /* tp_as_sequence */ \
    0, /* tp_as_mapping */ \
    0, /* tp_hash */ \
    0, /* tp_call */ \
    0, /* tp_str */ \
    0, /* tp_getattro */ \
    0, /* tp_setattro */ \
    0, /* tp_as_buffer */ \
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */ \
    #decomp_class " decomposition wrapper", /* tp_doc */ \
    0, /* tp_traverse */ \
    0, /* tp_clear */ \
    0, /* tp_richcompare */ \
    0, /* tp_weaklistoffset */ \
    0, /* tp_iter */ \
    0, /* tp_iternext */ \
    qgd_##decomp_class##_methods, /* tp_methods */ \
    0, /* tp_members */ \
    0, /* tp_getset */ \
    0, /* tp_base */ \
    0, /* tp_dict */ \
    0, /* tp_descr_get */ \
    0, /* tp_descr_set */ \
    0, /* tp_dictoffset */ \
    (initproc) qgd_##decomp_class##_Wrapper_init, /* tp_init */ \
    0, /* tp_alloc */ \
    (newfunc) qgd_N_Qubit_Decomposition_Wrapper_new, /* tp_new */ \
    0, /* tp_free */ \
    0, /* tp_is_gc */ \
    0, /* tp_bases */ \
    0, /* tp_mro */ \
    0, /* tp_cache */ \
    0, /* tp_subclasses */ \
    0, /* tp_weaklist */ \
    0, /* tp_del */ \
    0, /* tp_version_tag */ \
    0, /* tp_finalize */ \
    0, /* tp_vectorcall */ \
};

decomposition_wrapper_type_template(N_Qubit_Decomposition)
decomposition_wrapper_type_template(N_Qubit_Decomposition_adaptive)
decomposition_wrapper_type_template(N_Qubit_Decomposition_custom)
decomposition_wrapper_type_template(N_Qubit_Decomposition_Tree_Search)
decomposition_wrapper_type_template(N_Qubit_Decomposition_Tabu_Search)

//////////////////////////////////////////////////////////////////

/**
@brief Structure containing metadata about the module.
*/
static PyModuleDef qgd_N_Qubit_Decompositions_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    "qgd_N_Qubit_Decompositions_Wrapper", /* m_name */
    "Python binding for N-Qubit Decompositions wrapper module", /* m_doc */
    -1, /* m_size */
    0, /* m_methods */
    0, /* m_slots */
    0, /* m_traverse */
    0, /* m_clear */
    0, /* m_free */
};

#define Py_INCREF_template(decomp_name) \
    Py_INCREF(&qgd_##decomp_name##_Wrapper_Type); \
    if (PyModule_AddObject(m, "qgd_" #decomp_name, (PyObject *) &qgd_##decomp_name##_Wrapper_Type) < 0) { \
        Py_DECREF(&qgd_##decomp_name##_Wrapper_Type); \
        Py_DECREF(m); \
        return NULL; \
    }

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_N_Qubit_Decompositions_Wrapper(void)
{
    PyObject *m;
    
    // initialize numpy
    import_array();
    
    if (PyType_Ready(&qgd_N_Qubit_Decomposition_Wrapper_Type) < 0 || 
        PyType_Ready(&qgd_N_Qubit_Decomposition_adaptive_Wrapper_Type) < 0 ||
        PyType_Ready(&qgd_N_Qubit_Decomposition_custom_Wrapper_Type) < 0 ||
        PyType_Ready(&qgd_N_Qubit_Decomposition_Tree_Search_Wrapper_Type) < 0 ||
        PyType_Ready(&qgd_N_Qubit_Decomposition_Tabu_Search_Wrapper_Type) < 0) {
        return NULL;
    }

    m = PyModule_Create(&qgd_N_Qubit_Decompositions_Wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF_template(N_Qubit_Decomposition);
    Py_INCREF_template(N_Qubit_Decomposition_adaptive);
    Py_INCREF_template(N_Qubit_Decomposition_custom);
    Py_INCREF_template(N_Qubit_Decomposition_Tree_Search);
    Py_INCREF_template(N_Qubit_Decomposition_Tabu_Search);

    return m;
}

} // extern "C"
