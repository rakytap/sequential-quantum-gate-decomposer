/*
\file qgd_N_Qubit_Decomposition_Wrapper_New.cpp
\brief Python interface for N-Qubit Decomposition classes
*/
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <stdio.h>
#include <complex>
#include <cmath>

#include "numpy_interface.h"
#include "N_Qubit_Decomposition.h"
#include "N_Qubit_Decomposition_adaptive.h"
#include "N_Qubit_Decomposition_custom.h"
#include "N_Qubit_Decomposition_Tabu_Search.h"
#include "N_Qubit_Decomposition_Tree_Search.h"
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
typedef struct qgd_N_Qubit_Decomposition_Wrapper_New {
    PyObject_HEAD
    /// pointer to the unitary to be decomposed to keep it alive
    PyArrayObject* Umtx;
    /// An object to decompose the unitary
    Optimization_Interface* decomp;
} qgd_N_Qubit_Decomposition_Wrapper_New;

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
        return ZEROS;
    }
    const char* guess_str = PyUnicode_AsUTF8(PyObject_Str(initial_guess));
    if (strcasecmp("zeros", guess_str) == 0) return ZEROS;
    if (strcasecmp("random", guess_str) == 0) return RANDOM;
    if (strcasecmp("close_to_zero", guess_str) == 0) return CLOSE_TO_ZERO;
    std::cout << "Warning: Unknown guess '" << guess_str << "', using ZEROS" << std::endl;
    return ZEROS;
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

/**
 * @brief Validate qbit_num
 */
void validate_qbit_num(int qbit_num) {
    if (qbit_num <= 0) {
        throw std::runtime_error("qbit_num must be positive, got " + std::to_string(qbit_num));
    }
}

//////////////////////////////////////////////////////////////////

static int 
qgd_N_Qubit_Decomposition_Wrapper_init(qgd_N_Qubit_Decomposition_Wrapper_New* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {
        (char*)"Umtx", (char*)"qbit_num", (char*)"optimize_layer_num", 
        (char*)"initial_guess", NULL
    };
    
    PyObject *Umtx_arg = NULL, *initial_guess = NULL; 
    int qbit_num = -1;
    bool optimize_layer_num = false;
    
    if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "|OibO", kwlist,
        &Umtx_arg, &qbit_num, &optimize_layer_num, &initial_guess)
    ) {
        return -1;
    }

    try {
        Matrix Umtx_mtx = extract_matrix(Umtx_arg, &self->Umtx);
        validate_qbit_num(qbit_num);
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
qgd_N_Qubit_Decomposition_adaptive_Wrapper_init(qgd_N_Qubit_Decomposition_Wrapper_New* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {
        (char*)"Umtx", (char*)"qbit_num", (char*)"level_limit", 
        (char*)"level_limit_min", (char*)"topology", (char*)"config", 
        (char*)"accelerator_num", NULL
    };
    
    PyObject *Umtx_arg = NULL, *topology = NULL, *config_arg = NULL;
    int qbit_num = -1, level_limit = 0, level_limit_min = 0, accelerator_num = 0;
    
    if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "|OiiiOOi", kwlist,
        &Umtx_arg, &qbit_num, &level_limit, &level_limit_min, &topology, &config_arg, &accelerator_num)
    ) {
        return -1;
    }

    try {
        Matrix Umtx_mtx = extract_matrix(Umtx_arg, &self->Umtx);
        validate_qbit_num(qbit_num);
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
qgd_N_Qubit_Decomposition_custom_Wrapper_init(qgd_N_Qubit_Decomposition_Wrapper_New* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {
        (char*)"Umtx", (char*)"qbit_num", (char*)"initial_guess", 
        (char*)"config", (char*)"accelerator_num", NULL
    };
    
    PyObject *Umtx_arg = NULL, *initial_guess = NULL, *config_arg = NULL;
    int qbit_num = -1, accelerator_num = 0;
    
    if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "|OiOOi", kwlist,
        &Umtx_arg, &qbit_num, &initial_guess, &config_arg, &accelerator_num)
    ) {
        return -1;
    }

    try {
        Matrix Umtx_mtx = extract_matrix(Umtx_arg, &self->Umtx);
        validate_qbit_num(qbit_num);
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
static int search_wrapper_init(qgd_N_Qubit_Decomposition_Wrapper_New* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {
        (char*)"Umtx", (char*)"qbit_num", (char*)"topology", 
        (char*)"config", (char*)"accelerator_num", NULL
    };
    
    PyObject *Umtx_arg = NULL, *topology = NULL, *config_arg = NULL;
    int qbit_num = -1, accelerator_num = 0;
    
    if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "|OiOOi", kwlist,
        &Umtx_arg, &qbit_num, &topology, &config_arg, &accelerator_num)
    ) {
        return -1;
    }
    
    try {
        Matrix Umtx_mtx = extract_matrix(Umtx_arg, &self->Umtx);
        validate_qbit_num(qbit_num);
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
qgd_N_Qubit_Decomposition_Tree_Search_Wrapper_init(qgd_N_Qubit_Decomposition_Wrapper_New* self, PyObject* args, PyObject* kwds) {
    return search_wrapper_init<N_Qubit_Decomposition_Tree_Search>(self, args, kwds);
}

static int
qgd_N_Qubit_Decomposition_Tabu_Search_Wrapper_init(qgd_N_Qubit_Decomposition_Wrapper_New* self, PyObject* args, PyObject* kwds) {
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
qgd_N_Qubit_Decomposition_Wrapper_New_dealloc(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_N_Qubit_Decomposition_Wrapper_New *self;
    self = (qgd_N_Qubit_Decomposition_Wrapper_New *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->Umtx = NULL;
        self->decomp = NULL;
    }
    return (PyObject *) self;
}

////////////////////////////////////////////////////////////////// COMMON METHODs

/**
@brief Wrapper function to call the start_decomposition method of C++ class N_Qubit_Decomposition
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper_New.
@param kwds A tuple of keywords
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_Start_Decomposition(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args, PyObject *kwds)
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
qgd_N_Qubit_Decomposition_Wrapper_New_get_Gate_Num(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_get_Optimized_Parameters(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{

    int parameter_num = self->decomp->get_parameter_num();
    Matrix_real parameters_mtx(1, parameter_num);
    
    double* parameters = parameters_mtx.get_data();
    self->decomp->get_optimized_parameters(parameters);

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
qgd_N_Qubit_Decomposition_Wrapper_New_get_Circuit(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_List_Gates(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    // list gates with start_index = 0
    self->decomp->list_gates(0);
    return Py_None;
}

/**
@brief Set the maximal number of layers used in the subdecomposition of the qbit-th qubit
@param args Dictionary {'n': max_layer_num} labeling the maximal number of the gate layers used in the subdecomposition
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Max_Layer_Num(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Iteration_Loops(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Verbose(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Debugfile(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_Reorder_Qubits(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimization_Tolerance(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Convergence_Threshold(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimization_Blocks(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_add_Finalyzing_Layer_To_Gate_Structure(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper_New.
@param args PyObject containing either a dictionary {int: Gates_block} (Decomposition only) or a single gate structure (all types)
@return Returns with zero on success.
@note applicable to: Decomposition (both map and single), Adaptive, Custom, Tree Search, Tabu Search (single only)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Gate_Structure(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_get_Parameter_Num(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimized_Parameters(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    PyArrayObject* parameters_arr = NULL;
    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arr )) {
        return Py_BuildValue("i", -1);
    }
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
@brief Get the number of free parameters in the gate structure used for the decomposition
@return The number of iterations
@note applicable to: Adaptive, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Num_of_Iters(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    int number_of_iters = self->decomp->get_num_iters();   
    return Py_BuildValue("i", number_of_iters);
}

/**
@brief Export unitary matrix to binary file
@param args Tuple containing filename string
@return Py_None on success, NULL on error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_export_Unitary(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    const char* filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }
    try {
        std::string filename_str(filename);
        self->decomp->export_unitary(filename_str);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to get the project name
@return PyUnicode string with project name
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Project_Name(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimized_Parameters(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    PyArrayObject* parameters_arr = NULL;
    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arr )) {
        return Py_BuildValue("i", -1);
    }
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
@return Py_None on success, NULL on error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Project_Name(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }
    const char* project_name;
    if (!PyArg_ParseTuple(args, "s", &project_name)) {
        return NULL;
    }
    try {
        std::string project_name_str(project_name);
        self->decomp->set_project_name(project_name_str);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to get the global phase factor
@return PyComplex representing the global phase
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Global_Phase(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }
    try {
        QGD_Complex16 global_phase = self->decomp->get_global_phase_factor();
        return PyComplex_FromDoubles(global_phase.real, global_phase.imag);
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to set the global phase
@param args Tuple containing phase angle (double)
@return Py_None on success, NULL on error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Global_Phase(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }
    double phase_angle;
    if (!PyArg_ParseTuple(args, "d", &phase_angle)) {
        return NULL;
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
qgd_N_Qubit_Decomposition_Wrapper_New_apply_Global_Phase_Factor(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_get_Unitary(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimizer(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args, PyObject *kwds)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Max_Iterations(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_get_Matrix(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args, PyObject *kwds)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Cost_Function_Variant(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args, PyObject *kwds)
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
qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined_Unitary(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Grad(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Batch(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_Upload_Umtx_to_DFE(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_get_Trace_Offset(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
#else
    PyErr_SetString(PyExc_NotImplementedError, "upload_Umtx_to_DFE is only available when compiled with DFE support");
    return NULL;
#endif
}

/**
@brief Set trace offset for the compression
@param args Python tuple of arguments (trace_offset: int)
@return Py_None on success, NULL on error
@note applicable to: Decomposition, Adaptive, Custom, Tree Search, Tabu Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Trace_Offset(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args, PyObject *kwds)
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
qgd_N_Qubit_Decomposition_Wrapper_New_get_Decomposition_Error(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_get_Second_Renyi_Entropy(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_get_Qbit_Num(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Identical_Blocks(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_get_Initial_Circuit(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_Compress_Circuit(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_Finalize_Circuit(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args, PyObject *kwds)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Gate_Structure_From_Binary(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_add_Gate_Structure_From_Binary(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Unitary_From_Binary(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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
qgd_N_Qubit_Decomposition_Wrapper_New_add_Adaptive_Layers(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_add_Layer_To_Imported_Gate_Structure(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    // dynamic_cast to adaptive-specific type
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
qgd_N_Qubit_Decomposition_Wrapper_New_apply_Imported_Gate_Structure(qgd_N_Qubit_Decomposition_Wrapper_New *self)
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
qgd_N_Qubit_Decomposition_Wrapper_New_set_Unitary(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
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

extern "C"
{

/**
@brief Base methods shared by all decomposition types
These methods are available for all decomposition classes
*/
#define DECOMPOSITION_WRAPPER_BASE_METHODS \
    {"Start_Decomposition", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Start_Decomposition, METH_VARARGS | METH_KEYWORDS, \
     "Method to start the decomposition"}, \
    {"get_Gate_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Gate_Num, METH_NOARGS, \
     "Method to get the number of decomposing gates"}, \
    {"get_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Optimized_Parameters, METH_NOARGS, \
     "Method to get the array of optimized parameters"}, \
    {"get_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Circuit, METH_NOARGS, \
     "Method to get the incorporated circuit"}, \
    {"List_Gates", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_List_Gates, METH_NOARGS, \
     "Call to print the decomposing unitaries on standard output"}, \
    {"set_Max_Layer_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Max_Layer_Num, METH_VARARGS, \
     "Set the maximal number of layers used in the subdecomposition"}, \
    {"set_Iteration_Loops", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Iteration_Loops, METH_VARARGS, \
     "Set the number of iteration loops during the subdecomposition"}, \
    {"set_Verbose", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Verbose, METH_VARARGS, \
     "Set the verbosity of the decomposition class"}, \
    {"set_Debugfile", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Debugfile, METH_VARARGS, \
     "Set the debugfile name of the decomposition class"}, \
    {"Reorder_Qubits", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Reorder_Qubits, METH_VARARGS, \
     "Method to reorder the qubits in the decomposition class"}, \
    {"set_Optimization_Tolerance", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimization_Tolerance, METH_VARARGS, \
     "Wrapper method to set the optimization tolerance"}, \
    {"set_Convergence_Threshold", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Convergence_Threshold, METH_VARARGS, \
     "Wrapper method to set the threshold of convergence"}, \
    {"set_Optimization_Blocks", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimization_Blocks, METH_VARARGS, \
     "Wrapper method to set the number of gate blocks to be optimized"}, \
    {"get_Parameter_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Parameter_Num, METH_NOARGS, \
     "Get the number of free parameters"}, \
    {"set_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimized_Parameters, METH_VARARGS, \
     "Set the optimized parameters"}, \
    {"get_Num_of_Iters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Num_of_Iters, METH_NOARGS, \
     "Get the number of iterations"}, \
    {"export_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_export_Unitary, METH_VARARGS, \
     "Export unitary matrix"}, \
    {"get_Project_Name", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Project_Name, METH_NOARGS, \
     "Get the name of SQUANDER project"}, \
    {"set_Project_Name", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Project_Name, METH_VARARGS, \
     "Set the name of SQUANDER project"}, \
    {"get_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Global_Phase, METH_NOARGS, \
     "Call to get global phase"}, \
    {"set_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Global_Phase, METH_VARARGS, \
<<<<<<< HEAD
     "Set global phase"}, \
    {"apply_Global_Phase_Factor", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_apply_Global_Phase_Factor, METH_NOARGS, \
     "Apply global phase factor"}, \
    {"get_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Unitary, METH_NOARGS, \
     "Get Unitary Matrix"}, \
    {"set_Optimizer", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimizer, METH_VARARGS, \
     "Set the optimizer method"}, \
    {"set_Max_Iterations", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Max_Iterations, METH_VARARGS, \
     "Set the number of maximum iterations"}, \
    {"get_Matrix", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Matrix, METH_VARARGS | METH_KEYWORDS, \
     "Method to retrieve the unitary of the circuit"}, \
    {"set_Cost_Function_Variant", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Cost_Function_Variant, METH_VARARGS, \
     "Set the cost function variant"}, \
    {"Optimization_Problem", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem, METH_VARARGS, \
     "Optimization problem method"}, \
    {"Optimization_Problem_Combined_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined_Unitary, METH_VARARGS, \
     "Optimization problem combined unitary method"}, \
    {"Optimization_Problem_Grad", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Grad, METH_VARARGS, \
     "Optimization problem gradient method"}, \
    {"Optimization_Problem_Combined", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined, METH_VARARGS, \
     "Optimization problem combined method"}, \
    {"Optimization_Problem_Batch", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Batch, METH_VARARGS, \
     "Optimization problem batch method"}, \
    {"Upload_Umtx_to_DFE", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Upload_Umtx_to_DFE, METH_NOARGS, \
     "Upload unitary matrix to DFE"}, \
    {"get_Trace_Offset", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Trace_Offset, METH_NOARGS, \
     "Get trace offset"}, \
    {"set_Trace_Offset", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Trace_Offset, METH_VARARGS, \
     "Set trace offset"}, \
    {"get_Decomposition_Error", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Decomposition_Error, METH_NOARGS, \
     "Get decomposition error"}, \
    {"get_Second_Renyi_Entropy", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Second_Renyi_Entropy, METH_VARARGS, \
     "Get second Renyi entropy"}, \
    {"get_Qbit_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Qbit_Num, METH_NOARGS, \
     "Get the number of qubits"}, \
    {"set_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Gate_Structure, METH_VARARGS, \
     "Set custom gate structure for decomposition"}, \
    {"add_Finalyzing_Layer_To_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_add_Finalyzing_Layer_To_Gate_Structure, METH_NOARGS, \
     "Add finalizing layer to gate structure"}, \


/**
@brief Method table for base N_Qubit_Decomposition 
*/
static PyMethodDef qgd_N_Qubit_Decomposition_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS
    {"set_Identical_Blocks", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Identical_Blocks, METH_VARARGS,
     "Set the number of identical successive blocks during subdecomposition"},
    {NULL}
};

/**
@brief Method table for N_Qubit_Decomposition_adaptive
*/
static PyMethodDef qgd_N_Qubit_Decomposition_adaptive_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS
    {"get_Initial_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Initial_Circuit, METH_NOARGS,
     "Method to get initial circuit in decomposition"},
    {"Compress_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Compress_Circuit, METH_NOARGS,
     "Method to compress gate structure"},
    {"Finalize_Circuit", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Finalize_Circuit, METH_VARARGS | METH_KEYWORDS,
     "Method to finalize the decomposition"},
    {"set_Gate_Structure_From_Binary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Gate_Structure_From_Binary, METH_VARARGS,
     "Set gate structure from binary"},
    {"add_Gate_Structure_From_Binary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_add_Gate_Structure_From_Binary, METH_VARARGS,
     "Add gate structure from binary"},
    {"set_Unitary_From_Binary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Unitary_From_Binary, METH_VARARGS,
     "Set unitary from binary"},
    {"add_Adaptive_Layers", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_add_Adaptive_Layers, METH_NOARGS,
     "Call to add adaptive layers to the gate structure"},
    {"add_Layer_To_Imported_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_add_Layer_To_Imported_Gate_Structure, METH_VARARGS,
     "Add layer to imported gate structure"},
    {"apply_Imported_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_apply_Imported_Gate_Structure, METH_NOARGS,
     "Apply imported gate structure"},
    {"set_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Unitary, METH_VARARGS,
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
    {"set_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Unitary, METH_VARARGS,
     "Call to set unitary matrix"},
    {NULL}
};

/**
@brief Method table for N_Qubit_Decomposition_Tree_Search
*/
static PyMethodDef qgd_N_Qubit_Decomposition_Tree_Search_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS
    {"set_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Unitary, METH_VARARGS,
     "Call to set unitary matrix"},
    {NULL}
};

#define decomposition_wrapper_type_template(decomp_class) \
static PyTypeObject qgd_##decomp_class##_Wrapper_Type = { \
    PyVarObject_HEAD_INIT(NULL, 0) \
    .tp_name = "qgd_N_Qubit_Decomposition_Wrapper." #decomp_class, \
    .tp_basicsize = sizeof(qgd_N_Qubit_Decomposition_Wrapper_New), \
    .tp_itemsize = 0, \
    .tp_dealloc = (destructor) qgd_N_Qubit_Decomposition_Wrapper_New_dealloc, \
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, \
    .tp_doc = #decomp_class " decomposition wrapper", \
    .tp_methods = qgd_##decomp_class##_methods, \
    .tp_init = (initproc) qgd_##decomp_class##_Wrapper_init, \
    .tp_new = (newfunc) qgd_N_Qubit_Decomposition_Wrapper_New_new, \
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
static PyModuleDef qgd_N_Qubit_Decomposition_Wrapper_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "qgd_N_Qubit_Decomposition_Wrapper",
    .m_doc = "Python binding for N-Qubit Decomposition wrapper module",
    .m_size = -1,
};

#define Py_INCREF_template(decomp_name) \
    Py_INCREF(&qgd_##decomp_name##_Wrapper_Type); \
    if (PyModule_AddObject(m, #decomp_name, (PyObject *) &qgd_##decomp_name##_Wrapper_Type) < 0) { \
        Py_DECREF(&qgd_##decomp_name##_Wrapper_Type); \
        Py_DECREF(m); \
        return NULL; \
    }

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_N_Qubit_Decomposition_Wrapper(void)
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

    m = PyModule_Create(&qgd_N_Qubit_Decomposition_Wrapper_module);
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