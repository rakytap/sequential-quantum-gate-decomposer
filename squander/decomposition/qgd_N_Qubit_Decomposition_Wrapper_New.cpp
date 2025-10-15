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

//////////////////////////////////////////////////////////////////

// Helper macro for dynamic casting to all decomposition types
#define DYNAMIC_CAST_AND_CALL(method_call, self) \
    do { \
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp); \
        if (adaptive_decomp) { \
            adaptive_decomp->method_call; \
            break; \
        } \
        N_Qubit_Decomposition_custom* custom_decomp = dynamic_cast<N_Qubit_Decomposition_custom*>(self->decomp); \
        if (custom_decomp) { \
            custom_decomp->method_call; \
            break; \
        } \
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp); \
        if (tree_decomp) { \
            tree_decomp->method_call; \
            break; \
        } \
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp); \
        if (tabu_decomp) { \
            tabu_decomp->method_call; \
            break; \
        } \
        N_Qubit_Decomposition* basic_decomp = dynamic_cast<N_Qubit_Decomposition*>(self->decomp); \
        if (basic_decomp) { \
            basic_decomp->method_call; \
            break; \
        } \
        PyErr_SetString(PyExc_TypeError, "Unknown decomposition type"); \
    } while(0)

#define DYNAMIC_CAST_AND_RETURN(method_call, self, return_type) \
    do { \
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp); \
        if (adaptive_decomp) { \
            return adaptive_decomp->method_call; \
        } \
        N_Qubit_Decomposition_custom* custom_decomp = dynamic_cast<N_Qubit_Decomposition_custom*>(self->decomp); \
        if (custom_decomp) { \
            return custom_decomp->method_call; \
        } \
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp); \
        if (tree_decomp) { \
            return tree_decomp->method_call; \
        } \
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp); \
        if (tabu_decomp) { \
            return tabu_decomp->method_call; \
        } \
        N_Qubit_Decomposition* basic_decomp = dynamic_cast<N_Qubit_Decomposition*>(self->decomp); \
        if (basic_decomp) { \
            return basic_decomp->method_call; \
        } \
        PyErr_SetString(PyExc_TypeError, "Unknown decomposition type"); \
        return return_type(); \
    } while(0)

//////////////////////////////////////////////////////////////////

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
        
        self->decomp = new N_Qubit_Decomposition_custom(Umtx_mtx, qbit_num, false, guess, config, accelerator_num);
        
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

//////////////////////////////////////////////////////////////////

// =========================================================================
// METHOD IMPLEMENTATIONS - All methods from the method tables
// =========================================================================

/**
@brief Call to start the decomposition process
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_Start_Decomposition(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"finalize_decomp", NULL};
    
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        // Check if it's base N_Qubit_Decomposition (supports finalize_decomp parameter)
        N_Qubit_Decomposition* base_decomp = dynamic_cast<N_Qubit_Decomposition*>(self->decomp);
        if (base_decomp != NULL) {
            bool finalize_decomp = true;
            if (!PyArg_ParseTupleAndKeywords(args, kwds, "|b", kwlist, &finalize_decomp)) {
                return NULL;
            }
            base_decomp->start_decomposition(finalize_decomp);
            Py_RETURN_NONE;
        }
        
        // For adaptive, custom, tree, and tabu search - no parameters
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist)) {
            return NULL;
        }
        
        // Cast to appropriate derived class and call start_decomposition
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp) {
            adaptive_decomp->start_decomposition();
            Py_RETURN_NONE;
        }
        
        N_Qubit_Decomposition_custom* custom_decomp = dynamic_cast<N_Qubit_Decomposition_custom*>(self->decomp);
        if (custom_decomp) {
            custom_decomp->start_decomposition();
            Py_RETURN_NONE;
        }
        
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp) {
            tree_decomp->start_decomposition();
            Py_RETURN_NONE;
        }
        
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp) {
            tabu_decomp->start_decomposition();
            Py_RETURN_NONE;
        }
        
        N_Qubit_Decomposition* basic_decomp = dynamic_cast<N_Qubit_Decomposition*>(self->decomp);
        if (basic_decomp) {
            basic_decomp->start_decomposition();
            Py_RETURN_NONE;
        }
        
        PyErr_SetString(PyExc_TypeError, "Unknown decomposition type");
        return NULL;
        
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to get the number of gates
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Gate_Num(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        int gates_num = self->decomp->get_gate_num();
        return PyLong_FromLong(gates_num);
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to get the optimized parameters
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Optimized_Parameters(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        Matrix_real parameters = self->decomp->get_optimized_parameters();
        return matrix_real_to_numpy(parameters);
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to get the incorporated circuit
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Circuit(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        // Import the circuit module to get the wrapper type
        PyObject* qgd_Circuit = PyImport_ImportModule("squander.gates.qgd_Circuit");
        if (qgd_Circuit == NULL) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_Circuit");
            return NULL;
        }
        ret->gate = self->decomp;
        return (PyObject*)ret;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to list the gates decomposing the unitary
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_List_Gates(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        self->decomp->list_gates(0);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Set the maximal number of layers used in the subdecomposition
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Max_Layer_Num(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    PyObject* max_layer_num_dict;
    if (!PyArg_ParseTuple(args, "O", &max_layer_num_dict)) {
        return NULL;
    }

    if (!PyDict_Check(max_layer_num_dict)) {
        PyErr_SetString(PyExc_TypeError, "Expected dictionary argument");
        return NULL;
    }

    try {
        std::map<int, int> max_layer_num_map;
        
        PyObject* key;
        PyObject* value;
        Py_ssize_t pos = 0;
        
        while (PyDict_Next(max_layer_num_dict, &pos, &key, &value)) {
            if (!PyLong_Check(key) || !PyLong_Check(value)) {
                PyErr_SetString(PyExc_TypeError, "Dictionary keys and values must be integers");
                return NULL;
            }
            
            int qubit_idx = PyLong_AsLong(key);
            int max_layers = PyLong_AsLong(value);
            max_layer_num_map[qubit_idx] = max_layers;
        }
        
        self->decomp->set_max_layer_num(max_layer_num_map);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Set the number of iteration loops during the subdecomposition
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Iteration_Loops(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    PyObject* iteration_loops_dict;
    if (!PyArg_ParseTuple(args, "O", &iteration_loops_dict)) {
        return NULL;
    }

    if (!PyDict_Check(iteration_loops_dict)) {
        PyErr_SetString(PyExc_TypeError, "Expected dictionary argument");
        return NULL;
    }

    try {
        std::map<int, int> iteration_loops_map;
        
        PyObject* key;
        PyObject* value;
        Py_ssize_t pos = 0;
        
        while (PyDict_Next(iteration_loops_dict, &pos, &key, &value)) {
            if (!PyLong_Check(key) || !PyLong_Check(value)) {
                PyErr_SetString(PyExc_TypeError, "Dictionary keys and values must be integers");
                return NULL;
            }
            
            int qubit_idx = PyLong_AsLong(key);
            int loops = PyLong_AsLong(value);
            iteration_loops_map[qubit_idx] = loops;
        }
        
        self->decomp->set_iteration_loops(iteration_loops_map);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Set the verbosity of the decomposition class
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Verbose(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    int verbose;
    if (!PyArg_ParseTuple(args, "i", &verbose)) {
        return NULL;
    }

    try {
        self->decomp->set_verbose(verbose != 0);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Set the debugfile name of the decomposition class
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Debugfile(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    const char* debugfile_name;
    if (!PyArg_ParseTuple(args, "s", &debugfile_name)) {
        return NULL;
    }

    try {
        std::string debugfile_str(debugfile_name);
        self->decomp->set_debugfile(debugfile_str);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Method to reorder the qubits in the decomposition class
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_Reorder_Qubits(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    PyObject* qubit_list;
    if (!PyArg_ParseTuple(args, "O", &qubit_list)) {
        return NULL;
    }

    if (!PyList_Check(qubit_list)) {
        PyErr_SetString(PyExc_TypeError, "Expected list argument");
        return NULL;
    }

    try {
        std::vector<int> reordering_map;
        Py_ssize_t list_size = PyList_Size(qubit_list);
        
        for (Py_ssize_t i = 0; i < list_size; i++) {
            PyObject* item = PyList_GetItem(qubit_list, i);
            if (!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "List items must be integers");
                return NULL;
            }
            reordering_map.push_back(PyLong_AsLong(item));
        }
        
        self->decomp->reorder_qubits(reordering_map);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Wrapper method to set the optimization tolerance
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimization_Tolerance(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

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
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Convergence_Threshold(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

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
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimization_Blocks(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    int optimization_blocks;
    if (!PyArg_ParseTuple(args, "i", &optimization_blocks)) {
        return NULL;
    }

    try {
        DYNAMIC_CAST_AND_CALL(set_optimization_blocks(optimization_blocks), self);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

// =========================================================================
// TYPE-SPECIFIC METHOD IMPLEMENTATIONS
// =========================================================================

/**
@brief Set the number of identical successive blocks (N_Qubit_Decomposition only)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Identical_Blocks(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

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
        
        PyObject* key;
        PyObject* value;
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
@brief Call to get initial circuit (adaptive-specific)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Initial_Circuit(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "get_initial_circuit is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }

        // Call get_initial_circuit which returns void (just sets up the circuit)
        adaptive_decomp->get_initial_circuit();
        
        // Return success status like in the original implementation
        return Py_BuildValue("i", 0);
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to compress circuit (adaptive-specific)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_Compress_Circuit(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

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
@brief Call to finalize circuit (adaptive-specific)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_Finalize_Circuit(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args, PyObject *kwds)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

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
@brief Set gate structure from binary (adaptive-specific)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Gate_Structure_From_Binary(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    const char* filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }

    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "set_Gate_Structure_From_Binary is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }

        std::string filename_str(filename);
        adaptive_decomp->set_adaptive_gate_structure(filename_str);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Add gate structure from binary (adaptive-specific)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_add_Gate_Structure_From_Binary(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    const char* filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }

    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "add_Gate_Structure_From_Binary is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }

        std::string filename_str(filename);
        adaptive_decomp->add_Gate_Structure_From_Binary(filename_str);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Set unitary from binary (adaptive-specific)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Unitary_From_Binary(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    const char* filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }

    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "set_Unitary_From_Binary is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }

        std::string filename_str(filename);
        adaptive_decomp->set_Unitary_From_Binary(filename_str);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Add finalizing layer to gate structure (adaptive-specific)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_add_Finalyzing_Layer_To_Gate_Structure(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "add_Finalyzing_Layer_To_Gate_Structure is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }

        adaptive_decomp->add_Finalyzing_Layer_To_Gate_Structure();
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to add adaptive layers (adaptive-specific)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_add_Adaptive_Layers(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "add_adaptive_layers is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }

        adaptive_decomp->add_adaptive_layers();
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Add layer to imported gate structure (adaptive-specific)  
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_add_Layer_To_Imported_Gate_Structure(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    int layer_idx;
    if (!PyArg_ParseTuple(args, "i", &layer_idx)) {
        return NULL;
    }

    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "add_Layer_To_Imported_Gate_Structure is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }

        adaptive_decomp->add_Layer_To_Imported_Gate_Structure(layer_idx);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Apply imported gate structure (adaptive-specific)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_apply_Imported_Gate_Structure(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp == NULL) {
            PyErr_SetString(PyExc_AttributeError, "apply_Imported_Gate_Structure is only available for N_Qubit_Decomposition_adaptive");
            return NULL;
        }

        adaptive_decomp->apply_Imported_Gate_Structure();
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

// =========================================================================
// SHARED METHODS ACROSS MULTIPLE TYPES
// =========================================================================

/**
@brief Set custom gate structure for decomposition (adaptive & custom)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Gate_Structure(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    PyObject* gate_structure_dict;
    if (!PyArg_ParseTuple(args, "O", &gate_structure_dict)) {
        return NULL;
    }

    try {
        // Try adaptive first
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            // TODO: Implement gate structure conversion from dict
            // adaptive_decomp->set_gate_structure(...);
            Py_RETURN_NONE;
        }

        // Try custom
        N_Qubit_Decomposition_custom* custom_decomp = dynamic_cast<N_Qubit_Decomposition_custom*>(self->decomp);
        if (custom_decomp != NULL) {
            // TODO: Implement gate structure conversion from dict
            // custom_decomp->set_gate_structure(...);
            Py_RETURN_NONE;
        }

        PyErr_SetString(PyExc_AttributeError, "set_Gate_Structure is only available for adaptive and custom decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Get the number of free parameters
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Parameter_Num(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        int parameter_num = self->decomp->get_parameter_num();
        return PyLong_FromLong(parameter_num);
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Set the optimized parameters
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimized_Parameters(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    PyArrayObject* parameters_arg;
    if (!PyArg_ParseTuple(args, "O", &parameters_arg)) {
        return NULL;
    }

    try {
        PyArrayObject* params_numpy = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (params_numpy == NULL) {
            PyErr_SetString(PyExc_ValueError, "Failed to convert to numpy array");
            return NULL;
        }
        
        Matrix_real parameters = numpy2matrix_real(params_numpy);
        Py_DECREF(params_numpy);
        
        self->decomp->set_optimized_parameters(parameters);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Get the number of iterations (adaptive, tree, tabu)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Num_of_Iters(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        // Try adaptive first
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            int iters = adaptive_decomp->get_num_iters();
            return PyLong_FromLong(iters);
        }

        // Try tree search
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            int iters = tree_decomp->get_num_iters();
            return PyLong_FromLong(iters);
        }

        // Try tabu search
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp != NULL) {
            int iters = tabu_decomp->get_num_iters();
            return PyLong_FromLong(iters);
        }

        PyErr_SetString(PyExc_AttributeError, "get_Num_of_Iters is only available for adaptive, tree search, and tabu search decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to set unitary matrix (adaptive, tree search, tabu search)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Unitary(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    PyArrayObject *Umtx_arg = NULL;
    if (!PyArg_ParseTuple(args, "O", &Umtx_arg)) {
        return NULL;
    }

    try {
        PyArrayObject* Umtx_numpy = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)Umtx_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
        if (Umtx_numpy == NULL) {
            PyErr_SetString(PyExc_ValueError, "Failed to convert to numpy array");
            return NULL;
        }
        Matrix Umtx_matrix = numpy2matrix(Umtx_numpy);
        Py_DECREF(Umtx_numpy);

        // Try adaptive
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            adaptive_decomp->set_unitary(Umtx_matrix);
            Py_RETURN_NONE;
        }

        // Try tree search
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            tree_decomp->set_unitary(Umtx_matrix);
            Py_RETURN_NONE;
        }

        // Try tabu search
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp != NULL) {
            tabu_decomp->set_unitary(Umtx_matrix);
            Py_RETURN_NONE;
        }

        PyErr_SetString(PyExc_AttributeError, "set_unitary is only available for adaptive, tree search, and tabu search decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Export unitary matrix (adaptive, tree, tabu)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_export_Unitary(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    const char* filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }

    try {
        std::string filename_str(filename);

        // Try adaptive
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            adaptive_decomp->export_unitary(filename_str);
            Py_RETURN_NONE;
        }

        // Try tree search
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            tree_decomp->export_unitary(filename_str);
            Py_RETURN_NONE;
        }

        // Try tabu search
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp != NULL) {
            tabu_decomp->export_unitary(filename_str);
            Py_RETURN_NONE;
        }

        PyErr_SetString(PyExc_AttributeError, "export_Unitary is only available for adaptive, tree search, and tabu search decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to get the project name
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Project_Name(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
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
@brief Apply global phase factor (adaptive, tree)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_apply_Global_Phase_Factor(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        // Try adaptive
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            adaptive_decomp->apply_global_phase_factor();
            Py_RETURN_NONE;
        }

        // Try tree search
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            tree_decomp->apply_global_phase_factor();
            Py_RETURN_NONE;
        }

        PyErr_SetString(PyExc_AttributeError, "apply_Global_Phase_Factor is only available for adaptive and tree search decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to get the unitary matrix
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Unitary(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        Matrix Umtx = self->decomp->get_Umtx();
        return matrix_to_numpy(Umtx);
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to set the optimizer
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimizer(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    const char* optimizer_str;
    if (!PyArg_ParseTuple(args, "s", &optimizer_str)) {
        return NULL;
    }

    try {
        std::string optimizer(optimizer_str);
        optimization_aglorithms optimizer_type;
        if (optimizer == "ADAM") {
            optimizer_type = ADAM;
        } else if (optimizer == "BFGS") {
            optimizer_type = BFGS;
        } else if (optimizer == "BFGS2") {
            optimizer_type = BFGS2;
        } else if (optimizer == "ADAM_BATCHED") {
            optimizer_type = ADAM_BATCHED;
        } else if (optimizer == "AGENTS") {
            optimizer_type = AGENTS;
        } else if (optimizer == "COSINE") {
            optimizer_type = COSINE;
        } else if (optimizer == "AGENTS_COMBINED") {
            optimizer_type = AGENTS_COMBINED;
        } else {
            PyErr_SetString(PyExc_ValueError, "Unknown optimizer type");
            return NULL;
        }

        self->decomp->set_optimizer(optimizer_type);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Set the number of maximum iterations (adaptive, tree, tabu)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Max_Iterations(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    int max_iterations;
    if (!PyArg_ParseTuple(args, "i", &max_iterations)) {
        return NULL;
    }

    try {
        // Try adaptive
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            adaptive_decomp->set_max_inner_iterations(max_iterations);
            Py_RETURN_NONE;
        }

        // Try tree search
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            tree_decomp->set_max_inner_iterations(max_iterations);
            Py_RETURN_NONE;
        }

        // Try tabu search  
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp != NULL) {
            tabu_decomp->set_max_inner_iterations(max_iterations);
            Py_RETURN_NONE;
        }

        PyErr_SetString(PyExc_AttributeError, "set_Max_Iterations is only available for adaptive, tree search, and tabu search decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Call to get the decomposed matrix
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Matrix(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args, PyObject *kwds)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        Matrix decomposed_matrix = self->decomp->get_decomposed_matrix();
        return matrix_to_numpy(decomposed_matrix);
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

// Additional methods for remaining functionality

/**
@brief Set the cost function variant
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Cost_Function_Variant(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    int cost_function_variant;
    if (!PyArg_ParseTuple(args, "i", &cost_function_variant)) {
        return NULL;
    }

    try {
        self->decomp->set_cost_function_variant(cost_function_variant);
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Optimization problem method
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    PyArrayObject* parameters_arg;
    if (!PyArg_ParseTuple(args, "O", &parameters_arg)) {
        return NULL;
    }

    try {
        PyArrayObject* params_numpy = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (params_numpy == NULL) {
            PyErr_SetString(PyExc_ValueError, "Failed to convert to numpy array");
            return NULL;
        }
        
        Matrix_real parameters = numpy2matrix_real(params_numpy);
        Py_DECREF(params_numpy);
        
        double result = self->decomp->optimization_problem(parameters);
        return PyFloat_FromDouble(result);
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Optimization problem combined unitary method (adaptive, tree, tabu)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined_Unitary(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    PyArrayObject* parameters_arg;
    if (!PyArg_ParseTuple(args, "O", &parameters_arg)) {
        return NULL;
    }

    try {
        // Try adaptive
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            PyArrayObject* params_numpy = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            if (params_numpy == NULL) {
                PyErr_SetString(PyExc_ValueError, "Failed to convert to numpy array");
                return NULL;
            }
            Matrix_real parameters = numpy2matrix_real(params_numpy);
            Py_DECREF(params_numpy);
            
            double result = adaptive_decomp->optimization_problem_combined_unitary(parameters);
            return PyFloat_FromDouble(result);
        }

        // Try tree search
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            PyArrayObject* params_numpy = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            if (params_numpy == NULL) {
                PyErr_SetString(PyExc_ValueError, "Failed to convert to numpy array");
                return NULL;
            }
            Matrix_real parameters = numpy2matrix_real(params_numpy);
            Py_DECREF(params_numpy);
            
            double result = tree_decomp->optimization_problem_combined_unitary(parameters);
            return PyFloat_FromDouble(result);
        }

        // Try tabu search
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp != NULL) {
            PyArrayObject* params_numpy = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            if (params_numpy == NULL) {
                PyErr_SetString(PyExc_ValueError, "Failed to convert to numpy array");
                return NULL;
            }
            Matrix_real parameters = numpy2matrix_real(params_numpy);
            Py_DECREF(params_numpy);
            
            double result = tabu_decomp->optimization_problem_combined_unitary(parameters);
            return PyFloat_FromDouble(result);
        }

        PyErr_SetString(PyExc_AttributeError, "Optimization_Problem_Combined_Unitary is only available for adaptive, tree search, and tabu search decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Optimization problem gradient method
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Grad(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    PyArrayObject* parameters_arg;
    if (!PyArg_ParseTuple(args, "O", &parameters_arg)) {
        return NULL;
    }

    try {
        PyArrayObject* params_numpy = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (params_numpy == NULL) {
            PyErr_SetString(PyExc_ValueError, "Failed to convert to numpy array");
            return NULL;
        }
        
        Matrix_real parameters = numpy2matrix_real(params_numpy);
        Py_DECREF(params_numpy);
        
        Matrix_real grad = self->decomp->optimization_problem_grad(parameters);
        return matrix_real_to_numpy(grad);
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Optimization problem combined method
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    PyArrayObject* parameters_arg;
    if (!PyArg_ParseTuple(args, "O", &parameters_arg)) {
        return NULL;
    }

    try {
        PyArrayObject* params_numpy = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (params_numpy == NULL) {
            PyErr_SetString(PyExc_ValueError, "Failed to convert to numpy array");
            return NULL;
        }
        
        Matrix_real parameters = numpy2matrix_real(params_numpy);
        Py_DECREF(params_numpy);
        
        Matrix_real grad;
        double cost = self->decomp->optimization_problem_combined(parameters, grad);
        
        PyObject* result_tuple = PyTuple_New(2);
        PyTuple_SetItem(result_tuple, 0, PyFloat_FromDouble(cost));
        PyTuple_SetItem(result_tuple, 1, matrix_real_to_numpy(grad));
        
        return result_tuple;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Optimization problem batch method (adaptive, tree, tabu)
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Batch(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    PyArrayObject* parameters_arg;
    if (!PyArg_ParseTuple(args, "O", &parameters_arg)) {
        return NULL;
    }

    try {
        // Try adaptive
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            PyArrayObject* params_numpy = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            if (params_numpy == NULL) {
                PyErr_SetString(PyExc_ValueError, "Failed to convert to numpy array");
                return NULL;
            }
            Matrix_real parameters = numpy2matrix_real(params_numpy);
            Py_DECREF(params_numpy);
            
            double result = adaptive_decomp->optimization_problem_batch(parameters);
            return PyFloat_FromDouble(result);
        }

        // Try tree search
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            PyArrayObject* params_numpy = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            if (params_numpy == NULL) {
                PyErr_SetString(PyExc_ValueError, "Failed to convert to numpy array");
                return NULL;
            }
            Matrix_real parameters = numpy2matrix_real(params_numpy);
            Py_DECREF(params_numpy);
            
            double result = tree_decomp->optimization_problem_batch(parameters);
            return PyFloat_FromDouble(result);
        }

        // Try tabu search
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp != NULL) {
            PyArrayObject* params_numpy = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            if (params_numpy == NULL) {
                PyErr_SetString(PyExc_ValueError, "Failed to convert to numpy array");
                return NULL;
            }
            Matrix_real parameters = numpy2matrix_real(params_numpy);
            Py_DECREF(params_numpy);
            
            double result = tabu_decomp->optimization_problem_batch(parameters);
            return PyFloat_FromDouble(result);
        }

        PyErr_SetString(PyExc_AttributeError, "Optimization_Problem_Batch is only available for adaptive, tree search, and tabu search decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Upload unitary matrix to DFE
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_Upload_Umtx_to_DFE(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        self->decomp->upload_Umtx_to_DFE();
        Py_RETURN_NONE;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Get trace offset
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Trace_Offset(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        // Try adaptive
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            double offset = adaptive_decomp->get_trace_offset();
            return PyFloat_FromDouble(offset);
        }

        // Try tree search
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            double offset = tree_decomp->get_trace_offset();
            return PyFloat_FromDouble(offset);
        }

        // Try tabu search
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp != NULL) {
            double offset = tabu_decomp->get_trace_offset();
            return PyFloat_FromDouble(offset);
        }

        PyErr_SetString(PyExc_AttributeError, "get_Trace_Offset is only available for adaptive, tree search, and tabu search decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Set trace offset
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_set_Trace_Offset(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    double offset;
    if (!PyArg_ParseTuple(args, "d", &offset)) {
        return NULL;
    }

    try {
        // Try adaptive
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            adaptive_decomp->set_trace_offset(offset);
            Py_RETURN_NONE;
        }

        // Try tree search
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            tree_decomp->set_trace_offset(offset);
            Py_RETURN_NONE;
        }

        // Try tabu search
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp != NULL) {
            tabu_decomp->set_trace_offset(offset);
            Py_RETURN_NONE;
        }

        PyErr_SetString(PyExc_AttributeError, "set_Trace_Offset is only available for adaptive, tree search, and tabu search decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Get decomposition error
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Decomposition_Error(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        // Try adaptive
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            double error = adaptive_decomp->get_decomposition_error();
            return PyFloat_FromDouble(error);
        }

        // Try tree search
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            double error = tree_decomp->get_decomposition_error();
            return PyFloat_FromDouble(error);
        }

        // Try tabu search
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp != NULL) {
            double error = tabu_decomp->get_decomposition_error();
            return PyFloat_FromDouble(error);
        }

        PyErr_SetString(PyExc_AttributeError, "get_Decomposition_Error is only available for adaptive, tree search, and tabu search decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Get second Renyi entropy
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Second_Renyi_Entropy(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    int qubit_num;
    if (!PyArg_ParseTuple(args, "i", &qubit_num)) {
        return NULL;
    }

    try {
        // Try adaptive
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            double entropy = adaptive_decomp->get_second_Renyi_entropy(qubit_num);
            return PyFloat_FromDouble(entropy);
        }

        // Try tree search
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            double entropy = tree_decomp->get_second_Renyi_entropy(qubit_num);
            return PyFloat_FromDouble(entropy);
        }

        // Try tabu search
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp != NULL) {
            double entropy = tabu_decomp->get_second_Renyi_entropy(qubit_num);
            return PyFloat_FromDouble(entropy);
        }

        PyErr_SetString(PyExc_AttributeError, "get_Second_Renyi_Entropy is only available for adaptive, tree search, and tabu search decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/**
@brief Get the number of qubits
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Qbit_Num(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        // Try adaptive
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            int qbit_num = adaptive_decomp->get_qbit_num();
            return PyLong_FromLong(qbit_num);
        }

        // Try tree search
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            int qbit_num = tree_decomp->get_qbit_num();
            return PyLong_FromLong(qbit_num);
        }

        // Try tabu search
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp != NULL) {
            int qbit_num = tabu_decomp->get_qbit_num();
            return PyLong_FromLong(qbit_num);
        }

        PyErr_SetString(PyExc_AttributeError, "get_Qbit_Num is only available for adaptive, tree search, and tabu search decompositions");
        return NULL;
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
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
     "Wrapper method to set the number of gate blocks to be optimized"}

/**
@brief Method table for base N_Qubit_Decomposition 
*/
static PyMethodDef qgd_N_Qubit_Decomposition_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS,
    {"set_Identical_Blocks", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Identical_Blocks, METH_VARARGS,
     "Set the number of identical successive blocks during subdecomposition"},
    {NULL}
};

/**
@brief Method table for N_Qubit_Decomposition_adaptive
*/
static PyMethodDef qgd_N_Qubit_Decomposition_adaptive_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS,
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
    {"add_Finalyzing_Layer_To_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_add_Finalyzing_Layer_To_Gate_Structure, METH_NOARGS,
     "Add finalizing layer to gate structure"},
    {"add_Adaptive_Layers", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_add_Adaptive_Layers, METH_NOARGS,
     "Call to add adaptive layers to the gate structure"},
    {"add_Layer_To_Imported_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_add_Layer_To_Imported_Gate_Structure, METH_VARARGS,
     "Add layer to imported gate structure"},
    {"apply_Imported_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_apply_Imported_Gate_Structure, METH_NOARGS,
     "Apply imported gate structure"},
    {"set_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Gate_Structure, METH_VARARGS,
     "Set custom gate structure for decomposition"},
    {"get_Parameter_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Parameter_Num, METH_NOARGS,
     "Get the number of free parameters"},
    {"set_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimized_Parameters, METH_VARARGS,
     "Set the optimized parameters"},
    {"get_Num_of_Iters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Num_of_Iters, METH_NOARGS,
     "Get the number of iterations"},
    {"set_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Unitary, METH_VARARGS,
     "Call to set unitary matrix"},
    {"export_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_export_Unitary, METH_VARARGS,
     "Export unitary matrix"},
    {"get_Project_Name", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Project_Name, METH_NOARGS,
     "Get the name of SQUANDER project"},
    {"set_Project_Name", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Project_Name, METH_VARARGS,
     "Set the name of SQUANDER project"},
    {"get_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Global_Phase, METH_NOARGS,
     "Get global phase"},
    {"set_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Global_Phase, METH_VARARGS,
     "Set global phase"},
    {"apply_Global_Phase_Factor", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_apply_Global_Phase_Factor, METH_NOARGS,
     "Apply global phase factor"},
    {"get_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Unitary, METH_NOARGS,
     "Get Unitary Matrix"},
    {"set_Optimizer", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimizer, METH_VARARGS,
     "Set the optimizer method"},
    {"set_Max_Iterations", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Max_Iterations, METH_VARARGS,
     "Set the number of maximum iterations"},
    {"get_Matrix", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Matrix, METH_VARARGS | METH_KEYWORDS,
     "Method to retrieve the unitary of the circuit"},
    {"set_Cost_Function_Variant", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Cost_Function_Variant, METH_VARARGS,
     "Set the cost function variant"},
    {"Optimization_Problem", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem, METH_VARARGS,
     "Optimization problem method"},
    {"Optimization_Problem_Combined_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined_Unitary, METH_VARARGS,
     "Optimization problem combined unitary method"},
    {"Optimization_Problem_Grad", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Grad, METH_VARARGS,
     "Optimization problem gradient method"},
    {"Optimization_Problem_Combined", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined, METH_VARARGS,
     "Optimization problem combined method"},
    {"Optimization_Problem_Batch", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Batch, METH_VARARGS,
     "Optimization problem batch method"},
    {"Upload_Umtx_to_DFE", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Upload_Umtx_to_DFE, METH_NOARGS,
     "Upload unitary matrix to DFE"},
    {"get_Trace_Offset", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Trace_Offset, METH_NOARGS,
     "Get trace offset"},
    {"set_Trace_Offset", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Trace_Offset, METH_VARARGS,
     "Set trace offset"},
    {"get_Decomposition_Error", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Decomposition_Error, METH_NOARGS,
     "Get decomposition error"},
    {"get_Second_Renyi_Entropy", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Second_Renyi_Entropy, METH_VARARGS,
     "Get second Renyi entropy"},
    {"get_Qbit_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Qbit_Num, METH_NOARGS,
     "Get the number of qubits"},
    {NULL}
};

/**
@brief Method table for N_Qubit_Decomposition_custom
*/
static PyMethodDef qgd_N_Qubit_Decomposition_custom_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS,
    {"set_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Gate_Structure, METH_VARARGS,
     "Set custom gate structure for decomposition"},
    {"get_Parameter_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Parameter_Num, METH_NOARGS,
     "Get the number of free parameters"},
    {"set_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimized_Parameters, METH_VARARGS,
     "Set the optimized parameters"},
    {"set_Optimizer", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimizer, METH_VARARGS,
     "Set the optimizer method"},
    {"set_Cost_Function_Variant", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Cost_Function_Variant, METH_VARARGS,
     "Set the cost function variant"},
    {"Optimization_Problem", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem, METH_VARARGS,
     "Optimization problem method"},
    {"Optimization_Problem_Grad", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Grad, METH_VARARGS,
     "Optimization problem gradient method"},
    {"Optimization_Problem_Combined", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined, METH_VARARGS,
     "Optimization problem combined method"},
    {"Upload_Umtx_to_DFE", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Upload_Umtx_to_DFE, METH_NOARGS,
     "Upload unitary matrix to DFE"},
    {NULL}
};

/**
@brief Method table for N_Qubit_Decomposition_Tabu_Search
*/
static PyMethodDef qgd_N_Qubit_Decomposition_Tabu_Search_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS,
    {"get_Parameter_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Parameter_Num, METH_NOARGS,
     "Get the number of free parameters"},
    {"set_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimized_Parameters, METH_VARARGS,
     "Set the optimized parameters"},
    {"get_Num_of_Iters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Num_of_Iters, METH_NOARGS,
     "Get the number of iterations"},
    {"set_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Unitary, METH_VARARGS,
     "Call to set unitary matrix"},
    {"export_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_export_Unitary, METH_VARARGS,
     "Export unitary matrix"},
    {"get_Project_Name", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Project_Name, METH_NOARGS,
     "Get the name of SQUANDER project"},
    {"set_Project_Name", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Project_Name, METH_VARARGS,
     "Set the name of SQUANDER project"},
    {"get_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Global_Phase, METH_NOARGS,
     "Get global phase"},
    {"set_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Global_Phase, METH_VARARGS,
     "Set global phase"},
    {"get_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Unitary, METH_NOARGS,
     "Get Unitary Matrix"},
    {"set_Optimizer", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimizer, METH_VARARGS,
     "Set the optimizer method"},
    {"set_Max_Iterations", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Max_Iterations, METH_VARARGS,
     "Set the number of maximum iterations"},
    {"get_Matrix", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Matrix, METH_VARARGS | METH_KEYWORDS,
     "Method to retrieve the unitary of the circuit"},
    {"set_Cost_Function_Variant", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Cost_Function_Variant, METH_VARARGS,
     "Set the cost function variant"},
    {"Optimization_Problem", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem, METH_VARARGS,
     "Optimization problem method"},
    {"Optimization_Problem_Combined_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined_Unitary, METH_VARARGS,
     "Optimization problem combined unitary method"},
    {"Optimization_Problem_Grad", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Grad, METH_VARARGS,
     "Optimization problem gradient method"},
    {"Optimization_Problem_Combined", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined, METH_VARARGS,
     "Optimization problem combined method"},
    {"Optimization_Problem_Batch", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Batch, METH_VARARGS,
     "Optimization problem batch method"},
    {"Upload_Umtx_to_DFE", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Upload_Umtx_to_DFE, METH_NOARGS,
     "Upload unitary matrix to DFE"},
    {"get_Trace_Offset", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Trace_Offset, METH_NOARGS,
     "Get trace offset"},
    {"set_Trace_Offset", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Trace_Offset, METH_VARARGS,
     "Set trace offset"},
    {"get_Decomposition_Error", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Decomposition_Error, METH_NOARGS,
     "Get decomposition error"},
    {"get_Second_Renyi_Entropy", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Second_Renyi_Entropy, METH_VARARGS,
     "Get second Renyi entropy"},
    {"get_Qbit_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Qbit_Num, METH_NOARGS,
     "Get the number of qubits"},
    {NULL}
};

/**
@brief Method table for N_Qubit_Decomposition_Tree_Search
*/
static PyMethodDef qgd_N_Qubit_Decomposition_Tree_Search_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS,
    {"get_Parameter_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Parameter_Num, METH_NOARGS,
     "Get the number of free parameters"},
    {"set_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimized_Parameters, METH_VARARGS,
     "Set the optimized parameters"},
    {"get_Num_of_Iters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Num_of_Iters, METH_NOARGS,
     "Get the number of iterations"},
    {"set_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Unitary, METH_VARARGS,
     "Call to set unitary matrix"},
    {"export_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_export_Unitary, METH_VARARGS,
     "Export unitary matrix"},
    {"get_Project_Name", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Project_Name, METH_NOARGS,
     "Get the name of SQUANDER project"},
    {"set_Project_Name", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Project_Name, METH_VARARGS,
     "Set the name of SQUANDER project"},
    {"get_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Global_Phase, METH_NOARGS,
     "Get global phase"},
    {"set_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Global_Phase, METH_VARARGS,
     "Set global phase"},
    {"apply_Global_Phase_Factor", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_apply_Global_Phase_Factor, METH_NOARGS,
     "Apply global phase factor"},
    {"get_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Unitary, METH_NOARGS,
     "Get Unitary Matrix"},
    {"set_Optimizer", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Optimizer, METH_VARARGS,
     "Set the optimizer method"},
    {"set_Max_Iterations", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Max_Iterations, METH_VARARGS,
     "Set the number of maximum iterations"},
    {"get_Matrix", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Matrix, METH_VARARGS | METH_KEYWORDS,
     "Method to retrieve the unitary of the circuit"},
    {"set_Cost_Function_Variant", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Cost_Function_Variant, METH_VARARGS,
     "Set the cost function variant"},
    {"Optimization_Problem", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem, METH_VARARGS,
     "Optimization problem method"},
    {"Optimization_Problem_Combined_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined_Unitary, METH_VARARGS,
     "Optimization problem combined unitary method"},
    {"Optimization_Problem_Grad", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Grad, METH_VARARGS,
     "Optimization problem gradient method"},
    {"Optimization_Problem_Combined", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Combined, METH_VARARGS,
     "Optimization problem combined method"},
    {"Optimization_Problem_Batch", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Optimization_Problem_Batch, METH_VARARGS,
     "Optimization problem batch method"},
    {"Upload_Umtx_to_DFE", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Upload_Umtx_to_DFE, METH_NOARGS,
     "Upload unitary matrix to DFE"},
    {"get_Trace_Offset", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Trace_Offset, METH_NOARGS,
     "Get trace offset"},
    {"set_Trace_Offset", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_set_Trace_Offset, METH_VARARGS,
     "Set trace offset"},
    {"get_Decomposition_Error", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Decomposition_Error, METH_NOARGS,
     "Get decomposition error"},
    {"get_Second_Renyi_Entropy", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Second_Renyi_Entropy, METH_VARARGS,
     "Get second Renyi entropy"},
    {"get_Qbit_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Qbit_Num, METH_NOARGS,
     "Get the number of qubits"},
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