/*
\file qgd_N_Qubit_Decomposition_Wrapper_New.cpp
\brief Python interface for N-Qubit Decomposition classes
*/
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <stdio.h>

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

// Helper functions
Matrix extract_matrix_from_args(PyObject* Umtx_arg, PyArrayObject** store_ref) {
    if (Umtx_arg == NULL) {
        throw std::runtime_error("Matrix argument is NULL");
    }
    *store_ref = (PyArrayObject*)PyArray_FROM_OTF(Umtx_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (*store_ref == NULL) {
        throw std::runtime_error("Failed to convert to numpy array");
    }
    if (!PyArray_IS_C_CONTIGUOUS(*store_ref)) {
        std::cout << "Umtx is not memory contiguous" << std::endl;
    }
    return numpy2matrix(*store_ref);
}

std::map<std::string, Config_Element> extract_config_from_args(PyObject* config_arg) {
    // empty config, TODO: implement proper config parsing
    std::map<std::string, Config_Element> config;
    return config;
}

guess_type extract_guess_type_from_args(PyObject* initial_guess_arg) {
    if (initial_guess_arg == NULL) {
        return ZEROS;  // Default
    }
    
    // Handle integer input
    if (PyLong_Check(initial_guess_arg)) {
        int guess_val = PyLong_AsLong(initial_guess_arg);
        return static_cast<guess_type>(guess_val);
    }
    
    // Handle string input
    PyObject* initial_guess_string = PyObject_Str(initial_guess_arg);
    PyObject* initial_guess_string_unicode = PyUnicode_AsEncodedString(initial_guess_string, "utf-8", "~E~");
    const char* initial_guess_C = PyBytes_AS_STRING(initial_guess_string_unicode);
    
    guess_type qgd_initial_guess;
    if (strcmp("zeros", initial_guess_C) == 0 || strcmp("ZEROS", initial_guess_C) == 0) {
        qgd_initial_guess = ZEROS;        
    }
    else if (strcmp("random", initial_guess_C) == 0 || strcmp("RANDOM", initial_guess_C) == 0) {
        qgd_initial_guess = RANDOM;        
    }
    else if (strcmp("close_to_zero", initial_guess_C) == 0 || strcmp("CLOSE_TO_ZERO", initial_guess_C) == 0) {
        qgd_initial_guess = CLOSE_TO_ZERO;        
    }
    else {
        std::cout << "Wrong initial guess format. Using default ZEROS." << std::endl; 
        qgd_initial_guess = ZEROS;     
    }
    
    Py_XDECREF(initial_guess_string);
    Py_XDECREF(initial_guess_string_unicode);
    
    return qgd_initial_guess;
}

//////////////////////////////////////////////////////////////////

template<typename DecompT>
void release_decomposition(DecompT* instance) {
    if (instance != NULL) {
        delete instance;
    }
}

/**
@brief Method called when a python instance is destroyed
@param self A pointer pointing to an instance of the wrapper.
*/
static void
qgd_N_Qubit_Decomposition_Wrapper_New_dealloc(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp != NULL) {
        release_decomposition(self->decomp);
        self->decomp = NULL;
    }

    if (self->Umtx != NULL) {
        Py_DECREF(self->Umtx);
        self->Umtx = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Specialized new function for N_Qubit_Decomposition
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"Umtx", (char*)"optimize_layer_num", (char*)"config", (char*)"initial_guess", NULL};
    
    PyObject *Umtx_arg = NULL;
    bool optimize_layer_num = false;
    PyObject *config_arg = NULL;
    int initial_guess_int = 0;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|bOi", kwlist,
                                    &Umtx_arg, &optimize_layer_num, &config_arg, &initial_guess_int)) {
        return NULL;
    }

    qgd_N_Qubit_Decomposition_Wrapper_New *self;
    self = (qgd_N_Qubit_Decomposition_Wrapper_New *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->Umtx = NULL;
        self->decomp = NULL;
        
        try {
            Matrix Umtx_matrix = extract_matrix_from_args(Umtx_arg, &self->Umtx);
            int qbit_num = (int)round(log2(Umtx_matrix.rows));
            std::map<std::string, Config_Element> config = extract_config_from_args(config_arg);
            guess_type initial_guess = extract_guess_type_from_args(initial_guess_int ? PyLong_FromLong(initial_guess_int) : NULL);
            
            self->decomp = new N_Qubit_Decomposition(Umtx_matrix, qbit_num, optimize_layer_num, config, initial_guess);
        } catch (...) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_Exception, "Failed to create N_Qubit_Decomposition instance");
            return NULL;
        }
    }
    
    return (PyObject *) self;
}

/**
@brief Specialized new function for N_Qubit_Decomposition_adaptive
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"Umtx", (char*)"level_limit", (char*)"level_limit_min", 
                             (char*)"topology", (char*)"config", (char*)"accelerator_num", NULL};
    
    PyObject *Umtx_arg = NULL;
    int level_limit = 3;
    int level_limit_min = 1;
    PyObject *topology_arg = NULL;
    PyObject *config_arg = NULL;
    int accelerator_num = 0;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iiOOi", kwlist,
                                    &Umtx_arg, &level_limit, &level_limit_min, 
                                    &topology_arg, &config_arg, &accelerator_num)) {
        return NULL;
    }

    qgd_N_Qubit_Decomposition_Wrapper_New *self;
    self = (qgd_N_Qubit_Decomposition_Wrapper_New *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->Umtx = NULL;
        self->decomp = NULL;
        
        try {
            Matrix Umtx_matrix = extract_matrix_from_args(Umtx_arg, &self->Umtx);
            int qbit_num = (int)round(log2(Umtx_matrix.rows));
            std::map<std::string, Config_Element> config = extract_config_from_args(config_arg);
            
            // Default topology (empty for now)
            std::vector<matrix_base<int>> topology_in;
            // TODO: Parse topology_arg if provided
            
            self->decomp = new N_Qubit_Decomposition_adaptive(Umtx_matrix, qbit_num, level_limit, 
                                                             level_limit_min, topology_in, config, accelerator_num);
        } catch (...) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_Exception, "Failed to create N_Qubit_Decomposition_adaptive instance");
            return NULL;
        }
    }
    
    return (PyObject *) self;
}

/**
@brief Specialized new function for N_Qubit_Decomposition_custom
*/
static PyObject *
qgd_N_Qubit_Decomposition_custom_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"Umtx", (char*)"optimize_layer_num", (char*)"config", 
                             (char*)"initial_guess", (char*)"accelerator_num", NULL};
    
    PyObject *Umtx_arg = NULL;
    bool optimize_layer_num = false;
    PyObject *config_arg = NULL;
    int initial_guess_int = 0;
    int accelerator_num = 0;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|bOii", kwlist,
                                    &Umtx_arg, &optimize_layer_num, &config_arg, 
                                    &initial_guess_int, &accelerator_num)) {
        return NULL;
    }

    qgd_N_Qubit_Decomposition_Wrapper_New *self;
    self = (qgd_N_Qubit_Decomposition_Wrapper_New *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->Umtx = NULL;
        self->decomp = NULL;
        
        try {
            Matrix Umtx_matrix = extract_matrix_from_args(Umtx_arg, &self->Umtx);
            int qbit_num = (int)round(log2(Umtx_matrix.rows));
            std::map<std::string, Config_Element> config = extract_config_from_args(config_arg);
            guess_type initial_guess = extract_guess_type_from_args(initial_guess_int ? PyLong_FromLong(initial_guess_int) : NULL);
            
            self->decomp = new N_Qubit_Decomposition_custom(Umtx_matrix, qbit_num, optimize_layer_num, 
                                                           config, initial_guess, accelerator_num);
        } catch (...) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_Exception, "Failed to create N_Qubit_Decomposition_custom instance");
            return NULL;
        }
    }
    
    return (PyObject *) self;
}

/**
@brief Specialized new function for N_Qubit_Decomposition_Tree_Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Tree_Search_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"Umtx", (char*)"topology", (char*)"config", (char*)"accelerator_num", NULL};
    
    PyObject *Umtx_arg = NULL;
    PyObject *topology_arg = NULL;
    PyObject *config_arg = NULL;
    int accelerator_num = 0;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOi", kwlist,
                                    &Umtx_arg, &topology_arg, &config_arg, &accelerator_num)) {
        return NULL;
    }

    qgd_N_Qubit_Decomposition_Wrapper_New *self;
    self = (qgd_N_Qubit_Decomposition_Wrapper_New *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->Umtx = NULL;
        self->decomp = NULL;
        
        try {
            Matrix Umtx_matrix = extract_matrix_from_args(Umtx_arg, &self->Umtx);
            int qbit_num = (int)round(log2(Umtx_matrix.rows));
            std::map<std::string, Config_Element> config = extract_config_from_args(config_arg);
            
            // Default topology (empty for now)
            std::vector<matrix_base<int>> topology_in;
            // TODO: Parse topology_arg if provided
            
            self->decomp = new N_Qubit_Decomposition_Tree_Search(Umtx_matrix, qbit_num, topology_in, config, accelerator_num);
        } catch (...) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_Exception, "Failed to create N_Qubit_Decomposition_Tree_Search instance");
            return NULL;
        }
    }
    
    return (PyObject *) self;
}

/**
@brief Specialized new function for N_Qubit_Decomposition_Tabu_Search
*/
static PyObject *
qgd_N_Qubit_Decomposition_Tabu_Search_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"Umtx", (char*)"topology", (char*)"config", (char*)"accelerator_num", NULL};
    
    PyObject *Umtx_arg = NULL;
    PyObject *topology_arg = NULL;
    PyObject *config_arg = NULL;
    int accelerator_num = 0;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOi", kwlist,
                                    &Umtx_arg, &topology_arg, &config_arg, &accelerator_num)) {
        return NULL;
    }

    qgd_N_Qubit_Decomposition_Wrapper_New *self;
    self = (qgd_N_Qubit_Decomposition_Wrapper_New *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->Umtx = NULL;
        self->decomp = NULL;
        
        try {
            Matrix Umtx_matrix = extract_matrix_from_args(Umtx_arg, &self->Umtx);
            int qbit_num = (int)round(log2(Umtx_matrix.rows));
            std::map<std::string, Config_Element> config = extract_config_from_args(config_arg);
            
            // Default topology (empty for now)
            std::vector<matrix_base<int>> topology_in;
            // TODO: Parse topology_arg if provided
            
            self->decomp = new N_Qubit_Decomposition_Tabu_Search(Umtx_matrix, qbit_num, topology_in, config, accelerator_num);
        } catch (...) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_Exception, "Failed to create N_Qubit_Decomposition_Tabu_Search instance");
            return NULL;
        }
    }
    
    return (PyObject *) self;
}

/**
@brief Generic new method that does basic allocation
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

/**
@brief Call to start the decomposition process
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_Start_Decomposition(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"finalize_decomp", NULL};
    bool finalize_decomp = true;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|b", kwlist, &finalize_decomp)) {
        return NULL;
    }

    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    try {
        // Try casting to N_Qubit_Decomposition first (supports finalize_decomp parameter)
        N_Qubit_Decomposition* n_qubit_decomp = dynamic_cast<N_Qubit_Decomposition*>(self->decomp);
        if (n_qubit_decomp != NULL) {
            n_qubit_decomp->start_decomposition(finalize_decomp);
            Py_RETURN_NONE;
        }
        
        // Try other types that only support parameterless start_decomposition
        N_Qubit_Decomposition_adaptive* adaptive_decomp = dynamic_cast<N_Qubit_Decomposition_adaptive*>(self->decomp);
        if (adaptive_decomp != NULL) {
            adaptive_decomp->start_decomposition();
            Py_RETURN_NONE;
        }
        
        N_Qubit_Decomposition_custom* custom_decomp = dynamic_cast<N_Qubit_Decomposition_custom*>(self->decomp);
        if (custom_decomp != NULL) {
            custom_decomp->start_decomposition();
            Py_RETURN_NONE;
        }
        
        N_Qubit_Decomposition_Tree_Search* tree_decomp = dynamic_cast<N_Qubit_Decomposition_Tree_Search*>(self->decomp);
        if (tree_decomp != NULL) {
            tree_decomp->start_decomposition();
            Py_RETURN_NONE;
        }
        
        N_Qubit_Decomposition_Tabu_Search* tabu_decomp = dynamic_cast<N_Qubit_Decomposition_Tabu_Search*>(self->decomp);
        if (tabu_decomp != NULL) {
            tabu_decomp->start_decomposition();
            Py_RETURN_NONE;
        }
        
        PyErr_SetString(PyExc_RuntimeError, "Unknown decomposition type");
        return NULL;
        
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }

    Py_RETURN_NONE;
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

    Matrix_real parameters = self->decomp->get_optimized_parameters();
    return matrix_real_to_numpy(parameters);
}

/**
@brief Call to get the number of gates
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Gates_Num(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    int gates_num = self->decomp->get_gate_num();
    return PyLong_FromLong(gates_num);
}

/**
@brief Call to get the number of parameters
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_New_get_Parameter_Num(qgd_N_Qubit_Decomposition_Wrapper_New *self)
{
    if (self->decomp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Decomposition object is NULL");
        return NULL;
    }

    int parameter_num = self->decomp->get_parameter_num();
    return PyLong_FromLong(parameter_num);
}

static int
qgd_N_Qubit_Decomposition_Wrapper_New_init(qgd_N_Qubit_Decomposition_Wrapper_New *self, PyObject *args, PyObject *kwds)
{
    return 0;
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

    Matrix decomposed_matrix = self->decomp->get_decomposed_matrix();
    return matrix_to_numpy(decomposed_matrix);
}

extern "C"
{

/**
@brief Structure containing metadata about the methods of all decomposition types
*/
#define DECOMPOSITION_WRAPPER_BASE_METHODS \
    {"Start_Decomposition", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_Start_Decomposition, METH_VARARGS | METH_KEYWORDS, \
     "Start the decomposition process"}, \
    {"get_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Optimized_Parameters, METH_NOARGS, \
     "Get the optimized parameters"}, \
    {"get_Gate_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Gates_Num, METH_NOARGS, \
     "Get the number of gates"}, \
    {"get_Parameter_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Parameter_Num, METH_NOARGS, \
     "Get the number of parameters"}, \
    {"get_Matrix", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_New_get_Matrix, METH_VARARGS | METH_KEYWORDS, \
     "Get the decomposed matrix"}


// Single method table for all decomposition types
static PyMethodDef qgd_N_Qubit_Decomposition_methods[] = {
    DECOMPOSITION_WRAPPER_BASE_METHODS,
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
    .tp_methods = qgd_N_Qubit_Decomposition_methods, \
    .tp_init = (initproc) qgd_N_Qubit_Decomposition_Wrapper_New_init, \
    .tp_new = (newfunc) qgd_##decomp_class##_Wrapper_new, \
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