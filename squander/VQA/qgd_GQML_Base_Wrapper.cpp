/*
Created on Fri Jun 26 14:13:26 2020
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

@author: Peter Rakyta, Ph.D.
*/
/*! \file Gates_block.cpp
    \brief Class responsible for grouping two-qubit (CNOT,CZ,CH) and one-qubit gates into layers
*/

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <stdio.h>
#include "Generative_Quantum_Machine_Learning_Base.h"

#include "numpy_interface.h"





/**
@brief Type definition of the qgd_Circuit_Wrapper Python class of the qgd_Circuit_Wrapper module
*/
typedef struct qgd_Circuit_Wrapper {
    PyObject_HEAD
    Gates_block* gate;
} qgd_Circuit_Wrapper;


/**
@brief Type definition of the qgd_N_Qubit_Decomposition_Wrapper Python class of the qgd_N_Qubit_Decomposition_Wrapper module
*/
typedef struct qgd_Generative_Quantum_Machine_Learning_Base_Wrapper{
    PyObject_HEAD
    /// pointer to the unitary to be decomposed to keep it alive
    PyObject *x_vectors;
    PyObject *P_star;
    /// An object to decompose the unitary
    Generative_Quantum_Machine_Learning_Base* gqml;

} qgd_Generative_Quantum_Machine_Learning_Base_Wrapper;



/**
@brief Creates an instance of class Generative_Quantum_Machine_Learning_Base and return with a pointer pointing to the class instance (C++ linking is needed)
@param x_vectors The input data indices
@param x_bitstrings The input data bitstrings
@param P_star The distribution to approximate
@param sigma Parameter of the gaussian kernels
@param qbit_num The number of qubits spanning the unitary Umtx
@param use_lookup Use lookup table for the Gaussian kernels
@param cliques The cliques in the graph
@param use_exact Use exact calculation for MMD or just approximation with samples
@param config A map that can be used to set hyperparameters during the process
@return Return with a void pointer pointing to an instance of Generative_Quantum_Machine_Learning_Base class.
*/
Generative_Quantum_Machine_Learning_Base* 
create_qgd_Generative_Quantum_Machine_Learning_Base( std::vector<int> x_vectors, std::vector<std::vector<int>> x_bitstrings, Matrix_real P_star, Matrix_real sigma, int qbit_num, bool use_lookup_table, std::vector<std::vector<int>> cliques, bool use_exact, std::map<std::string, Config_Element>& config) {

    return new Generative_Quantum_Machine_Learning_Base( x_vectors, x_bitstrings, P_star, sigma, qbit_num, use_lookup_table, cliques, use_exact, config);
}


/**
@brief Call to deallocate an instance of Generative_Quantum_Machine_Learning_Base class
@param ptr A pointer pointing to an instance of Generative_Quantum_Machine_Learning_Base class.
*/
void
release_Generative_Quantum_Machine_Learning_Base( Generative_Quantum_Machine_Learning_Base*  instance ) {

    if (instance != NULL ) {
        delete instance;
    }
    return;
}



extern "C"
{


/**
@brief Method called when a python instance of the class qgd_Generative_Quantum_Machine_Learning_Base_Wrapper is destroyed
@param self A pointer pointing to an instance of class qgd_Generative_Quantum_Machine_Learning_Base_Wrapper.
*/
static void
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_dealloc(qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self)
{

    if ( self->gqml != NULL ) {
        // deallocate the instance of class Generative_Quantum_Machine_Learning_Base 
        release_Generative_Quantum_Machine_Learning_Base( self->gqml );
        self->gqml = NULL;
    }

    if ( self->x_vectors != NULL ) {
        // release the unitary to be decomposed
        Py_DECREF(self->x_vectors);    
        self->x_vectors = NULL;
    }

    if ( self->P_star != NULL ) {
        // release the unitary to be decomposed
        Py_DECREF(self->P_star);    
        self->P_star = NULL;
    }
    
    Py_TYPE(self)->tp_free((PyObject *) self);

}

/**
@brief Method called when a python instance of the class qgd_Generative_Quantum_Machine_Learning_Base_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_Generative_Quantum_Machine_Learning_Base_Wrapper.
*/
static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self;
    self = (qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->gqml = NULL;
    self->x_vectors = NULL;
    self->P_star = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_Generative_Quantum_Machine_Learning_Base_Wrapper is initialized
@param self A pointer pointing to an instance of the class qgd_Generative_Quantum_Machine_Learning_Base_Wrapper.
@param args A tuple of the input arguments: x_bitsring_data (numpy array), p_star_data (numpy array), sigma (double), qbit_num (integer), cliques (numpy array), use_lookup_table (bool), cliques (list), use_exact (bool)
@param kwds A tuple of keywords
*/
static int
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_init(qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"x_bitstring_data", (char*)"p_star_data", (char*) "sigma", (char*)"qbit_num", (char*)"use_lookup_table", (char*)"cliques", (char*)"use_exact", (char*)"config", NULL};
 
    // initiate variables for input arguments
    PyArrayObject *x_bitstring_data_arg = NULL;
    PyArrayObject *p_star_data_arg = NULL;
    PyObject *cliques_data_arg = NULL;
    PyArrayObject *sigma_data_arg;
    int  qbit_num = -1; 
    int use_lookup_table;
    PyObject *config_arg = NULL;
    int use_exact;
    
    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOipOpO", kwlist,
                                   &x_bitstring_data_arg, &p_star_data_arg, &sigma_data_arg, &qbit_num, &use_lookup_table, &cliques_data_arg, &use_exact,&config_arg))
        return -1;

    
    int shape = Power_of_2(qbit_num);
    // convert python object array to numpy C API array
    if ( x_bitstring_data_arg == NULL ) return -1;

    if ( !PyArray_ISINTEGER(x_bitstring_data_arg) && PyArray_TYPE(x_bitstring_data_arg) != NPY_BOOL) {
        PyErr_SetString(PyExc_TypeError, "x_bitstring_data should be int type or bool!" );
        return -1;
    }

    x_bitstring_data_arg    = (PyArrayObject*)PyArray_FROM_OF( (PyObject*)x_bitstring_data_arg, NPY_ARRAY_IN_ARRAY);
    int* x_bitsring_data    = (int*)PyArray_DATA(x_bitstring_data_arg);
    npy_intp* x_bistring_shape   = PyArray_DIMS(x_bitstring_data_arg);

    if (x_bistring_shape[1] != qbit_num) {
        std::cout<< shape << " " << x_bistring_shape[1] << std::endl;
        PyErr_SetString(PyExc_ValueError, "Each vector in x_bitsring_data should be qbit_num length!");
        return -1;
    }
    
    if ( p_star_data_arg == NULL ) return -1;
    
    p_star_data_arg  = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)p_star_data_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    double* p_star_data = (double*)PyArray_DATA(p_star_data_arg);
    int p_star_ndim = PyArray_NDIM(p_star_data_arg);
    int p_star_shape = static_cast<int>(PyArray_DIMS(p_star_data_arg)[0]);

    if ( p_star_ndim != 1 ) {
        PyErr_SetString(PyExc_ValueError, "p_star_data should be 1D array!");
        return -1;
    }
    
    std::vector<int> x_bitstrings_continous(x_bitsring_data, x_bitsring_data+(x_bistring_shape[0]*x_bistring_shape[1]));
    std::vector<std::vector<int>> x_bitstrings(x_bistring_shape[0], std::vector<int>(x_bistring_shape[1]));

    // Calculate which data corresponds with which element of the state vector
    std::vector<int> x_indices(x_bistring_shape[0], 0);
    for (int idx_data=0; idx_data < x_bistring_shape[0]; idx_data++) {
        for (int idx=0; idx < x_bistring_shape[1]; idx++) {
            x_bitstrings[idx_data][idx] = x_bitstrings_continous[idx_data*x_bistring_shape[1]+idx];
            if (x_bitstrings_continous[idx_data*x_bistring_shape[1]+idx] == 1) {
                x_indices[idx_data] += Power_of_2(qbit_num-idx-1);
            }
        }
    }

    Matrix_real p_stars = Matrix_real(p_star_data, p_star_shape, 1);

    if ( sigma_data_arg == NULL ) return -1;
    if (!PyList_Check(sigma_data_arg)) {
        PyErr_SetString(PyExc_TypeError, "sigma expected to be a list");
        return -1;
    }

    sigma_data_arg = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)sigma_data_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    double* sigma_data = (double*)PyArray_DATA(sigma_data_arg);
    int sigma_ndim = PyArray_NDIM(sigma_data_arg);
    int sigma_shape = static_cast<int>(PyArray_DIMS(sigma_data_arg)[0]);

    if ( sigma_ndim != 1 || sigma_shape != 3 ) {
        PyErr_SetString(PyExc_TypeError, "sigma expected to be a 1 by 3 list");
        return -1;
    }

    Matrix_real sigma(sigma_data, sigma_shape, 1);

    if ( cliques_data_arg == NULL ) return -1;
    if (!PyList_Check(cliques_data_arg)) {
        PyErr_SetString(PyExc_TypeError, "cliques expected to be a list of lists");
        return -1;
    }

    Py_ssize_t nrows = PyList_Size(cliques_data_arg);
    std::vector<std::vector<int>> cliques;

    for (Py_ssize_t i = 0; i < nrows; i++) {
        PyObject *row = PyList_GetItem(cliques_data_arg, i);  // borrowed ref
        std::vector<int> clique;

        if (!PyList_Check(row)) {
            PyErr_SetString(PyExc_TypeError, "cliques Expected a list of lists");
            return -1;
        }

        Py_ssize_t ncols = PyList_Size(row);

        for (Py_ssize_t j = 0; j < ncols; j++) {
            PyObject *item = PyList_GetItem(row, j);  // borrowed ref

            if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "List elements must be numbers");
                return -1;
            }

            clique.push_back(static_cast<int>(PyFloat_AsDouble(item)));
        }
        cliques.push_back(clique);
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

    // create an instance of the class Generative_Quantum_Machine_Learning_Base
    if (qbit_num > 0 ) {
        self->gqml =  create_qgd_Generative_Quantum_Machine_Learning_Base(x_indices, x_bitstrings, p_stars, sigma, qbit_num, use_lookup_table, cliques, use_exact, config);
    }
    else {
        std::cout << "The number of qubits should be given as a positive integer, " << qbit_num << "  was given" << std::endl;
        return -1;
    }


    std::cout << "reading args done" << std::endl;
    return 0;
}



/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_get_Optimized_Parameters( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self ) {

    int parameter_num = self->gqml->get_parameter_num();

    Matrix_real parameters_mtx(1, parameter_num);
    double* parameters = parameters_mtx.get_data();
    self->gqml->get_optimized_parameters(parameters);


    // convert to numpy array
    parameters_mtx.set_owner(false);
    PyObject * parameter_arr = matrix_real_to_numpy( parameters_mtx );

    return parameter_arr;
}


static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Start_Optimization(qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self)
{

    // starting the decomposition
    try {
        self->gqml->start_optimization();
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
@brief Call to retrieve the number of qubits in the circuit
*/
static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_get_Qbit_Num(qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self ) {

    int qbit_num = 0;

    try {
        qbit_num = self->gqml->get_qbit_num();
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
@brief Set parameters for the solver
*/
static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Optimized_Parameters( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args ) {

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
        self->gqml->set_optimized_parameters(parameters_mtx.get_data(), parameters_mtx.size());
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
@brief Set the initial state used in the VQE process
*/
static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Initial_State( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args ) {

    PyArrayObject * initial_state_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &initial_state_arg )) {
        PyErr_SetString(PyExc_Exception, "error occured during input parsing");
        return NULL;
    }
    
    if ( PyArray_IS_C_CONTIGUOUS(initial_state_arg) ) {
        Py_INCREF(initial_state_arg);
    }
    else {
        initial_state_arg = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)initial_state_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    Matrix initial_state_mtx = numpy2matrix( initial_state_arg );



    try {
        self->gqml->set_initial_state( initial_state_mtx );
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

    


    Py_DECREF(initial_state_arg);

    return Py_BuildValue("i", 0);
}


/**
@brief Wrapper function to set custom layers to the gate structure that are intended to be used in the decomposition.
*/
static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Gate_Structure_From_Binary(  qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args ) {



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
        self->gqml->set_gate_structure( filename_str );
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
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Optimization_Tolerance(qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    double tolerance;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|d", &tolerance )) return Py_BuildValue("i", -1);


    // set maximal layer nums on the C++ side
    self->gqml->set_optimization_tolerance( tolerance );


    return Py_BuildValue("i", 0);
}



static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_apply_to( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args ) {

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


    self->gqml->apply_to( parameters_mtx, unitary_mtx );
    
    if (unitary_mtx.data != PyArray_DATA(unitary)) {
        memcpy(PyArray_DATA(unitary), unitary_mtx.data, unitary_mtx.size() * sizeof(QGD_Complex16));
    }

    Py_DECREF(parameters_arr);
    Py_DECREF(unitary);

    return Py_BuildValue("i", 0);
}



static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Optimizer( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args, PyObject *kwds)
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
    if ( strcmp("agents", optimizer_C) == 0 || strcmp("AGENTS", optimizer_C) == 0) {
        qgd_optimizer = AGENTS;        
    }
    else if ( strcmp("agents_combined", optimizer_C)==0 || strcmp("AGENTS_COMBINED", optimizer_C)==0) {
        qgd_optimizer = AGENTS_COMBINED;        
    }
    else if ( strcmp("cosined", optimizer_C)==0 || strcmp("COSINE", optimizer_C)==0) {
        qgd_optimizer = COSINE;        
    }
    else if ( strcmp("grad_descend_phase_shift_rule", optimizer_C)==0 || strcmp("GRAD_DESCEND_PARAMETER_SHIFT_RULE", optimizer_C)==0) {
        qgd_optimizer = GRAD_DESCEND_PARAMETER_SHIFT_RULE;        
    }    
    else if ( strcmp("bfgs", optimizer_C)==0 || strcmp("BFGS", optimizer_C)==0) {
        qgd_optimizer = BFGS;        
    }
    else if ( strcmp("adam", optimizer_C)==0 || strcmp("ADAM", optimizer_C)==0) {
        qgd_optimizer = ADAM;        
    }
    else if ( strcmp("grad_descend", optimizer_C)==0 || strcmp("GRAD_DESCEND", optimizer_C)==0) {
        qgd_optimizer = GRAD_DESCEND;        
    }
    else if ( strcmp("bayes_opt", optimizer_C)==0 || strcmp("BAYES_OPT", optimizer_C)==0) {
        qgd_optimizer = BAYES_OPT;        
    }
    else if ( strcmp("bayes_agents", optimizer_C)==0 || strcmp("BAYES_AGENTS", optimizer_C)==0) {
        qgd_optimizer = BAYES_AGENTS;        
    }
    else {
        std::cout << "Wrong optimizer. Using default: AGENTS" << std::endl; 
        qgd_optimizer = AGENTS;     
    }
    
    
    try {
        self->gqml->set_optimizer(qgd_optimizer);
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
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Ansatz( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"optimizer", NULL};

    PyObject* ansatz_arg = NULL;


    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &ansatz_arg)) {

        std::string err( "Unsuccessful argument parsing not ");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;       
 
    }


    if ( ansatz_arg == NULL ) {
        std::string err( "optimizer argument not set");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;        
    }

   

    PyObject* ansatz_string = PyObject_Str(ansatz_arg);
    PyObject* ansatz_string_unicode = PyUnicode_AsEncodedString(ansatz_string, "utf-8", "~E~");
    const char* ansatz_C = PyBytes_AS_STRING(ansatz_string_unicode);


    ansatz_type qgd_ansatz;
    
    if ( strcmp("hea", ansatz_C) == 0 || strcmp("HEA", ansatz_C) == 0) {
        qgd_ansatz = HEA;        
    }
    else if ( strcmp("hea_zyz", ansatz_C) == 0 || strcmp("HEA_ZYZ", ansatz_C) == 0) {
        qgd_ansatz = HEA_ZYZ;        
    }
    else if ( strcmp("qcmrf", ansatz_C) == 0 || strcmp("QCMRF", ansatz_C) == 0) {
        qgd_ansatz = QCMRF;        
    }
    else {
        std::cout << "Wrong ansatz. Using default: HEA" << std::endl; 
        qgd_ansatz = HEA;     
    }
    
    
    try {
        self->gqml->set_ansatz(qgd_ansatz);
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
@brief Wrapper function to evaluate the second Rényi entropy of a quantum circuit at a specific parameter set.
*/
static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_get_Second_Renyi_Entropy( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args)
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
        parameters_arr = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }

    // get the C++ wrapper around the data
    Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );

    // convert python object array to numpy C API array
    if ( input_state_arg == NULL ) {
        PyErr_SetString(PyExc_Exception, "Input matrix was not given");
        return NULL;
    }

    PyArrayObject* input_state = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)input_state_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

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
        entropy = self->gqml->get_second_Renyi_entropy( parameters_mtx, input_state_mtx, qbit_list_mtx );
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


static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Generate_Circuit( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    int layers;
    int inner_blocks;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|ii", &layers, &inner_blocks )) return Py_BuildValue("i", -1);


    try {
        self->gqml->generate_circuit( layers, inner_blocks );
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
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Optimization_Problem( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args)
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
        f0 = self->gqml->optimization_problem(parameters_mtx );
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
@brief Get the number of free parameters in the gate structure used for the decomposition
*/
static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_get_Parameter_Num( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self ) {

    int parameter_num = self->gqml->get_parameter_num();

    return Py_BuildValue("i", parameter_num);
}





/**
@brief Wrapper function to retrieve the circuit (Squander format) incorporated in the instance.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
*/
static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_get_circuit( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self ) {


    PyObject* qgd_Circuit  = PyImport_ImportModule("squander.gates.qgd_Circuit");

    if ( qgd_Circuit == NULL ) {
        PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_Circuit" );
        return NULL;
    }

    // retrieve the C++ variant of the flat circuit (flat circuit does not conatain any sub-circuits)
    Gates_block* circuit = self->gqml->get_flat_circuit();



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
@brief Call to set a project name.
*/
static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Project_Name( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args ) {
    // initiate variables for input arguments
    PyObject* project_name_new=NULL; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &project_name_new)) return Py_BuildValue("i", -1);
    
    
    PyObject* project_name_new_string = PyObject_Str(project_name_new);
    PyObject* project_name_new_unicode = PyUnicode_AsEncodedString(project_name_new_string, "utf-8", "~E~");
    const char* project_name_new_C = PyBytes_AS_STRING(project_name_new_unicode);
    std::string project_name_new_str = ( project_name_new_C );
    
    // convert to python string
    self->gqml->set_project_name(project_name_new_str);

   return Py_BuildValue("i", 0);
}


/**
@brief Wrapper function to set custom gate structure for the decomposition.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@return Returns with zero on success.
*/
static PyObject *
qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Gate_Structure( qgd_Generative_Quantum_Machine_Learning_Base_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    PyObject* gate_structure_py; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &gate_structure_py )) return Py_BuildValue("i", -1);


    // convert gate structure from PyObject to qgd_Circuit_Wrapper
    qgd_Circuit_Wrapper* qgd_op_block = (qgd_Circuit_Wrapper*) gate_structure_py;

    try {
        self->gqml->set_custom_gate_structure( qgd_op_block->gate );
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
@brief Structure containing metadata about the members of class qgd_N_Qubit_Decomposition_Wrapper.
*/
static PyMemberDef qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_members[] = {
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the methods of class qgd_N_Qubit_Decomposition_Wrapper.
*/
static PyMethodDef qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_methods[] = {
    {"Start_Optimization", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Start_Optimization, METH_NOARGS,
     "Method to start the decomposition."
    },
    {"get_Optimized_Parameters", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_get_Optimized_Parameters, METH_NOARGS,
     "Method to get the array of optimized parameters."
    },
    {"set_Optimized_Parameters", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Optimized_Parameters, METH_VARARGS,
     "Method to set the array of optimized parameters."
    },
    {"set_Optimization_Tolerance", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Optimization_Tolerance, METH_VARARGS,
    "Method to set optimization tolerance"
    },
    {"get_Circuit", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_get_circuit, METH_NOARGS,
     "Method to get the incorporated circuit."
    },
    {"set_Project_Name", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Project_Name, METH_VARARGS,
    "method to set project name."
    },
    {"set_Gate_Structure_From_Binary", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Gate_Structure_From_Binary, METH_VARARGS,
     "Method to set the gate structure from a file created in SQUANDER."
    },
    {"apply_to", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_apply_to, METH_VARARGS,
     "Call to apply the gate on the input matrix."
    },
    {"set_Optimizer", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Optimizer, METH_VARARGS | METH_KEYWORDS,
     "Method to set optimizer."
    },
    {"set_Ansatz", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Ansatz, METH_VARARGS | METH_KEYWORDS,
     "Method to set ansatz type."
    },
    {"get_Parameter_Num", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_get_Parameter_Num, METH_NOARGS,
     "Call to get the number of free parameters in the gate structure used for the decomposition"
    },
    {"Generate_Circuit", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Generate_Circuit, METH_VARARGS,
     "Method to set the circuit based on the ansatz type."
    },
    {"Optimization_Problem", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Optimization_Problem, METH_VARARGS,
     "Method to get the expected energy of the circuit at parameters."
    },
    {"get_Second_Renyi_Entropy", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_get_Second_Renyi_Entropy, METH_VARARGS,
     "Wrapper function to evaluate the second Rényi entropy of a quantum circuit at a specific parameter set."
    },
    {"get_Qbit_Num", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_get_Qbit_Num, METH_NOARGS,
     "Call to get the number of qubits in the circuit"
    },
    {"set_Initial_State", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Initial_State, METH_VARARGS,
     "Call to set the initial state used in the VQE process."
    },
    {"set_Gate_Structure", (PyCFunction) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_set_Gate_Structure, METH_VARARGS,
     "Call to set custom gate structure for VQE experiments."
    },
    {NULL}  /* Sentinel */
};

/**
@brief A structure describing the type of the class qgd_N_Qubit_Decomposition_Wrapper.
*/
static PyTypeObject qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_N_Qubit_Decomposition_Wrapper.qgd_N_Qubit_Decomposition_Wrapper", /*tp_name*/
  sizeof(qgd_Generative_Quantum_Machine_Learning_Base_Wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_dealloc, /*tp_dealloc*/
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
  qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_methods, /*tp_methods*/
  qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_new, /*tp_new*/
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
static PyModuleDef qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    "qgd_N_Qubit_Decomposition_Wrapper",
    "Python binding for QGD N_Qubit_Decomposition class",
    -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_Generative_Quantum_Machine_Learning_Base_Wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Type);
    if (PyModule_AddObject(m, "qgd_Generative_Quantum_Machine_Learning_Base_Wrapper", (PyObject *) &qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Type) < 0) {
        Py_DECREF(&qgd_Generative_Quantum_Machine_Learning_Base_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}


} //extern C


