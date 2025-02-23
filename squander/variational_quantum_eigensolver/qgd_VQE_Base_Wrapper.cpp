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


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <stdio.h>
#include "Variational_Quantum_Eigensolver_Base.h"

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
typedef struct qgd_Variational_Quantum_Eigensolver_Base_Wrapper {
    PyObject_HEAD
    /// pointer to the unitary to be decomposed to keep it alive
    PyObject *Hamiltonian;
    /// An object to decompose the unitary
    Variational_Quantum_Eigensolver_Base* vqe;

} qgd_Variational_Quantum_Eigensolver_Base_Wrapper;



/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param Umtx An instance of class Matrix containing the unitary to be decomposed
@param qbit_num Number of qubits spanning the unitary
@param optimize_layer_num Logical value. Set true to optimize the number of decomposing layers during the decomposition procedure, or false otherwise.
@param initial_guess Type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
Variational_Quantum_Eigensolver_Base* 
create_qgd_Variational_Quantum_Eigensolver_Base( Matrix_sparse Hamiltonian, int qbit_num, std::map<std::string, Config_Element>& config) {

    return new Variational_Quantum_Eigensolver_Base( Hamiltonian, qbit_num, config);
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_Variational_Quantum_Eigensolver_Base( Variational_Quantum_Eigensolver_Base*  instance ) {

    if (instance != NULL ) {
        delete instance;
    }
    return;
}



extern "C"
{


/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition_Wrapper is destroyed
@param self A pointer pointing to an instance of class qgd_N_Qubit_Decomposition_Wrapper.
*/
static void
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_dealloc(qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self)
{

    if ( self->vqe != NULL ) {
        // deallocate the instance of class N_Qubit_Decomposition
        release_Variational_Quantum_Eigensolver_Base( self->vqe );
        self->vqe = NULL;
    }

    if ( self->Hamiltonian != NULL ) {
        // release the unitary to be decomposed
        Py_DECREF(self->Hamiltonian);    
        self->Hamiltonian = NULL;
    }
    
    Py_TYPE(self)->tp_free((PyObject *) self);

}

/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_N_Qubit_Decomposition_Wrapper.
*/
static PyObject *
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self;
    self = (qgd_Variational_Quantum_Eigensolver_Base_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->vqe = NULL;
    self->Hamiltonian = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition_Wrapper is initialized
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper.
@param args A tuple of the input arguments: Umtx (numpy array), qbit_num (integer), optimize_layer_num (bool), initial_guess (string PyObject 
@param kwds A tuple of keywords
*/
static int
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_init(qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"Hamiltonian_data", (char*)"Hamiltonian_indices", (char*)"Hamiltonian_indptr", (char*)"qbit_num", (char*)"config", NULL};
 
    // initiate variables for input arguments
    PyArrayObject *Hamiltonian_data_arg = NULL;
    PyArrayObject *Hamiltonian_indices_arg = NULL;
    PyArrayObject *Hamiltonian_indptr_arg = NULL;
    int  qbit_num = -1; 
    PyObject *config_arg = NULL;
    
    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOiO", kwlist,
                                   &Hamiltonian_data_arg, &Hamiltonian_indices_arg, &Hamiltonian_indptr_arg, &qbit_num, &config_arg))
        return -1;
    
    int shape = Power_of_2(qbit_num);
    // convert python object array to numpy C API array
    if ( Hamiltonian_data_arg == NULL ) return -1;
    
    Hamiltonian_data_arg            = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)Hamiltonian_data_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    QGD_Complex16* Hamiltonian_data = (QGD_Complex16*)PyArray_DATA(Hamiltonian_data_arg);
    int NNZ                         = PyArray_DIMS(Hamiltonian_data_arg)[0];
    
    if ( Hamiltonian_indices_arg == NULL ) return -1;
    
    Hamiltonian_indices_arg  = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)Hamiltonian_indices_arg, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    int* Hamiltonian_indices = (int*)PyArray_DATA(Hamiltonian_indices_arg);
    
    if ( Hamiltonian_indptr_arg == NULL ) return -1;
    
    Hamiltonian_indptr_arg  = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)Hamiltonian_indptr_arg, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    int* Hamiltonian_indptr = (int*)PyArray_DATA(Hamiltonian_indptr_arg);
    
    Matrix_sparse Hamiltonian_mtx = Matrix_sparse(Hamiltonian_data, shape, shape, NNZ, Hamiltonian_indices, Hamiltonian_indptr);

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
        self->vqe =  create_qgd_Variational_Quantum_Eigensolver_Base(Hamiltonian_mtx, qbit_num, config);
    }
    else {
        std::cout << "The number of qubits should be given as a positive integer, " << qbit_num << "  was given" << std::endl;
        return -1;
    }


    return 0;
}



/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_get_Optimized_Parameters( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self ) {

    int parameter_num = self->vqe->get_parameter_num();

    Matrix_real parameters_mtx(1, parameter_num);
    double* parameters = parameters_mtx.get_data();
    self->vqe->get_optimized_parameters(parameters);


    // convert to numpy array
    parameters_mtx.set_owner(false);
    PyObject * parameter_arr = matrix_real_to_numpy( parameters_mtx );

    return parameter_arr;
}


static PyObject *
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Start_Optimization(qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self)
{

    // starting the decomposition
    try {
        self->vqe->start_optimization();
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
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_get_Qbit_Num(qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self ) {

    int qbit_num = 0;

    try {
        qbit_num = self->vqe->get_qbit_num();
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
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Optimized_Parameters( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args ) {

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
        self->vqe->set_optimized_parameters(parameters_mtx.get_data(), parameters_mtx.size());
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
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Initial_State( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args ) {

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
        self->vqe->set_initial_state( initial_state_mtx );
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
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Gate_Structure_From_Binary(  qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args ) {



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
        self->vqe->set_gate_structure( filename_str );
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
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Optimization_Tolerance(qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    double tolerance;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|d", &tolerance )) return Py_BuildValue("i", -1);


    // set maximal layer nums on the C++ side
    self->vqe->set_optimization_tolerance( tolerance );


    return Py_BuildValue("i", 0);
}



static PyObject *
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_apply_to( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args ) {

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


    self->vqe->apply_to( parameters_mtx, unitary_mtx );
    
    if (unitary_mtx.data != PyArray_DATA(unitary)) {
        memcpy(PyArray_DATA(unitary), unitary_mtx.data, unitary_mtx.size() * sizeof(QGD_Complex16));
    }

    Py_DECREF(parameters_arr);
    Py_DECREF(unitary);

    return Py_BuildValue("i", 0);
}



static PyObject *
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Optimizer( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args, PyObject *kwds)
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
        self->vqe->set_optimizer(qgd_optimizer);
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
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Ansatz( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args, PyObject *kwds)
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
    else {
        std::cout << "Wrong ansatz. Using default: HEA" << std::endl; 
        qgd_ansatz = HEA;     
    }
    
    
    try {
        self->vqe->set_ansatz(qgd_ansatz);
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
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_get_Second_Renyi_Entropy( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args)
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
        entropy = self->vqe->get_second_Renyi_entropy( parameters_mtx, input_state_mtx, qbit_list_mtx );
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
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Generate_Circuit( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    int layers;
    int inner_blocks;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|ii", &layers, &inner_blocks )) return Py_BuildValue("i", -1);


    try {
        self->vqe->generate_circuit( layers, inner_blocks );
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
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Optimization_Problem( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args)
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
        f0 = self->vqe->optimization_problem(parameters_mtx );
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
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_get_Parameter_Num( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self ) {

    int parameter_num = self->vqe->get_parameter_num();

    return Py_BuildValue("i", parameter_num);
}





/**
@brief Wrapper function to retrieve the circuit (Squander format) incorporated in the instance.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
*/
static PyObject *
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_get_circuit( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self ) {


    PyObject* qgd_Circuit  = PyImport_ImportModule("squander.gates.qgd_Circuit");

    if ( qgd_Circuit == NULL ) {
        PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_Circuit" );
        return NULL;
    }

    // retrieve the C++ variant of the flat circuit (flat circuit does not conatain any sub-circuits)
    Gates_block* circuit = self->vqe->get_flat_circuit();



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
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Project_Name( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args ) {
    // initiate variables for input arguments
    PyObject* project_name_new=NULL; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &project_name_new)) return Py_BuildValue("i", -1);
    
    
    PyObject* project_name_new_string = PyObject_Str(project_name_new);
    PyObject* project_name_new_unicode = PyUnicode_AsEncodedString(project_name_new_string, "utf-8", "~E~");
    const char* project_name_new_C = PyBytes_AS_STRING(project_name_new_unicode);
    std::string project_name_new_str = ( project_name_new_C );
    
    // convert to python string
    self->vqe->set_project_name(project_name_new_str);

   return Py_BuildValue("i", 0);
}


/**
@brief Wrapper function to set custom gate structure for the decomposition.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@return Returns with zero on success.
*/
static PyObject *
qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Gate_Structure( qgd_Variational_Quantum_Eigensolver_Base_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    PyObject* gate_structure_py; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &gate_structure_py )) return Py_BuildValue("i", -1);


    // convert gate structure from PyObject to qgd_Circuit_Wrapper
    qgd_Circuit_Wrapper* qgd_op_block = (qgd_Circuit_Wrapper*) gate_structure_py;

    try {
        self->vqe->set_custom_gate_structure( qgd_op_block->gate );
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
static PyMemberDef qgd_Variational_Quantum_Eigensolver_Base_Wrapper_members[] = {
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the methods of class qgd_N_Qubit_Decomposition_Wrapper.
*/
static PyMethodDef qgd_Variational_Quantum_Eigensolver_Base_Wrapper_methods[] = {
    {"Start_Optimization", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Start_Optimization, METH_NOARGS,
     "Method to start the decomposition."
    },
    {"get_Optimized_Parameters", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_get_Optimized_Parameters, METH_NOARGS,
     "Method to get the array of optimized parameters."
    },
    {"set_Optimized_Parameters", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Optimized_Parameters, METH_VARARGS,
     "Method to set the array of optimized parameters."
    },
    {"set_Optimization_Tolerance", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Optimization_Tolerance, METH_VARARGS,
    "Method to set optimization tolerance"
    },
    {"get_Circuit", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_get_circuit, METH_NOARGS,
     "Method to get the incorporated circuit."
    },
    {"set_Project_Name", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Project_Name, METH_VARARGS,
    "method to set project name."
    },
    {"set_Gate_Structure_From_Binary", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Gate_Structure_From_Binary, METH_VARARGS,
     "Method to set the gate structure from a file created in SQUANDER."
    },
    {"apply_to", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_apply_to, METH_VARARGS,
     "Call to apply the gate on the input matrix."
    },
    {"set_Optimizer", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Optimizer, METH_VARARGS | METH_KEYWORDS,
     "Method to set optimizer."
    },
    {"set_Ansatz", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Ansatz, METH_VARARGS | METH_KEYWORDS,
     "Method to set ansatz type."
    },
    {"get_Parameter_Num", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_get_Parameter_Num, METH_NOARGS,
     "Call to get the number of free parameters in the gate structure used for the decomposition"
    },
    {"Generate_Circuit", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Generate_Circuit, METH_VARARGS,
     "Method to set the circuit based on the ansatz type."
    },
    {"Optimization_Problem", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Optimization_Problem, METH_VARARGS,
     "Method to get the expected energy of the circuit at parameters."
    },
    {"get_Second_Renyi_Entropy", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_get_Second_Renyi_Entropy, METH_VARARGS,
     "Wrapper function to evaluate the second Rényi entropy of a quantum circuit at a specific parameter set."
    },
    {"get_Qbit_Num", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_get_Qbit_Num, METH_NOARGS,
     "Call to get the number of qubits in the circuit"
    },
    {"set_Initial_State", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Initial_State, METH_VARARGS,
     "Call to set the initial state used in the VQE process."
    },
    {"set_Gate_Structure", (PyCFunction) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_set_Gate_Structure, METH_VARARGS,
     "Call to set custom gate structure for VQE experiments."
    },
    {NULL}  /* Sentinel */
};

/**
@brief A structure describing the type of the class qgd_N_Qubit_Decomposition_Wrapper.
*/
static PyTypeObject qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_N_Qubit_Decomposition_Wrapper.qgd_N_Qubit_Decomposition_Wrapper", /*tp_name*/
  sizeof(qgd_Variational_Quantum_Eigensolver_Base_Wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_dealloc, /*tp_dealloc*/
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
  qgd_Variational_Quantum_Eigensolver_Base_Wrapper_methods, /*tp_methods*/
  qgd_Variational_Quantum_Eigensolver_Base_Wrapper_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_Variational_Quantum_Eigensolver_Base_Wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_Variational_Quantum_Eigensolver_Base_Wrapper_new, /*tp_new*/
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
static PyModuleDef qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    "qgd_N_Qubit_Decomposition_Wrapper",
    "Python binding for QGD N_Qubit_Decomposition class",
    -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_Variational_Quantum_Eigensolver_Base_Wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Type);
    if (PyModule_AddObject(m, "qgd_Variational_Quantum_Eigensolver_Base_Wrapper", (PyObject *) &qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Type) < 0) {
        Py_DECREF(&qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}


} //extern C


