/*
Created on Fri Jun 26 14:42:56 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*
\file qgd_N_Qubit_Decomposition_Wrapper.cpp
\brief Python interface for the N_Qubit_Decomposition class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <stdio.h>

#include "N_Qubit_Decomposition.h"
#include "Sub_Matrix_Decomposition.h"


/**
@brief Type definition of the qgd_Operation_Block Python class of the qgd_Operation_Block module
*/
typedef struct qgd_Operation_Block {
    PyObject_HEAD
    Operation_block* gate;
} qgd_Operation_Block;


/**
@brief Type definition of the qgd_N_Qubit_Decomposition_Wrapper Python class of the qgd_N_Qubit_Decomposition_Wrapper module
*/
typedef struct qgd_N_Qubit_Decomposition_Wrapper {
    PyObject_HEAD
    /// pointer to the unitary to be decomposed to keep it alive
    PyObject *Umtx;
    /// An object to decompose the unitary
    N_Qubit_Decomposition<Sub_Matrix_Decomposition>* decomp;

} qgd_N_Qubit_Decomposition_Wrapper;



/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param Umtx An instance of class Matrix containing the unitary to be decomposed
@param qbit_num Number of qubits spanning the unitary
@param optimize_layer_num Logical value. Set true to optimize the number of decomposing layers during the decomposition procedure, or false otherwise.
@param initial_guess Type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
N_Qubit_Decomposition<Sub_Matrix_Decomposition>* 
create_N_Qubit_Decomposition( Matrix& Umtx, int qbit_num, bool optimize_layer_num, guess_type initial_guess ) {

    return new N_Qubit_Decomposition<Sub_Matrix_Decomposition>( Umtx, qbit_num, optimize_layer_num, initial_guess );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_N_Qubit_Decomposition( N_Qubit_Decomposition<Sub_Matrix_Decomposition>*  instance ) {
    delete instance;
    return;
}




extern "C"
{


/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition_Wrapper is destroyed
@param self A pointer pointing to an instance of class qgd_N_Qubit_Decomposition_Wrapper.
*/
static void
qgd_N_Qubit_Decomposition_Wrapper_dealloc(qgd_N_Qubit_Decomposition_Wrapper *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_N_Qubit_Decomposition( self->decomp );

    // release the unitary to be decomposed
    Py_DECREF(self->Umtx);    
    
    Py_TYPE(self)->tp_free((PyObject *) self);

}

/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_N_Qubit_Decomposition_Wrapper.
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_N_Qubit_Decomposition_Wrapper *self;
    self = (qgd_N_Qubit_Decomposition_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition_Wrapper is initialized
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper.
@param args A tuple of the input arguments: Umtx (numpy array), qbit_num (integer), optimize_layer_num (bool), initial_guess (string PyObject 
@param kwds A tuple of keywords
*/
static int
qgd_N_Qubit_Decomposition_Wrapper_init(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"Umtx", (char*)"qbit_num", (char*)"optimize_layer_num", (char*)"initial_guess", NULL};
 
    // initiate variables for input arguments
    PyObject *Umtx_arg = NULL;
    int  qbit_num = -1; 
    bool optimize_layer_num = false;
    PyObject *initial_guess = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OibO", kwlist,
                                     &Umtx_arg, &qbit_num, &optimize_layer_num, &initial_guess))
        return -1;

    // convert python object array to numpy C API array
    if ( Umtx_arg == NULL ) return -1;
    self->Umtx = PyArray_FROM_OTF(Umtx_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(self->Umtx) ) {
        std::cout << "Umtx is not memory contiguous" << std::endl;
    }

    // get the dimensions of the array self->Umtx
    int dim_num = PyArray_NDIM( self->Umtx );
    npy_intp* dims = PyArray_DIMS(self->Umtx);

    // insert a test for dimensions
    if (dim_num != 2) {
        std::cout << "The number of dimensions of the input matrix should be 2, but Umtx with " << dim_num << " dimensions was given" << std::endl; 
        return -1;
    }

    // get the pointer to the data stored in the matrix self->Umtx
    QGD_Complex16* data = (QGD_Complex16*)PyArray_DATA(self->Umtx);
 

    // create QGD version of the Umtx
    Matrix Umtx_mtx = Matrix(data, dims[0], dims[1]);    


    // determine the initial guess type
    PyObject* initial_guess_string = PyObject_Str(initial_guess);
    PyObject* initial_guess_string_unicode = PyUnicode_AsEncodedString(initial_guess_string, "utf-8", "~E~");
    const char* initial_guess_C = PyBytes_AS_STRING(initial_guess_string_unicode);

    guess_type qgd_initial_guess;
    if ( strcmp("zeros", initial_guess_C) == 0 or strcmp("ZEROS", initial_guess_C) == 0) {
        qgd_initial_guess = ZEROS;        
    }
    else if ( strcmp("random", initial_guess_C)==0 or strcmp("RANDOM", initial_guess_C)==0) {
        qgd_initial_guess = RANDOM;        
    }
    else if ( strcmp("close_to_zero", initial_guess_C)==0 or strcmp("CLOSE_TO_ZERO", initial_guess_C)==0) {
        qgd_initial_guess = CLOSE_TO_ZERO;        
    }
    else {
        std::cout << "Wring initial guess format. Using default ZEROS." << std::endl; 
        qgd_initial_guess = ZEROS;     
    }
   
    // create an instance of the class N_Qubit_Decomposition
    if (qbit_num > 0 ) {
        self->decomp =  create_N_Qubit_Decomposition( Umtx_mtx, qbit_num, optimize_layer_num, qgd_initial_guess);
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
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper.
@param args A tuple of the input arguments: finalize_decomp (bool), prepare_export (bool)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_Start_Decomposition(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"finalize_decomp", (char*)"prepare_export", NULL};

    // initiate variables for input arguments
    bool  finalize_decomp = true; 
    bool  prepare_export = true; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|bb", kwlist,
                                     &finalize_decomp, &prepare_export))
        return Py_BuildValue("i", -1);


    // starting the decomposition
    self->decomp->start_decomposition(finalize_decomp, prepare_export);


    return Py_BuildValue("i", 0);

}





/**
@brief Wrapper function to get the number of decomposing operations.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper.
@return Returns with the number of operations
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_operation_num( qgd_N_Qubit_Decomposition_Wrapper *self ) {

    // get the number of operations
    int ret = self->decomp->get_operation_num();


    return Py_BuildValue("i", ret);

}



/**
@brief Call to get the metadata organised into Python dictionary of the idx-th operation 
@param decomp A pointer pointing to an instance of the class N_Qubit_Decomposition.
@param idx Labels the idx-th decomposing operation.
@return Returns with a python dictionary containing the metadata of the idx-th operation
*/
static PyObject *
get_operation( N_Qubit_Decomposition<Sub_Matrix_Decomposition>* decomp, int &idx ) {


    // create dictionary conatining the gate data
    PyObject* gate = PyDict_New();

    Operation* operation = decomp->get_operation( idx );

    if (operation->get_type() == CNOT_OPERATION) {

        // create gate parameters
        PyObject* type = Py_BuildValue("s",  "CNOT" );
        PyObject* target_qbit = Py_BuildValue("i",  operation->get_target_qbit() );
        PyObject* control_qbit = Py_BuildValue("i",  operation->get_control_qbit() );


        PyDict_SetItemString(gate, "type", type );
        PyDict_SetItemString(gate, "target_qbit", target_qbit );
        PyDict_SetItemString(gate, "control_qbit", control_qbit );            

        Py_XDECREF(type);
        Py_XDECREF(target_qbit);
        Py_XDECREF(control_qbit);

    }
    else if (operation->get_type() == U3_OPERATION) {

        // get U3 parameters
        U3* u3_operation = static_cast<U3*>(operation);
        double* parameters = (double*)malloc(3*sizeof(double));
        u3_operation->get_optimized_parameters(parameters);
 

        // create gate parameters
        PyObject* type = Py_BuildValue("s",  "U3" );
        PyObject* target_qbit = Py_BuildValue("i",  operation->get_target_qbit() );
        PyObject* Theta = Py_BuildValue("f",  parameters[0] );
        PyObject* Phi = Py_BuildValue("f",  parameters[1] );
        PyObject* Lambda = Py_BuildValue("f",  parameters[2] );


        PyDict_SetItemString(gate, "type", type );
        PyDict_SetItemString(gate, "target_qbit", target_qbit );
        PyDict_SetItemString(gate, "Theta", Theta );
        PyDict_SetItemString(gate, "Phi", Phi );
        PyDict_SetItemString(gate, "Lambda", Lambda );

        Py_XDECREF(type);
        Py_XDECREF(target_qbit);
        Py_XDECREF(Theta);
        Py_XDECREF(Phi);
        Py_XDECREF(Lambda);


        free( parameters);
    }
    else {
  
    }

    return gate;

}



/**
@brief Wrapper function to set the number of identical successive blocks during the subdecomposition of the qbit-th qubit.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper.
@param args A tuple of the input arguments: idx (int)
idx: labels the idx-th operation.
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_operation( qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    int  idx; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|i", &idx )) return Py_BuildValue("i", -1);


    return get_operation( self->decomp, idx );


}







/**
@brief Wrapper function to set the number of identical successive blocks during the subdecomposition of the qbit-th qubit.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper.
@param args A tuple of the input arguments: qbit (bool), identical_blocks (bool)
qbit: The number of qubits for which the subdecomposition should contain identical_blocks successive identical blocks.
identical_blocks: Number of successive identical blocks in the decomposition.
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_get_operations( qgd_N_Qubit_Decomposition_Wrapper *self ) {


    // get the number of operations
    int op_num = self->decomp->get_operation_num();

    // preallocate Python tuple for the output
    PyObject* ret = PyTuple_New( (Py_ssize_t) op_num );



    // iterate over the operations to get the operation listű
    for (int idx = 0; idx < op_num; idx++ ) {

        // get metadata about the idx-th gate
        PyObject* gate = get_operation( self->decomp, idx );

        // adding gate information to the tuple
        PyTuple_SetItem( ret, (Py_ssize_t) idx, gate );

    }


    return ret;

}


/**
@brief Lists the operations decomposing the initial unitary. (These operations are the inverse operations of the operations bringing the intial matrix into unity.)
@param start_index The index of the first inverse operation
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_List_Operations( qgd_N_Qubit_Decomposition_Wrapper *self ) {

    self->decomp->list_operations( 0 );

    return Py_None;
}



/**
@brief Set the maximal number of layers used in the subdecomposition of the qbit-th qubit.
@param max_layer_num A dictionary {'n': max_layer_num} labeling the maximal number of the operation layers used in the subdecomposition.
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Max_Layer_Num(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args ) {

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
        assert(PyInt_Check(value) == 1);
        int value_int = (int) PyLong_AsLong(value);

        // convert keylue fron PyObject to int
        assert(PyInt_Check(key) == 1);
        int key_int = (int) PyLong_AsLong(key);

        // set maximal layer nums on the C++ side
        self->decomp->set_max_layer_num( key_int, value_int );
    }

    return Py_BuildValue("i", 0);
}




/**
@brief Set the number of identical successive blocks during the subdecomposition of the qbit-th qubit.
@param identical_blocks A dictionary {'n': identical_blocks} labeling the number of successive identical layers used in the subdecomposition at the disentangling of the n-th qubit.
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Identical_Blocks(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    PyObject* identical_blocks; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &identical_blocks )) return Py_BuildValue("i", -1);

    // Check whether input is dictionary
    if (!PyDict_Check(identical_blocks)) {
        printf("Input must be dictionary!\n");
        return Py_BuildValue("i", -1);
    }


    PyObject* key = NULL;
    PyObject* value = NULL;
    Py_ssize_t pos = 0;


    while (PyDict_Next(identical_blocks, &pos, &key, &value)) {

        // convert value fron PyObject to int
        assert(PyInt_Check(value) == 1);
        int value_int = (int) PyLong_AsLong(value);

        // convert keylue fron PyObject to int
        assert(PyInt_Check(key) == 1);
        int key_int = (int) PyLong_AsLong(key);

        // set maximal layer nums on the C++ side
        self->decomp->set_identical_blocks( key_int, value_int );
    }

    return Py_BuildValue("i", 0);
}





/**
@brief Set the number of iteration loops during the subdecomposition of the qbit-th qubit.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper.
@param args A tuple of the input arguments: identical_blocks (PyDict)
identical_blocks: A dictionary {'n': iteration_loops} labeling the number of successive identical layers used in the subdecomposition at the disentangling of the n-th qubit.
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Iteration_Loops(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args ) {

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
        assert(PyInt_Check(value) == 1);
        int value_int = (int) PyLong_AsLong(value);

        // convert keylue fron PyObject to int
        assert(PyInt_Check(key) == 1);
        int key_int = (int) PyLong_AsLong(key);

        // set maximal layer nums on the C++ side
        self->decomp->set_iteration_loops( key_int, value_int );
    }

    return Py_BuildValue("i", 0);
}



/**
@brief Set the verbosity of the N_Qubit_Decomposition class
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper.
@param args A tuple of the input arguments: verbose (bool)
verbose: Set False to suppress the output messages of the decompostion, or True (deafult) otherwise.
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Verbose(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    bool verbose; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|b", &verbose )) return Py_BuildValue("i", -1);


    // set maximal layer nums on the C++ side
    self->decomp->set_verbose( verbose );

    return Py_BuildValue("i", 0);
}



/**
@brief Wrapper method to set the optimization tolerance of the optimization process during the decomposition. The final error of the decomposition would scale with the square root of this value.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper.
@param args A tuple of the input arguments: tolerance (double)
tolerance: Set False to suppress the output messages of the decompostion, or True (deafult) otherwise.
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Optimization_Tolerance(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    double tolerance; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|d", &tolerance )) return Py_BuildValue("i", -1);


    // set maximal layer nums on the C++ side
    self->decomp->set_optimization_tolerance( tolerance );

    return Py_BuildValue("i", 0);
}



/**
@brief Wrapper function to set custom gate structure for the decomposition.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_Wrapper.
@param args A tuple of the input arguments: gate_structure_dict (PyDict)
gate_structure_dict: ?????????????????????????????
@return Returns with zero on success.
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_set_Gate_Structure( qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args ) {


    // initiate variables for input arguments
    PyObject* gate_structure_dict; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &gate_structure_dict )) return Py_BuildValue("i", -1);

    // Check whether input is dictionary
    if (!PyDict_Check(gate_structure_dict)) {
        printf("Input must be dictionary!\n");
        return Py_BuildValue("i", -1);
    }


    PyObject* key = NULL;
    PyObject* value = NULL;
    Py_ssize_t pos = 0;

    std::map< int, Operation_block* > gate_structure;


    while (PyDict_Next(gate_structure_dict, &pos, &key, &value)) {

        // convert keylue from PyObject to int
        assert(PyInt_Check(key) == 1);
        int key_int = (int) PyLong_AsLong(key);

        // convert keylue from PyObject to qgd_Operation_Block
        qgd_Operation_Block* qgd_op_block = (qgd_Operation_Block*) value;

        gate_structure.insert( std::pair<int, Operation_block*>( key_int, qgd_op_block->gate ));

    }

    self->decomp->set_custom_gate_structure( gate_structure );

    return Py_BuildValue("i", 0);


}


/**
@brief Wrapper method to reorder the qubits in the decomposition class.
@param 
*/
static PyObject *
qgd_N_Qubit_Decomposition_Wrapper_Reorder_Qubits(qgd_N_Qubit_Decomposition_Wrapper *self, PyObject *args ) {

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
@brief Structure containing metadata about the members of class qgd_N_Qubit_Decomposition_Wrapper.
*/
static PyMemberDef qgd_N_Qubit_Decomposition_Wrapper_members[] = {
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the methods of class qgd_N_Qubit_Decomposition_Wrapper.
*/
static PyMethodDef qgd_N_Qubit_Decomposition_Wrapper_methods[] = {
    {"Start_Decomposition", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_Start_Decomposition, METH_VARARGS | METH_KEYWORDS,
     "Method to start the decomposition."
    },
    {"get_Operation_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_operation_num, METH_NOARGS,
     "Method to get the number of decomposing operations."
    },
    {"get_Operation", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_operation, METH_VARARGS,
     "Method to get the i-th decomposing operations."
    },
    {"get_Operations", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_get_operations, METH_NOARGS,
     "Method to get the tuple of decomposing operations."
    },
    {"List_Operations", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_List_Operations, METH_NOARGS,
     "Call to print the decomposing nitaries on standard output"
    },
    {"set_Max_Layer_Num", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Max_Layer_Num, METH_VARARGS,
     "Call to set the maximal number of layers used in the subdecomposition of the qbit-th qubit."
    },
    {"set_Identical_Blocks", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Identical_Blocks, METH_VARARGS,
     "Call to set the number of identical successive blocks during the subdecomposition of the qbit-th qubit."
    },
    {"set_Iteration_Loops", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Iteration_Loops, METH_VARARGS,
     "Call to set the number of iteration loops during the subdecomposition of the qbit-th qubit."
    },
    {"set_Verbose", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Verbose, METH_VARARGS,
     "Call to set the verbosity of the qgd_N_Qubit_Decomposition class."
    },
    {"set_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Gate_Structure, METH_VARARGS,
     "Call to set custom gate structure in the decomposition."
    },
    {"Reorder_Qubits", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_Reorder_Qubits, METH_VARARGS,
     "Wrapper method to reorder the qubits in the decomposition class."
    },
    {"set_Optimization_Tolerance", (PyCFunction) qgd_N_Qubit_Decomposition_Wrapper_set_Optimization_Tolerance, METH_VARARGS,
     "Wrapper method to set the optimization tolerance of the optimization process during the decomposition. The final error of the decomposition would scale with the square root of this value."
    },
    {NULL}  /* Sentinel */
};

/**
@brief A structure describing the type of the class qgd_N_Qubit_Decomposition_Wrapper.
*/
static PyTypeObject qgd_N_Qubit_Decomposition_Wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_N_Qubit_Decomposition_Wrapper.qgd_N_Qubit_Decomposition_Wrapper", /*tp_name*/
  sizeof(qgd_N_Qubit_Decomposition_Wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_N_Qubit_Decomposition_Wrapper_dealloc, /*tp_dealloc*/
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
  "Object to represent a Operation_block class of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  qgd_N_Qubit_Decomposition_Wrapper_methods, /*tp_methods*/
  qgd_N_Qubit_Decomposition_Wrapper_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_N_Qubit_Decomposition_Wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_N_Qubit_Decomposition_Wrapper_new, /*tp_new*/
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
static PyModuleDef qgd_N_Qubit_Decomposition_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "qgd_N_Qubit_Decomposition_Wrapper",
    .m_doc = "Python binding for QGD N_Qubit_Decomposition class",
    .m_size = -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_N_Qubit_Decomposition_Wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&qgd_N_Qubit_Decomposition_Wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&qgd_N_Qubit_Decomposition_Wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&qgd_N_Qubit_Decomposition_Wrapper_Type);
    if (PyModule_AddObject(m, "qgd_N_Qubit_Decomposition_Wrapper", (PyObject *) &qgd_N_Qubit_Decomposition_Wrapper_Type) < 0) {
        Py_DECREF(&qgd_N_Qubit_Decomposition_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}


} //extern C

