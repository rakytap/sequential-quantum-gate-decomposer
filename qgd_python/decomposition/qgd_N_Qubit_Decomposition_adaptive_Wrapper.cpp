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
\file qgd_N_Qubit_Decomposition_adaptive_Wrapper.cpp
\brief Python interface for the N_Qubit_Decomposition class
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <stdio.h>
#include "N_Qubit_Decomposition_adaptive.h"
#include "N_Qubit_Decomposition_adaptive_general.h"
#include "Gates_block.h"

#include "numpy_interface.h"




/**
@brief Type definition of the qgd_gates_Block Python class of the qgd_Gates_Block module
*/
typedef struct qgd_Gates_Block {
    PyObject_HEAD
    Gates_block* gate;
} qgd_Gates_Block;


/**
@brief Type definition of the qgd_N_Qubit_Decomposition_adaptive_Wrapper Python class of the qgd_N_Qubit_Decomposition_adaptive_Wrapper module
*/
typedef struct qgd_N_Qubit_Decomposition_adaptive_Wrapper {
    PyObject_HEAD
    /// pointer to the unitary to be decomposed to keep it alive
    PyObject *Umtx;
    /// An object to decompose the unitary
    N_Qubit_Decomposition_adaptive* decomp;
    /// An object to decompose the unitary
    N_Qubit_Decomposition_adaptive_general* decomp_general;
    /// a pointer to base class represented by the decomposing classes
    N_Qubit_Decomposition_Base* decomp_base;

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
create_N_Qubit_Decomposition_adaptive( Matrix& Umtx, int qbit_num, int level_limit, int level_limit_min, std::vector<matrix_base<int>> topology_in ) {

    return new N_Qubit_Decomposition_adaptive( Umtx, qbit_num, level_limit, level_limit_min, topology_in );
}



/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param Umtx An instance of class Matrix containing the unitary to be decomposed
@param qbit_num Number of qubits spanning the unitary
@param level_limit The maximal number of layers used in the decomposition
@param initial_guess Type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
N_Qubit_Decomposition_adaptive_general* 
create_N_Qubit_Decomposition_adaptive_general( Matrix& Umtx, int qbit_num, int level_limit, int level_limit_min, std::vector<matrix_base<int>> topology_in ) {

    return new N_Qubit_Decomposition_adaptive_general( Umtx, qbit_num, level_limit, level_limit_min, topology_in );
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



/**
@brief Call to deallocate an instance of N_Qubit_Decomposition_adaptive_general class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_N_Qubit_Decomposition_adaptive_general( N_Qubit_Decomposition_adaptive_general*  instance ) {

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

    if ( self->decomp_general != NULL ) {
        // deallocate the instance of class N_Qubit_Decomposition
        release_N_Qubit_Decomposition_adaptive_general( self->decomp_general );
        self->decomp_general = NULL;
    }

    self->decomp_base = NULL;

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
    if (self != NULL) {}

    self->decomp = NULL;
    self->decomp_general = NULL;
    self->Umtx = NULL;

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
    static char *kwlist[] = {(char*)"Umtx", (char*)"qbit_num", (char*)"level_limit", (char*)"level_limit_min", (char*)"method", (char*)"topology", NULL};
 
    // initiate variables for input arguments
    PyObject *Umtx_arg = NULL;
    int  qbit_num = -1; 
    int level_limit = 0;
    int level_limit_min = 0;
    PyObject *method = NULL;
    PyObject *topology = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OiiiOO", kwlist,
                                     &Umtx_arg, &qbit_num, &level_limit, &level_limit_min, &method, &topology))
        return -1;

    // convert python object array to numpy C API array
    if ( Umtx_arg == NULL ) return -1;
    self->Umtx = PyArray_FROM_OTF(Umtx_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(self->Umtx) ) {
        std::cout << "Umtx is not memory contiguous" << std::endl;
    }


    // create QGD version of the Umtx
    Matrix Umtx_mtx = numpy2matrix(self->Umtx);


    // determine the optimizaton method
    PyObject* method_string = PyObject_Str(method);
    PyObject* method_string_unicode = PyUnicode_AsEncodedString(method_string, "utf-8", "~E~");
    const char* method_C = PyBytes_AS_STRING(method_string_unicode);

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


    // create an instance of the class N_Qubit_Decomposition
    if (qbit_num > 0 ) {
        if ( strcmp("limited", method_C)==0 or strcmp("LIMITED", method_C)==0) {
            self->decomp = create_N_Qubit_Decomposition_adaptive( Umtx_mtx, qbit_num, level_limit, level_limit_min, topology_Cpp);
            self->decomp_base = (N_Qubit_Decomposition_Base*)self->decomp;
        }
        else if ( strcmp("general", method_C)==0 or strcmp("GENERAL", method_C)==0) {
            self->decomp_general = create_N_Qubit_Decomposition_adaptive_general( Umtx_mtx, qbit_num, level_limit, level_limit_min, topology_Cpp);    
            self->decomp_base = (N_Qubit_Decomposition_Base*)self->decomp_general;
        }
        else {
            std::cout << "Wrong optmimization method. Falling back to limited." << std::endl;
            self->decomp = create_N_Qubit_Decomposition_adaptive( Umtx_mtx, qbit_num, level_limit, level_limit_min, topology_Cpp );
            self->decomp_base = (N_Qubit_Decomposition_Base*)self->decomp;
        }
    }
    else {
        std::cout << "The number of qubits should be given as a positive integer, " << qbit_num << "  was given" << std::endl;
        Py_XDECREF(method_string);
        Py_XDECREF(method_string_unicode);
        return -1;
    }



    Py_XDECREF(method_string);
    Py_XDECREF(method_string_unicode);

    return 0;
}

/**
@brief Wrapper function to call the start_decomposition method of C++ class N_Qubit_Decomposition
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param args A tuple of the input arguments: finalize_decomp (bool), prepare_export (bool)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_Start_Decomposition(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"prepare_export", NULL};

    // initiate variables for input arguments
    bool  prepare_export = true; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|b", kwlist,
                                     &prepare_export))
        return Py_BuildValue("i", -1);

    // starting the decomposition
    if (  self->decomp != NULL ) {
        try {
            self->decomp->start_decomposition(prepare_export);
        }
        catch (std::string err) {
            PyErr_SetString(PyExc_Exception, err.c_str());
            std::cout << err << std::endl;
            return NULL;
        }
    }
    else if(  self->decomp_general != NULL ) {
        self->decomp_general->start_decomposition(prepare_export);
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
    int ret = self->decomp_base->get_gate_num();


    return Py_BuildValue("i", ret);

}



/**
@brief Call to get the metadata organised into Python dictionary of the idx-th gate
@param decomp A pointer pointing to an instance of the class N_Qubit_Decomposition.
@param idx Labels the idx-th decomposing gate.
@return Returns with a python dictionary containing the metadata of the idx-th gate
*/
static PyObject *
get_gate( N_Qubit_Decomposition_Base* decomp, int &idx ) {


    // create dictionary conatining the gate data
    PyObject* py_gate = PyDict_New();

    Gate* gate = decomp->get_gate( idx );

    if (gate->get_type() == CNOT_OPERATION) {

        // create gate parameters
        PyObject* type = Py_BuildValue("s",  "CNOT" );
        PyObject* target_qbit = Py_BuildValue("i",  gate->get_target_qbit() );
        PyObject* control_qbit = Py_BuildValue("i",  gate->get_control_qbit() );


        PyDict_SetItemString(py_gate, "type", type );
        PyDict_SetItemString(py_gate, "target_qbit", target_qbit );
        PyDict_SetItemString(py_gate, "control_qbit", control_qbit );            

        Py_XDECREF(type);
        Py_XDECREF(target_qbit);
        Py_XDECREF(control_qbit);

    }
    else if (gate->get_type() == CZ_OPERATION) {

        // create gate parameters
        PyObject* type = Py_BuildValue("s",  "CZ" );
        PyObject* target_qbit = Py_BuildValue("i",  gate->get_target_qbit() );
        PyObject* control_qbit = Py_BuildValue("i",  gate->get_control_qbit() );


        PyDict_SetItemString(py_gate, "type", type );
        PyDict_SetItemString(py_gate, "target_qbit", target_qbit );
        PyDict_SetItemString(py_gate, "control_qbit", control_qbit );            

        Py_XDECREF(type);
        Py_XDECREF(target_qbit);
        Py_XDECREF(control_qbit);

    }
    else if (gate->get_type() == CH_OPERATION) {

        // create gate parameters
        PyObject* type = Py_BuildValue("s",  "CH" );
        PyObject* target_qbit = Py_BuildValue("i",  gate->get_target_qbit() );
        PyObject* control_qbit = Py_BuildValue("i",  gate->get_control_qbit() );


        PyDict_SetItemString(py_gate, "type", type );
        PyDict_SetItemString(py_gate, "target_qbit", target_qbit );
        PyDict_SetItemString(py_gate, "control_qbit", control_qbit );            

        Py_XDECREF(type);
        Py_XDECREF(target_qbit);
        Py_XDECREF(control_qbit);

    }
    else if (gate->get_type() == SYC_OPERATION) {

        // create gate parameters
        PyObject* type = Py_BuildValue("s",  "SYC" );
        PyObject* target_qbit = Py_BuildValue("i",  gate->get_target_qbit() );
        PyObject* control_qbit = Py_BuildValue("i",  gate->get_control_qbit() );


        PyDict_SetItemString(py_gate, "type", type );
        PyDict_SetItemString(py_gate, "target_qbit", target_qbit );
        PyDict_SetItemString(py_gate, "control_qbit", control_qbit );            

        Py_XDECREF(type);
        Py_XDECREF(target_qbit);
        Py_XDECREF(control_qbit);

    }
    else if (gate->get_type() == U3_OPERATION) {

        // get U3 parameters
        U3* u3_gate = static_cast<U3*>(gate);
        Matrix_real&& parameters = u3_gate->get_optimized_parameters();

        // create gate parameters
        PyObject* type = Py_BuildValue("s",  "U3" );
        PyObject* target_qbit = Py_BuildValue("i",  gate->get_target_qbit() );
        PyObject* Theta = Py_BuildValue("f",  parameters[0] );
        PyObject* Phi = Py_BuildValue("f",  parameters[1] );
        PyObject* Lambda = Py_BuildValue("f",  parameters[2] );


        PyDict_SetItemString(py_gate, "type", type );
        PyDict_SetItemString(py_gate, "target_qbit", target_qbit );
        PyDict_SetItemString(py_gate, "Theta", Theta );
        PyDict_SetItemString(py_gate, "Phi", Phi );
        PyDict_SetItemString(py_gate, "Lambda", Lambda );

        Py_XDECREF(type);
        Py_XDECREF(target_qbit);
        Py_XDECREF(Theta);
        Py_XDECREF(Phi);
        Py_XDECREF(Lambda);


    }
    else if (gate->get_type() == RX_OPERATION) {

        // get U3 parameters
        RX* rx_gate = static_cast<RX*>(gate);
        Matrix_real&& parameters = rx_gate->get_optimized_parameters();
 

        // create gate parameters
        PyObject* type = Py_BuildValue("s",  "RX" );
        PyObject* target_qbit = Py_BuildValue("i",  gate->get_target_qbit() );
        PyObject* Theta = Py_BuildValue("f",  parameters[0] );


        PyDict_SetItemString(py_gate, "type", type );
        PyDict_SetItemString(py_gate, "target_qbit", target_qbit );
        PyDict_SetItemString(py_gate, "Theta", Theta );

        Py_XDECREF(type);
        Py_XDECREF(target_qbit);
        Py_XDECREF(Theta);

    }
    else if (gate->get_type() == RY_OPERATION) {

        // get U3 parameters
        RY* ry_gate = static_cast<RY*>(gate);
        Matrix_real&& parameters = ry_gate->get_optimized_parameters();
 

        // create gate parameters
        PyObject* type = Py_BuildValue("s",  "RY" );
        PyObject* target_qbit = Py_BuildValue("i",  gate->get_target_qbit() );
        PyObject* Theta = Py_BuildValue("f",  parameters[0] );


        PyDict_SetItemString(py_gate, "type", type );
        PyDict_SetItemString(py_gate, "target_qbit", target_qbit );
        PyDict_SetItemString(py_gate, "Theta", Theta );

        Py_XDECREF(type);
        Py_XDECREF(target_qbit);
        Py_XDECREF(Theta);

    }
    else if (gate->get_type() == RZ_OPERATION) {

        // get U3 parameters
        RZ* rz_gate = static_cast<RZ*>(gate);
        Matrix_real&& parameters = rz_gate->get_optimized_parameters();
 

        // create gate parameters
        PyObject* type = Py_BuildValue("s",  "RZ" );
        PyObject* target_qbit = Py_BuildValue("i",  gate->get_target_qbit() );
        PyObject* Phi = Py_BuildValue("f",  parameters[0] );


        PyDict_SetItemString(py_gate, "type", type );
        PyDict_SetItemString(py_gate, "target_qbit", target_qbit );
        PyDict_SetItemString(py_gate, "Phi", Phi );

        Py_XDECREF(type);
        Py_XDECREF(target_qbit);
        Py_XDECREF(Phi);


    }
    else if (gate->get_type() == X_OPERATION) {

        // create gate parameters
        PyObject* type = Py_BuildValue("s",  "X" );
        PyObject* target_qbit = Py_BuildValue("i",  gate->get_target_qbit() );

        PyDict_SetItemString(py_gate, "type", type );
        PyDict_SetItemString(py_gate, "target_qbit", target_qbit );

        Py_XDECREF(type);
        Py_XDECREF(target_qbit);

    }
    else if (gate->get_type() == SX_OPERATION) {

        // create gate parameters
        PyObject* type = Py_BuildValue("s",  "SX" );
        PyObject* target_qbit = Py_BuildValue("i",  gate->get_target_qbit() );

        PyDict_SetItemString(py_gate, "type", type );
        PyDict_SetItemString(py_gate, "target_qbit", target_qbit );

        Py_XDECREF(type);
        Py_XDECREF(target_qbit);

    }
    else {
  
    }

    return py_gate;

}



/**
@brief Wrapper function to set the number of identical successive blocks during the subdecomposition of the qbit-th qubit.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param args A tuple of the input arguments: idx (int)
idx: labels the idx-th gate.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_gate( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    int  idx; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|i", &idx )) return Py_BuildValue("i", -1);


    return get_gate( self->decomp_base, idx );


}







/**
@brief Wrapper function to set the number of identical successive blocks during the subdecomposition of the qbit-th qubit.
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param args A tuple of the input arguments: qbit (bool), identical_blocks (bool)
qbit: The number of qubits for which the subdecomposition should contain identical_blocks successive identical blocks.
identical_blocks: Number of successive identical blocks in the decomposition.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_gates( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {


    // get the number of gates
    int op_num = self->decomp_base->get_gate_num();

    // preallocate Python tuple for the output
    PyObject* ret = PyTuple_New( (Py_ssize_t) op_num );



    // iterate over the gates to get the gate list
    for (int idx = 0; idx < op_num; idx++ ) {

        // get metadata about the idx-th gate
        PyObject* gate = get_gate( self->decomp_base, idx );

        // adding gate information to the tuple
        PyTuple_SetItem( ret, (Py_ssize_t) idx, gate );

    }


    return ret;

}

/**
@brief returns the angle of the global phase (the radius us always sqrt(2))
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Global_Phase(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

    QGD_Complex16 global_phase_C = self->decomp_base->get_global_phase();
    PyObject* global_phase = PyFloat_FromDouble( std::atan2(global_phase_C.imag,global_phase_C.real));
    return global_phase;
    
}

/**
@brief sets the global phase to the new angle given
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
@param arg Global_phase_new_angle the angle to be set
*/
static PyObject * qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Global_Phase(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args) {
	double global_phase_new_angle;
    if (!PyArg_ParseTuple(args, "|d", &global_phase_new_angle )) return Py_BuildValue("i", -1);
    std::cout<<global_phase_new_angle<<std::endl;
    self->decomp_base->set_global_phase(global_phase_new_angle);
    return Py_BuildValue("i", 0);
    
}

/**
@brief applies the global phase to the Unitary matrix
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_adaptive_Wrapper.
*/
static PyObject * qgd_N_Qubit_Decomposition_adaptive_Wrapper_apply_Global_Phase(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

    // get the number of gates
    self->decomp_base->apply_global_phase();
    return Py_BuildValue("i", 0);
    
}

/**
@brief Lists the gates decomposing the initial unitary. (These gates are the inverse gates of the gates bringing the intial matrix into unity.)
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_List_Gates( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

    self->decomp_base->list_gates( 0 );

    return Py_None;
}




/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Optimized_Parameters( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {

    int parameter_num = self->decomp->get_parameter_num();

    matrix_base<double> parameters_mtx(1, parameter_num);
    double* parameters = parameters_mtx.get_data();
    self->decomp_base->get_optimized_parameters(parameters);

    // reversing the order
    Matrix_real parameters_mtx_reversed(1, parameter_num);
    double* parameters_reversed = parameters_mtx_reversed.get_data();
    for (int idx=0; idx<parameter_num; idx++ ) {
        parameters_reversed[idx] = parameters[parameter_num-1-idx];
    }

    // convert to numpy array
    parameters_mtx_reversed.set_owner(false);
    PyObject * parameter_arr = matrix_real_to_numpy( parameters_mtx_reversed );

    return parameter_arr;
}





/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Optimized_Parameters( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

    PyObject * parameters_arr = NULL;


    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &parameters_arr )) 
        return Py_BuildValue("i", -1);

    
    if ( PyArray_IS_C_CONTIGUOUS(parameters_arr) ) {
        Py_INCREF(parameters_arr);
    }
    else {
        parameters_arr = PyArray_FROM_OTF(parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }


    // get the pointer to the data stored in the input matrices
    double* parameters = (double*)PyArray_DATA(parameters_arr);


    npy_intp param_num = PyArray_Size( parameters_arr );

    // reversing the order
    matrix_base<double> parameters_mtx_reversed(param_num, 1);
    double* parameters_reversed = parameters_mtx_reversed.get_data();
    for (int idx=0; idx<param_num; idx++ ) {
        parameters_reversed[idx] = parameters[param_num-1-idx];
    }

    self->decomp_base->set_optimized_parameters(parameters_reversed, param_num);


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
        self->decomp_base->set_max_layer_num( key_int, value_int );

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
        self->decomp_base->set_iteration_loops( key_int, value_int );

    }

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
    self->decomp_base->set_verbose( verbose );


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
    self->decomp_base->set_debugfile( debugfile_Cpp );


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
    self->decomp_base->set_optimization_tolerance( tolerance );


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
    self->decomp_base->set_optimization_blocks( optimization_block );


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
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Gate_Structure( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    PyObject* gate_structure_py; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &gate_structure_py )) return Py_BuildValue("i", -1);


    // convert gate structure from PyObject to qgd_Gates_Block
    qgd_Gates_Block* qgd_op_block = (qgd_Gates_Block*) gate_structure_py;

    if (  self->decomp != NULL ) {
        self->decomp->set_adaptive_gate_structure( qgd_op_block->gate );
    }
    else if(  self->decomp_general != NULL ) {
        self->decomp_general->set_adaptive_gate_structure( qgd_op_block->gate );
    }
    else {
        return Py_BuildValue("i", 1);
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

    if (  self->decomp != NULL ) {
        self->decomp->add_adaptive_gate_structure( filename_str );
    }
    else if(  self->decomp_general != NULL ) {
        self->decomp_general->add_adaptive_gate_structure( filename_str );
    }
    else {
        return Py_BuildValue("i", 1);
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

    if (  self->decomp != NULL ) {
        self->decomp->set_adaptive_gate_structure( filename_str );
    }
    else if(  self->decomp_general != NULL ) {
        self->decomp_general->set_adaptive_gate_structure( filename_str );
    }
    else {
        return Py_BuildValue("i", 1);
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
    
    if (  self->decomp != NULL ) {
        self->decomp->set_unitary_from_file( filename_str );
    }
    else if(  self->decomp_general != NULL ) {
        self->decomp_general->set_unitary_from_file( filename_str );
    }

    return Py_BuildValue("i", 0);
}

/**
@brief Wrapper function to apply the imported gate structure on the unitary. The transformed unitary is to be decomposed in the calculations, and the imported gate structure is released.
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_apply_Imported_Gate_Structure( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {


    if (  self->decomp != NULL ) {
        self->decomp->apply_imported_gate_structure();
    }
    else if(  self->decomp_general != NULL ) {
        self->decomp_general->apply_imported_gate_structure();
    }
    else {
        return Py_BuildValue("i", 1);
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


    Matrix&& Unitary_mtx = self->decomp->get_Umtx().copy();
    
    // convert to numpy array
    Unitary_mtx.set_owner(false);
    PyObject *Unitary_py = matrix_to_numpy( Unitary_mtx );

    return Unitary_py;
}



static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Unitary( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {


       if ( self->Umtx != NULL ) {
           // release the unitary to be decomposed
           Py_DECREF(self->Umtx);    
           self->Umtx = NULL;
       }

       PyObject *Umtx_arg = NULL;
       //Parse arguments 
       if (!PyArg_ParseTuple(args, "|O", &Umtx_arg )) return Py_BuildValue("i", -1);
	   
       // convert python object array to numpy C API array
       if ( Umtx_arg == NULL ) {
           PyErr_SetString(PyExc_Exception, "Umtx argument in empty");
           return NULL;
       }
	
	self->Umtx = PyArray_FROM_OTF(Umtx_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

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
    self->decomp_base->reorder_qubits( qbit_list_C );


    

    return Py_BuildValue("i", 0);
}





/**
@brief Wrapper method to reorder the qubits in the decomposition class.
@param 
*/
static PyObject *
qgd_N_Qubit_Decomposition_adaptive_Wrapper_add_Layer_To_Imported_Gate_Structure(qgd_N_Qubit_Decomposition_adaptive_Wrapper *self ) {


    if (  self->decomp != NULL ) {
        self->decomp->add_layer_to_imported_gate_structure();
    }
    else if(  self->decomp_general != NULL ) {
        self->decomp_general->add_layer_to_imported_gate_structure();
    }
    else {
        return Py_BuildValue("i", 1);
    }

    return Py_BuildValue("i", 0);
}



/**
@brief Wrapper function to set the radius in which randomized parameters are generated around the current minimum duting the optimization process
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition_custom_Wrapper.
@param args A tuple of the input arguments: gate_structure_dict (PyDict)
@return Returns with zero on success.
*/
static PyObject * qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Randomized_Radius( qgd_N_Qubit_Decomposition_adaptive_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    double radius = 1.0; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|d", &radius )) return Py_BuildValue("i", -1);


    if (  self->decomp != NULL ) {
        self->decomp->set_randomized_radius( radius );
    }
    else if(  self->decomp_general != NULL ) {
        self->decomp_general->set_randomized_radius( radius );
    }
    else {
        return Py_BuildValue("i", 1);
    }

    return Py_BuildValue("i", 0);


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
    {"get_Gate_Num", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_gate_num, METH_NOARGS,
     "Method to get the number of decomposing gates."
    },
    {"get_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Optimized_Parameters, METH_NOARGS,
     "Method to get the array of optimized parameters."
    },
    {"set_Optimized_Parameters", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Optimized_Parameters, METH_VARARGS,
     "Method to set the initial array of optimized parameters."
    },
    {"get_Gate", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_gate, METH_VARARGS,
     "Method to get the i-th decomposing gates."
    },
    {"get_Gates", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_gates, METH_NOARGS,
     "Method to get the tuple of decomposing gates."
    },
    {"List_Gates", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_List_Gates, METH_NOARGS,
     "Call to print the decomposing nitaries on standard output"
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
    {"apply_Global_Phase", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_apply_Global_Phase, METH_NOARGS,
     "Call to apply global phase on Unitary matrix"},
    {"get_Unitary", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_get_Unitary, METH_NOARGS,
     "Call to get Unitary Matrix"
    },
    {"add_Layer_To_Imported_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_add_Layer_To_Imported_Gate_Structure, METH_NOARGS,
     "Call to add an adaptive layer to the gate structure previously imported gate structure"
    },
    {"apply_Imported_Gate_Structure", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_apply_Imported_Gate_Structure, METH_NOARGS,
     "Call to apply the imported gate structure on the unitary. The transformed unitary is to be decomposed in the calculations, and the imported gate structure is released."
    },
    {"set_Randomized_Radius", (PyCFunction) qgd_N_Qubit_Decomposition_adaptive_Wrapper_set_Randomized_Radius, METH_VARARGS,
     "Call to set the radius in which randomized parameters are generated around the current minimum duting the optimization process."
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
  "Object to represent a Gates_block class of the QGD package.", /*tp_doc*/
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
    .m_name = "qgd_N_Qubit_Decomposition_adaptive_Wrapper",
    .m_doc = "Python binding for QGD N_Qubit_Decomposition class",
    .m_size = -1,
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

