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
\file gates_Wrapper.cpp
\brief Python interface to expose Squander gates to Python
*/

#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "Gate.h"
#include "CH.h"
#include "CNOT.h"
#include "CZ.h"
#include "CRY.h"
#include "H.h"
#include "RX.h"
#include "RY.h"
#include "RZ.h"
#include "SX.h"
#include "SYC.h"
#include "U1.h"
#include "U2.h"
#include "U3.h"
#include "X.h"
#include "Y.h"
#include "Z.h"
#include "T.h"
#include "Tdg.h"
#include "R.h"
#include "CR.h"
#include "CROT.h"
#include "numpy_interface.h"


//////////////////////////////////////

/**
@brief Type definition of the  Gate_Wrapper Python class of the  gates module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the CH gate
    Gate* gate;
} Gate_Wrapper;




template<typename GateT>
Gate* create_gate( int qbit_num, int target_qbit ) {
    GateT* gate = new GateT( qbit_num, target_qbit );
    return static_cast<Gate*>( gate );
}


template<typename GateT>
Gate* create_controlled_gate( int qbit_num, int target_qbit, int control_qbit ) {

    GateT* gate = new GateT( qbit_num, target_qbit, control_qbit );
    return static_cast<Gate*>( gate );
        
}



/**
@brief Method called when a python instance of the class  Gate_Wrapper is destroyed
@param self A pointer pointing to an instance of class  Gate_Wrapper.
*/
static void
 Gate_Wrapper_dealloc(Gate_Wrapper *self)
{
    if( self->gate != NULL ) {
        delete( self->gate );
        self->gate = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class  qgd_CH_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class  qgd_CH_Wrapper.
*/
static PyObject *
 generic_Gate_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {(char*)"qbit_num", NULL};
    int qbit_num = -1; 


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &qbit_num)) {
        std::string err( "Unable to parse arguments");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;   
    }


    Gate_Wrapper *self;
    self = (Gate_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->gate = new Gate( qbit_num );
    }


    return (PyObject *) self;
}



/**
@brief Method called when a python instance of a non-controlled gate class is initialized
@param self A pointer pointing to an instance of the class  Gate_Wrapper.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int)
@param kwds A tuple of keywords
*/
template<typename GateT>
static PyObject *
 Gate_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {(char*)"qbit_num", (char*)"target_qbit", NULL};
    int qbit_num = -1; 
    int target_qbit = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist, &qbit_num, &target_qbit)) {
        std::string err( "Unable to parse arguments");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;   
    }


    Gate_Wrapper *self;
    self = (Gate_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->gate = create_gate<GateT>( qbit_num, target_qbit );
    }


    return (PyObject *) self;
}


/**
@brief Method called when a python instance of a controlled gate class  is initialized
@param self A pointer pointing to an instance of the class  Gate_Wrapper.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), control_qbit (int)
@param kwds A tuple of keywords
*/
template<typename GateT>
static PyObject *
 controlled_gate_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"qbit_num", (char*)"target_qbit", (char*)"control_qbit", NULL};
    int qbit_num = -1; 
    int target_qbit = -1;
    int control_qbit = -1;


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist, &qbit_num, &target_qbit, &control_qbit)) {
        std::string err( "Unable to parse arguments");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;   
    }


    Gate_Wrapper *self;
    self = (Gate_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->gate = create_controlled_gate<GateT>( qbit_num, target_qbit, control_qbit );
    }

    return (PyObject *) self;
    
}


/**
@brief Method called when a python instance of a non-controlled gate class is initialized
@param self A pointer pointing to an instance of the class  Gate_Wrapper.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static int
 Gate_Wrapper_init(Gate_Wrapper *self, PyObject *args, PyObject *kwds)
{

    

    return 0;
}







/**
@brief Call te extract t he matric representation of the gate
@param start_index The index of the first inverse gate
*/
static PyObject *
Gate_Wrapper_get_Matrix( Gate_Wrapper *self, PyObject *args, PyObject *kwds ) {

    static char *kwlist[] = {(char*)"parameters", NULL};

    PyArrayObject * parameters_arr = NULL;


    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &parameters_arr )) {
        std::string err( "Unable to parse keyword arguments");
        return NULL;
    }

    Gate* gate = self->gate;

    Matrix gate_mtx;

    if( gate->get_parameter_num() == 0 ) {

        if( parameters_arr != NULL ) {
            std::string err( "The gate contains no parameters to set, but parameter array was given as input");
            return NULL;
        }

        int parallel = 1;
        gate_mtx = gate->get_matrix( parallel );

    }
    else if( gate->get_parameter_num() > 0 ) {

        if( parameters_arr == NULL ) {
            std::string err( "The gate has free parameters to set, but no parameter array was given as input");
            return NULL;
        }

        if ( PyArray_TYPE(parameters_arr) != NPY_DOUBLE ) {
            PyErr_SetString(PyExc_Exception, "Parameter vector should be real typed");
            return NULL;
        }

        if ( PyArray_IS_C_CONTIGUOUS(parameters_arr) ) {
            Py_INCREF(parameters_arr);
        }
        else {
            parameters_arr = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        }


        // get the C++ wrapper around the input data
        Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );

        int parallel = 1;
        gate_mtx = self->gate->get_matrix( parameters_mtx, parallel );

        Py_DECREF(parameters_arr);


    }
    else {
        std::string err( "The number of parameters in a gate is set to a negative value");
        return NULL;

    }


    // convert to numpy array
    gate_mtx.set_owner(false);
    PyObject *gate_mtx_py = matrix_to_numpy( gate_mtx );
    

    return gate_mtx_py;

}



/**
@brief Call to apply the gate operation on an input state or matrix
*/
static PyObject *
Gate_Wrapper_Wrapper_apply_to( Gate_Wrapper *self, PyObject *args, PyObject *kwds ) {

    static char *kwlist[] = {(char*)"unitary", (char*)"parameters", (char*)"parallel", NULL};

    PyArrayObject * input = NULL;
    PyArrayObject * parameters_arr = NULL;
    int parallel = 1;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|Oi", kwlist, &input, &parameters_arr, &parallel )) {
        std::string err( "Unable to parse keyword arguments");
        return NULL;
    }

    // Check if input matrix is provided
    if ( input == NULL ) {
        PyErr_SetString(PyExc_Exception, "Input matrix was not given");
        return NULL;
    }

    if ( PyArray_TYPE(input) != NPY_COMPLEX128 ) {
        PyErr_SetString(PyExc_Exception, "input matrix or state should be complex typed");
        return NULL;
    }    

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(input) ) {
        PyErr_SetString(PyExc_Exception, "input state/matrix is not memory contiguous");
        return NULL;
    }

    // create QGD version of the input matrix
    Matrix input_mtx = numpy2matrix(input);

    Gate* gate = self->gate;

    const int param_count = gate->get_parameter_num();

    try {
        if (param_count == 0) {
            // Non-parameterized gate
            gate->apply_to(input_mtx, parallel);
        } 
        else if (param_count > 0) {
            // Parameterized gate
            if( parameters_arr == NULL ) {
                std::string err( "The gate has free parameters to set, but no parameter array was given as input");
                return NULL;
            }

            if ( PyArray_TYPE(parameters_arr) != NPY_DOUBLE ) {
                PyErr_SetString(PyExc_TypeError, "Parameter vector should be real typed");
                return NULL;
            }

            // Convert parameters to C++ matrix
            if ( PyArray_IS_C_CONTIGUOUS(parameters_arr) ) {
                Py_INCREF(parameters_arr);
            }
            else {
                parameters_arr = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            }
            
            Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );

            gate->apply_to(parameters_mtx, input_mtx, parallel);

            Py_DECREF(parameters_arr);
        }
        else {
            PyErr_SetString(PyExc_ValueError, "The number of parameters in a gate is set to a negative value");
            return NULL;
        }
    }
    catch (const std::string& err) {
        PyErr_SetString(PyExc_RuntimeError, err.c_str());
        return NULL;
    }
    catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown error in gate operation");
        return NULL;
    }


    // if numpy array was not aligned to memory boundaries, the input is reallocated on the C++ side
    if (input_mtx.data != PyArray_DATA(input)) {
        memcpy(PyArray_DATA(input), input_mtx.data, input_mtx.size() * sizeof(QGD_Complex16));
    }

    return Py_BuildValue("i", 0);
}



/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on a single qbit space.
@param ThetaOver2 Real parameter standing for the parameter theta (optional).
@param Phi Real parameter standing for the parameter phi (optional).
@param Lambda Real parameter standing for the parameter lambda (optional).
@return Returns with the matrix of the one-qubit matrix.
*/

static PyObject *
Gate_Wrapper_get_Gate_Kernel( Gate_Wrapper *self, PyObject *args, PyObject *kwds) {


    static char *kwlist[] = {(char*)"ThetaOver2", (char*)"Phi", (char*)"Lambda", NULL};

    double ThetaOver2;
    double Phi; 
    double Lambda; 


    try {
        self->gate->parameters_for_calc_one_qubit(ThetaOver2, Phi, Lambda);
    }
    catch (std::string err) {    
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to gate class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ddd", kwlist, &ThetaOver2, &Phi, &Lambda ))  {
        std::string err( "Unable to parse keyword arguments");
        return NULL;
    }


    Matrix CH_1qbit_;

    // create QGD version of the input matrix
    Gate* gate = self->gate;

    if( gate->get_parameter_num() == 0 ) {
       CH_1qbit_ = self->gate->calc_one_qubit_u3( );
    }
    else if( gate->get_parameter_num() > 0 ) {
       CH_1qbit_ = self->gate->calc_one_qubit_u3( ThetaOver2, Phi, Lambda );
    }
    else {
        std::string err( "The number of parameters in a gate is set to a negative value");
        return NULL;

    }

 
    PyObject *CH_1qbit = matrix_to_numpy( CH_1qbit_ );

    return CH_1qbit;


}



/**
@brief Call to get the number of free parameters in the gate
@return Returns with the number of parameters
*/
static PyObject *
Gate_Wrapper_get_Parameter_Num( Gate_Wrapper *self ) {

    int parameter_num;

    try {
        parameter_num = self->gate->get_parameter_num();
    }
    catch (std::string err) {    
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to gate class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    return Py_BuildValue("i", parameter_num);

}



/**
@brief Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated
@return Returns with the starting index
*/
static PyObject *
Gate_Wrapper_get_Parameter_Start_Index( Gate_Wrapper *self ) {

    int start_index;

    try {
        start_index = self->gate->get_parameter_start_idx();
    }
    catch (std::string err) {    
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to gate class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    return Py_BuildValue("i", start_index);

}




/**
@brief Call to get the target qbit
@return Returns with the target qbit
*/
static PyObject *
Gate_Wrapper_get_Target_Qbit( Gate_Wrapper *self ) {

    int target_qbit;

    try {
        target_qbit = self->gate->get_target_qbit();
    }
    catch (std::string err) {    
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to gate class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    return Py_BuildValue("i", target_qbit);

}

/**
@brief Call to get the control qbit (returns with -1 if no control qbit is used in the gate)
@return Returns with the control qbit
*/
static PyObject *
Gate_Wrapper_get_Control_Qbit( Gate_Wrapper *self ) {

    int control_qbit;

    try {
        control_qbit = self->gate->get_control_qbit();
    }
    catch (std::string err) {    
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to gate class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    return Py_BuildValue("i", control_qbit);

} 



/**
@brief Call to set the target qbit
*/
static PyObject *
Gate_Wrapper_set_Target_Qbit( Gate_Wrapper *self, PyObject *args ) {

    int target_qbit_in = -1;
    if (!PyArg_ParseTuple(args, "|i", &target_qbit_in ))   {
        std::string err( "Unable to parse arguments");
        return NULL;
    }
        
    try{
        self->gate->set_target_qbit(target_qbit_in);
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to circuit class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    return Py_BuildValue("i", 0);

}

/**
@brief Call to set the target qbit
*/
static PyObject *
Gate_Wrapper_set_Control_Qbit( Gate_Wrapper *self, PyObject *args ) {

    int control_qbit_in = -1;
    if (!PyArg_ParseTuple(args, "|i", &control_qbit_in ))    {
        std::string err( "Unable to parse arguments");
        return NULL;
    }
        
    try{
        self->gate->set_control_qbit(control_qbit_in);
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to circuit class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }



    return Py_BuildValue("i", 0);

}



/**
@brief Call to extract the paramaters corresponding to the gate, from a parameter array associated to the circuit in which the gate is embedded.
@return Returns with he extracted parameters
*/
static PyObject *
Gate_Wrapper_Extract_Parameters( Gate_Wrapper *self, PyObject *args ) {

    PyArrayObject * parameters_arr = NULL;


    // parsing input arguments
    if (!PyArg_ParseTuple(args, "O", &parameters_arr )) {
        PyErr_SetString(PyExc_ValueError, "Unable to parse arguments");
        return NULL;
    }

    if( parameters_arr == NULL ) {
        PyErr_SetString(PyExc_ValueError, "Missing input parameter array");
        return NULL;
    }

    if (PyArray_TYPE(parameters_arr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Parameter array must contain double values");
        return NULL;
    }
    
    if ( PyArray_IS_C_CONTIGUOUS(parameters_arr) ) {
        Py_INCREF(parameters_arr);
    }
    else {
        parameters_arr = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }

    // get the C++ wrapper around the data
    Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );

    Matrix_real extracted_parameters;

    try {
        extracted_parameters = self->gate->extract_parameters( parameters_mtx );
    }
    catch (std::string err) {
        Py_DECREF(parameters_arr);
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        Py_DECREF(parameters_arr);
        std::string err( "Invalid pointer to circuit class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    // convert to numpy array
    extracted_parameters.set_owner(false);
    PyObject *extracted_parameters_py = matrix_real_to_numpy( extracted_parameters );

    // flatten the extracted array
    long int param_num = (long int)extracted_parameters.size();
    PyArray_Dims new_shape;
    new_shape.ptr = &param_num;
    new_shape.len = 1;

    PyObject *extracted_parameters_py_flatten = PyArray_Newshape( (PyArrayObject*)extracted_parameters_py, &new_shape, NPY_CORDER);
   
    Py_DECREF(parameters_arr);
    return extracted_parameters_py_flatten;
}




/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
Gate_Wrapper_get_Name( Gate_Wrapper *self ) {

    std::string name;
    try {
        name = self->gate->get_name();
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to circuit class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    return PyUnicode_FromString(name.c_str());
}



/**
@brief Method to extract the stored quantum gate in a human-readable data serialized and pickle-able format
*/
static PyObject * 
Gate_Wrapper_getstate( Gate_Wrapper *self ) {

    PyObject* gate_state = PyDict_New();

    if( gate_state == NULL ) {
        std::string err( "Failed to create dictionary");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;    
    }
    

    PyObject* key = Py_BuildValue( "s", "type" );
    PyObject* val = Py_BuildValue("i", self->gate->get_type() );    
    PyDict_SetItem(gate_state, key, val);

    key = Py_BuildValue( "s", "qbit_num" );
    val = Py_BuildValue("i", self->gate->get_qbit_num() );    
    PyDict_SetItem(gate_state, key, val);

    key = Py_BuildValue( "s", "target_qbit" );
    val = Py_BuildValue("i", self->gate->get_target_qbit() );    
    PyDict_SetItem(gate_state, key, val);

    key = Py_BuildValue( "s", "control_qbit" );
    val = Py_BuildValue("i", self->gate->get_control_qbit() );    
    PyDict_SetItem(gate_state, key, val);


    return gate_state;

}





/**
@brief Call to set the state of quantum gate from a human-readable data serialized and pickle-able format
*/
static PyObject * 
Gate_Wrapper_setstate( Gate_Wrapper *self, PyObject *args ) {


    PyObject* gate_state = NULL;


    // parsing input arguments
    if (!PyArg_ParseTuple(args, "O", &gate_state )) {
        PyErr_SetString(PyExc_ValueError, "Unable to parse arguments");
        return NULL;
    }
    
    if( !PyDict_Check( gate_state ) ) {
        std::string err( "Gate state should be given by a dictionary");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }   
    
    PyObject* qbit_num_key = Py_BuildValue( "s", "qbit_num" );    
    if ( PyDict_Contains(gate_state, qbit_num_key) == 0 ) {
        std::string err( "Gate state should contain the number of qubits");
        PyErr_SetString(PyExc_Exception, err.c_str());

        Py_DECREF( qbit_num_key );
        return NULL;
    }    
    PyObject* qbit_num_py = PyDict_GetItem(gate_state, qbit_num_key); // borrowed reference
    Py_DECREF( qbit_num_key );
    
    
    PyObject* target_qbit_key = Py_BuildValue( "s", "target_qbit" );
    if ( PyDict_Contains(gate_state, target_qbit_key) == 0 ) {
        std::string err( "Gate state should contain a target qubit");
        PyErr_SetString(PyExc_Exception, err.c_str());

        Py_DECREF( target_qbit_key );
        return NULL;
    }    
    PyObject* target_qbit_py = PyDict_GetItem(gate_state, target_qbit_key); // borrowed reference
    Py_DECREF( target_qbit_key );
    
    
    PyObject* control_qbit_key = Py_BuildValue( "s", "control_qbit" );
    if ( PyDict_Contains(gate_state, control_qbit_key) == 0 ) {
        std::string err( "Gate state should contain a control qubit (-1 for gates with no control qubits)");
        PyErr_SetString(PyExc_Exception, err.c_str());

        Py_DECREF( control_qbit_key );
        return NULL;
    }    
    PyObject* control_qbit_py = PyDict_GetItem(gate_state, control_qbit_key); // borrowed reference
    Py_DECREF( control_qbit_key );





    PyObject* type_key = Py_BuildValue( "s", "type" );
    if ( PyDict_Contains(gate_state, type_key) == 0 ) {
        std::string err( "Gate state should contain a type ID (see gate.h for the gate type IDs)");
        PyErr_SetString(PyExc_Exception, err.c_str());

        Py_DECREF( type_key );
        return NULL;
    }    
    PyObject* type_py = PyDict_GetItem(gate_state, type_key); // borrowed reference
    Py_DECREF( type_key );



    int qbit_num = (int)PyLong_AsLong( qbit_num_py );
    int target_qbit = (int)PyLong_AsLong( target_qbit_py );
    int control_qbit = (int)PyLong_AsLong( control_qbit_py );    
    int gate_type = (int)PyLong_AsLong( type_py );    
    

    Gate* gate = NULL;
    
    switch (gate_type) {
    case CNOT_OPERATION: {
        gate = create_controlled_gate<CNOT>( qbit_num, target_qbit, control_qbit );
        break;
    }    
    case CZ_OPERATION:
    {
        gate = create_controlled_gate<CZ>( qbit_num, target_qbit, control_qbit );
        break;
    }    
    case CH_OPERATION: {
        gate = create_controlled_gate<CH>( qbit_num, target_qbit, control_qbit );
        break;
    }     
    case SYC_OPERATION: {
        gate = create_controlled_gate<SYC>( qbit_num, target_qbit, control_qbit );
        break;
    }    
    case X_OPERATION: {
        gate = create_gate<X>( qbit_num, target_qbit );
        break;
    }    
    case Y_OPERATION: {
        gate = create_gate<Y>( qbit_num, target_qbit );
        break;
    }    
    case Z_OPERATION: {
        gate = create_gate<Z>( qbit_num, target_qbit );
        break;
    }    
    case SX_OPERATION: {
        gate = create_gate<SX>( qbit_num, target_qbit );
        break;
    }    
    case T_OPERATION: {
        gate = create_gate<T>( qbit_num, target_qbit );
        break;
    }    
    case TDG_OPERATION: {
        gate = create_gate<Tdg>( qbit_num, target_qbit );
        break;
    }    
    case H_OPERATION: {
        gate = create_gate<H>( qbit_num, target_qbit );
        break;
    }  
    case U1_OPERATION: {
        gate = create_gate<U1>( qbit_num, target_qbit );
        break;
    }    
    case U2_OPERATION: {
        gate = create_gate<U2>( qbit_num, target_qbit );
        break;
    }      
    case U3_OPERATION: {
        gate = create_gate<U3>( qbit_num, target_qbit );
        break; 
    }
    case R_OPERATION: {
        gate = create_gate<R>( qbit_num, target_qbit );
        break;
    }    
    case RX_OPERATION: {
        gate = create_gate<RX>( qbit_num, target_qbit );
        break;
    }    
    case RY_OPERATION: {
        gate = create_gate<RY>( qbit_num, target_qbit );
        break;
    }    
    case CRY_OPERATION: {
        gate = create_controlled_gate<CRY>( qbit_num, target_qbit, control_qbit );
        break;
    }    
    case CROT_OPERATION: {
        gate = create_controlled_gate<CROT>( qbit_num, target_qbit, control_qbit );
        break;
    }    
    case CR_OPERATION: {
        gate = create_controlled_gate<CR>( qbit_num, target_qbit, control_qbit );
        break;
    }    
    
    case RZ_OPERATION: {
        gate = create_gate<RZ>( qbit_num, target_qbit );
        break;
    }    
    case BLOCK_OPERATION: {
        std::string err( "Unsupported gate type: block operation");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;  
    }
    default:
        std::string err( "Unsupported gate type");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;  
    }



    try {
        delete( self->gate );
        self->gate = gate;
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to circuit class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }

    
    return Py_None;

}





extern "C"
{

/**
@brief Structure containing metadata about the methods of class qgd_U3.
*/
static PyMethodDef Gate_Wrapper_methods[] = {
    {"get_Matrix", (PyCFunction) Gate_Wrapper_get_Matrix, METH_VARARGS | METH_KEYWORDS,
     "Method to get the matrix representation of the gate."
    },
    {"apply_to", (PyCFunction) Gate_Wrapper_Wrapper_apply_to, METH_VARARGS | METH_KEYWORDS,
     "Call to apply the gate on an input state/matrix."
    },
    {"get_Gate_Kernel", (PyCFunction) Gate_Wrapper_get_Gate_Kernel, METH_VARARGS | METH_KEYWORDS,
     "Call to calculate the gate matrix acting on a single qbit space."
    },
    {"get_Parameter_Num", (PyCFunction) Gate_Wrapper_get_Parameter_Num, METH_NOARGS,
     "Call to get the number of free parameters in the gate."
    },
    {"get_Parameter_Start_Index", (PyCFunction) Gate_Wrapper_get_Parameter_Start_Index, METH_NOARGS,
     "Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated."
    },
    {"get_Target_Qbit", (PyCFunction) Gate_Wrapper_get_Target_Qbit, METH_NOARGS,
     "Call to get the target qbit."
    },
    {"get_Control_Qbit", (PyCFunction) Gate_Wrapper_get_Control_Qbit, METH_NOARGS,
     "Call to get the control qbit (returns with -1 if no control qbit is used in the gate)."
    },
    {"set_Target_Qbit", (PyCFunction) Gate_Wrapper_set_Target_Qbit, METH_VARARGS,
     "Call to set the target qbit."
    },
    {"set_Control_Qbit", (PyCFunction) Gate_Wrapper_set_Control_Qbit, METH_VARARGS,
     "Call to set the control qbit."
    },
    {"Extract_Parameters", (PyCFunction) Gate_Wrapper_Extract_Parameters, METH_VARARGS,
     "Call to extract the paramaters corresponding to the gate, from a parameter array associated to the circuit in which the gate is embedded."
    },
    {"get_Name", (PyCFunction) Gate_Wrapper_get_Name, METH_NOARGS,
     "Method to get the name label of the gate"
    },
    {"__getstate__", (PyCFunction) Gate_Wrapper_getstate, METH_NOARGS,
     "Method to extract the stored quantum gate in a human-readable data serialized and pickle-able format"
    },
    {"__setstate__", (PyCFunction) Gate_Wrapper_setstate, METH_VARARGS,
     "Call to set the state of quantum gate from a human-readable data serialized and pickle-able format"
    },
    {NULL}  /* Sentinel */
};


/**
@brief Structure containing metadata about the members of class  qgd_CH_Wrapper.
*/
static PyMemberDef  Gate_Wrapper_members[] = {
    {NULL}  /* Sentinel */
};


struct Gate_Wrapper_Type_tmp : PyTypeObject {


    Gate_Wrapper_Type_tmp() {
    
        //PyVarObject tt = { PyVarObject_HEAD_INIT(NULL, 0) };
    
        ob_base.ob_size = 0;
        tp_name      = "Gate";
        tp_basicsize = sizeof(Gate_Wrapper);
        tp_dealloc   = (destructor)  Gate_Wrapper_dealloc;
        tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        tp_doc       = "Object to represent python binding for a generic base gate of the Squander package.";
        tp_methods   = Gate_Wrapper_methods;
        tp_members   = Gate_Wrapper_members;
        tp_init      = (initproc)  Gate_Wrapper_init;
        tp_new       = generic_Gate_Wrapper_new;
    }
    

};


static Gate_Wrapper_Type_tmp Gate_Wrapper_Type;


struct CH_Wrapper_Type : Gate_Wrapper_Type_tmp {

    CH_Wrapper_Type() {
        tp_name      = "CH";
        tp_doc       = "Object to represent python binding for a CH gate of the Squander package.";
        tp_new      = (newfunc) controlled_gate_Wrapper_new<CH>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct CNOT_Wrapper_Type : Gate_Wrapper_Type_tmp {

    CNOT_Wrapper_Type() {    
        tp_name      = "CNOT";
        tp_doc       = "Object to represent python binding for a CNOT gate of the Squander package.";
        tp_new      = (newfunc) controlled_gate_Wrapper_new<CNOT>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct CZ_Wrapper_Type : Gate_Wrapper_Type_tmp {

    CZ_Wrapper_Type() {    
        tp_name      = "CZ";
        tp_doc       = "Object to represent python binding for a CZ gate of the Squander package.";
        tp_new      = (newfunc) controlled_gate_Wrapper_new<CZ>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct CRY_Wrapper_Type : Gate_Wrapper_Type_tmp {

    CRY_Wrapper_Type() {    
        tp_name      = "CRY";
        tp_doc       = "Object to represent python binding for a CRY gate of the Squander package.";
        tp_new      = (newfunc) controlled_gate_Wrapper_new<CRY>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct H_Wrapper_Type : Gate_Wrapper_Type_tmp {

    H_Wrapper_Type() {    
        tp_name      = "H";
        tp_doc       = "Object to represent python binding for a H gate of the Squander package.";
        tp_new      = (newfunc) Gate_Wrapper_new<H>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct RX_Wrapper_Type : Gate_Wrapper_Type_tmp {

    RX_Wrapper_Type() {    
        tp_name      = "RX";
        tp_doc       = "Object to represent python binding for a RX gate of the Squander package.";
        tp_new      = (newfunc) Gate_Wrapper_new<RX>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct RY_Wrapper_Type : Gate_Wrapper_Type_tmp {

    RY_Wrapper_Type() {    
        tp_name      = "RY";
        tp_doc       = "Object to represent python binding for a RY gate of the Squander package.";
        tp_new      = (newfunc) Gate_Wrapper_new<RY>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct RZ_Wrapper_Type : Gate_Wrapper_Type_tmp {

    RZ_Wrapper_Type() {    
        tp_name      = "RZ";
        tp_doc       = "Object to represent python binding for a RZ gate of the Squander package.";
        tp_new      = (newfunc) Gate_Wrapper_new<RZ>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct SX_Wrapper_Type : Gate_Wrapper_Type_tmp {

    SX_Wrapper_Type() {    
        tp_name      = "SX";
        tp_doc       = "Object to represent python binding for a SX gate of the Squander package.";
        tp_new      = (newfunc) Gate_Wrapper_new<SX>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct SYC_Wrapper_Type : Gate_Wrapper_Type_tmp {

    SYC_Wrapper_Type() {    
        tp_name      = "SYC";
        tp_doc       = "Object to represent python binding for a SYC gate of the Squander package.";
        tp_new      = (newfunc) controlled_gate_Wrapper_new<SYC>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct U1_Wrapper_Type : Gate_Wrapper_Type_tmp {

    U1_Wrapper_Type() {    
        tp_name      = "U1";
        tp_doc       = "Object to represent python binding for a U1 gate of the Squander package.";
        tp_new       = (newfunc) Gate_Wrapper_new<U1>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct U2_Wrapper_Type : Gate_Wrapper_Type_tmp {

    U2_Wrapper_Type() {    
        tp_name      = "U2";
        tp_doc       = "Object to represent python binding for a U2 gate of the Squander package.";
        tp_new       = (newfunc) Gate_Wrapper_new<U2>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct U3_Wrapper_Type : Gate_Wrapper_Type_tmp {

    U3_Wrapper_Type() {    
        tp_name      = "U3";
        tp_doc       = "Object to represent python binding for a U3 gate of the Squander package.";
        tp_new       = (newfunc) Gate_Wrapper_new<U3>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct X_Wrapper_Type : Gate_Wrapper_Type_tmp {

    X_Wrapper_Type() {    
        tp_name      = "X";
        tp_doc       = "Object to represent python binding for a X gate of the Squander package.";
        tp_new      = (newfunc) Gate_Wrapper_new<X>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct Y_Wrapper_Type : Gate_Wrapper_Type_tmp {

    Y_Wrapper_Type() {    
        tp_name      = "Y";
        tp_doc       = "Object to represent python binding for a Y gate of the Squander package.";
        tp_new      = (newfunc) Gate_Wrapper_new<Y>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct Z_Wrapper_Type : Gate_Wrapper_Type_tmp {

    Z_Wrapper_Type() {    
        tp_name      = "Z";
        tp_doc       = "Object to represent python binding for a Z gate of the Squander package.";
        tp_new      = (newfunc) Gate_Wrapper_new<Z>;
        tp_base      = &Gate_Wrapper_Type;
    }
};


struct T_Wrapper_Type : Gate_Wrapper_Type_tmp {

    T_Wrapper_Type() {    
        tp_name      = "T";
        tp_doc       = "Object to represent python binding for a T gate of the Squander package.";
        tp_new      = (newfunc) Gate_Wrapper_new<T>;
        tp_base      = &Gate_Wrapper_Type;
    }
};


struct Tdg_Wrapper_Type : Gate_Wrapper_Type_tmp {

    Tdg_Wrapper_Type() {    
        tp_name      = "Tdg";
        tp_doc       = "Object to represent python binding for a T gate of the Squander package.";
        tp_new      = (newfunc) Gate_Wrapper_new<Tdg>;
        tp_base      = &Gate_Wrapper_Type;
    }
};


struct R_Wrapper_Type : Gate_Wrapper_Type_tmp {

    R_Wrapper_Type() {    
        tp_name      = "R";
        tp_doc       = "Object to represent python binding for a R gate of the Squander package.";
        tp_new      = (newfunc) Gate_Wrapper_new<R>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct CR_Wrapper_Type : Gate_Wrapper_Type_tmp {

    CR_Wrapper_Type() {    
        tp_name      = "CR";
        tp_doc       = "Object to represent python binding for a CR gate of the Squander package.";
        tp_new      = (newfunc) controlled_gate_Wrapper_new<CR>;
        tp_base      = &Gate_Wrapper_Type;
    }
};

struct CROT_Wrapper_Type : Gate_Wrapper_Type_tmp {

    CROT_Wrapper_Type() {    
        tp_name      = "CROT";
        tp_doc       = "Object to represent python binding for a CROT gate of the Squander package.";
        tp_new      = (newfunc) controlled_gate_Wrapper_new<CROT>;
        tp_base      = &Gate_Wrapper_Type;
    }
};


static CH_Wrapper_Type CH_Wrapper_Type_ins;
static CNOT_Wrapper_Type CNOT_Wrapper_Type_ins;
static CZ_Wrapper_Type CZ_Wrapper_Type_ins;
static CRY_Wrapper_Type CRY_Wrapper_Type_ins;
static H_Wrapper_Type H_Wrapper_Type_ins;
static RX_Wrapper_Type RX_Wrapper_Type_ins;
static RY_Wrapper_Type RY_Wrapper_Type_ins;
static RZ_Wrapper_Type RZ_Wrapper_Type_ins;
static SX_Wrapper_Type SX_Wrapper_Type_ins;
static SYC_Wrapper_Type SYC_Wrapper_Type_ins;
static U1_Wrapper_Type U1_Wrapper_Type_ins;
static U2_Wrapper_Type U2_Wrapper_Type_ins;
static U3_Wrapper_Type U3_Wrapper_Type_ins;
static X_Wrapper_Type X_Wrapper_Type_ins;
static Y_Wrapper_Type Y_Wrapper_Type_ins;
static Z_Wrapper_Type Z_Wrapper_Type_ins;
static T_Wrapper_Type T_Wrapper_Type_ins;
static Tdg_Wrapper_Type Tdg_Wrapper_Type_ins;
static R_Wrapper_Type R_Wrapper_Type_ins;
static CR_Wrapper_Type CR_Wrapper_Type_ins;
static CROT_Wrapper_Type CROT_Wrapper_Type_ins;




////////////////////////////////////////




/**
@brief Structure containing metadata about the module.
*/
static PyModuleDef  gates_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    "gates_Wrapper",
    "Python binding for gates implemented in Squander C++",
    -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_gates_Wrapper(void)
{

    // initialize Numpy API
    import_array();


    PyObject * m= PyModule_Create(& gates_Wrapper_Module);
    if (m == NULL)
        return NULL;


    if (PyType_Ready(&Gate_Wrapper_Type) < 0 ||
        PyType_Ready(&CH_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&CNOT_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&CZ_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&CRY_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&H_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&RX_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&RY_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&RZ_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&SX_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&SYC_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&U1_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&U2_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&U3_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&X_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&Y_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&Z_Wrapper_Type_ins) < 0 || 
        PyType_Ready(&T_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&Tdg_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&CR_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&CROT_Wrapper_Type_ins) < 0 ||
        PyType_Ready(&R_Wrapper_Type_ins) < 0 ) {

        Py_DECREF(m);
        return NULL;
    }


    Py_INCREF(&Gate_Wrapper_Type);
    if (PyModule_AddObject(m, "Gate", (PyObject *) & Gate_Wrapper_Type) < 0) {
        Py_DECREF(& Gate_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }


    Py_INCREF(&CH_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "CH", (PyObject *) & CH_Wrapper_Type_ins) < 0) {
        Py_DECREF(& CH_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }


    Py_INCREF(&CNOT_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "CNOT", (PyObject *) & CNOT_Wrapper_Type_ins) < 0) {
        Py_DECREF(& CNOT_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }


    Py_INCREF(&CZ_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "CZ", (PyObject *) & CZ_Wrapper_Type_ins) < 0) {
        Py_DECREF(& CZ_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&CRY_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "CRY", (PyObject *) & CRY_Wrapper_Type_ins) < 0) {
        Py_DECREF(& CRY_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&H_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "H", (PyObject *) & H_Wrapper_Type_ins) < 0) {
        Py_DECREF(& H_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&RX_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "RX", (PyObject *) & RX_Wrapper_Type_ins) < 0) {
        Py_DECREF(& RX_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&RY_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "RY", (PyObject *) & RY_Wrapper_Type_ins) < 0) {
        Py_DECREF(& RY_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }
    
    Py_INCREF(&RZ_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "RZ", (PyObject *) & RZ_Wrapper_Type_ins) < 0) {
        Py_DECREF(& RZ_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&SX_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "SX", (PyObject *) & SX_Wrapper_Type_ins) < 0) {
        Py_DECREF(& SX_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&SYC_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "SYC", (PyObject *) & SYC_Wrapper_Type_ins) < 0) {
        Py_DECREF(& SYC_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&U1_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "U1", (PyObject *) & U1_Wrapper_Type_ins) < 0) {
        Py_DECREF(& U1_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&U2_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "U2", (PyObject *) & U2_Wrapper_Type_ins) < 0) {
        Py_DECREF(& U2_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&U3_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "U3", (PyObject *) & U3_Wrapper_Type_ins) < 0) {
        Py_DECREF(& U3_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&X_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "X", (PyObject *) & X_Wrapper_Type_ins) < 0) {
        Py_DECREF(& X_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }
    
    Py_INCREF(&Y_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "Y", (PyObject *) & Y_Wrapper_Type_ins) < 0) {
        Py_DECREF(& Y_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&Z_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "Z", (PyObject *) & Z_Wrapper_Type_ins) < 0) {
        Py_DECREF(& Z_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }


    Py_INCREF(&T_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "T", (PyObject *) & T_Wrapper_Type_ins) < 0) {
        Py_DECREF(& T_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }


    Py_INCREF(&Tdg_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "Tdg", (PyObject *) & Tdg_Wrapper_Type_ins) < 0) {
        Py_DECREF(& Tdg_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }


    Py_INCREF(&R_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "R", (PyObject *) & R_Wrapper_Type_ins) < 0) {
        Py_DECREF(&R_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }
    
    Py_INCREF(&CR_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "CR", (PyObject *) & CR_Wrapper_Type_ins) < 0) {
        Py_DECREF(&CR_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }
    
    Py_INCREF(&CROT_Wrapper_Type_ins);
    if (PyModule_AddObject(m, "CROT", (PyObject *) & CROT_Wrapper_Type_ins) < 0) {
        Py_DECREF(&CROT_Wrapper_Type_ins);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
