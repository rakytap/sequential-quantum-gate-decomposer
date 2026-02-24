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
/*! \file qgd_Circuit_Wrapper.cpp
    \brief Python interface for the Gates_block class (quantum circuit wrapper)
*/

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "structmember.h"
#include "Gates_block.h"
#include "CZ.h"
#include "CH.h"
#include "CNOT.h"
#include "U1.h"
#include "U2.h"
#include "U3.h"
#include "RX.h"
#include "R.h"
#include "RY.h"
#include "CRY.h"
#include "CRX.h"
#include "CRZ.h"
#include "CP.h"
#include "CCX.h"
#include "SWAP.h"
#include "CSWAP.h"
#include "CROT.h"
#include "CR.h"
#include "RZ.h"
#include "H.h"
#include "X.h"
#include "Y.h"
#include "Z.h"
#include "SX.h"
#include "SXdg.h"
#include "SYC.h"
#include "UN.h"
#include "ON.h"
#include "Adaptive.h"
#include "Composite.h"
#include "RXX.h"
#include "RYY.h"
#include "RZZ.h"

#include "numpy_interface.h"

#ifdef __DFE__
#include <numpy/arrayobject.h>
#include "numpy_interface.h"
#endif

/**
@brief Type definition of the qgd_Gate Python class of the qgd_Gate module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the base Gate gate
    Gate* gate;
} qgd_Gate;

/**
@brief Type definition of the qgd_Operation_Block Python class of the qgd_Operation_Block module
*/
typedef struct qgd_Circuit_Wrapper {
    PyObject_HEAD
    /// Pointer to the C++ class of the base Gate_block module
    Gates_block* circuit;
} qgd_Circuit_Wrapper;

/**
@brief Creates an instance of class Gates_block (Circuit) and returns a pointer to the class instance
@param qbit_num Number of qubits spanning the circuit
@return Returns a pointer to an instance of Gates_block class
*/
Gates_block* 
create_Circuit( int qbit_num ) {
    return new Gates_block(qbit_num);
}

/**
@brief Call to deallocate an instance of Gates_block class
@param ptr A pointer pointing to an instance of Gates_block class
*/
void
release_Circuit( Gates_block*  instance ) {
    delete instance;
    return;
}





extern "C"
{

/**
@brief Method called when a python instance of the class qgd_Circuit_Wrapper is destroyed
@param self A pointer pointing to an instance of class qgd_Circuit_Wrapper.
*/
static void
qgd_Circuit_Wrapper_dealloc(qgd_Circuit_Wrapper *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_Circuit( self->circuit );

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class qgd_Circuit_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class Circuit.
*/
static PyObject *
qgd_Circuit_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"qbit_num", NULL};

    // initiate variables for input arguments
    int  qbit_num = 0; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &qbit_num)) {
        std::string err( "Unable to parse arguments");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;   
    }

    qgd_Circuit_Wrapper *self;
    self = (qgd_Circuit_Wrapper *) type->tp_alloc(type, 0);

    if (self != NULL) {
        self->circuit = create_Circuit( qbit_num );
    }

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_Circuit_Wrapper is initialized
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: qbit_num (integer)
qbit_num: the number of qubits spanning the operations
@param kwds A tuple of keywords
*/
static int
qgd_Circuit_Wrapper_init(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{
    return 0;
}


/**
@brief Structure containing metadata about the members of class qgd_Circuit_Wrapper.
*/
static PyMemberDef qgd_Circuit_Wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


#define qgd_Circuit_Wrapper_add_one_qubit_gate(gate_name, GATE_NAME)\
static PyObject * \
qgd_Circuit_Wrapper_add_##GATE_NAME(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds) \
{\
    static char *kwlist[] = {(char*)"target_qbit", NULL};\
\
    int target_qbit = -1; \
\
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,\
                                     &target_qbit))\
        return Py_BuildValue("i", -1);\
\
    if (target_qbit != -1 ) {\
        self->circuit->add_##gate_name(target_qbit);\
    }\
\
    return Py_BuildValue("i", 0);\
}

#define qgd_Circuit_Wrapper_add_two_qubit_gate(gate_name, GATE_NAME)\
static PyObject * \
qgd_Circuit_Wrapper_add_##GATE_NAME(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds) \
{\
    static char *kwlist[] = {(char*)"target_qbit",  (char*)"control_qbit", NULL};\
    int  target_qbit = -1; \
    int  control_qbit = -1; \
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,\
                                     &target_qbit, &control_qbit))\
        return Py_BuildValue("i", -1);\
    if (target_qbit != -1 ) {\
        self->circuit->add_##gate_name(target_qbit, control_qbit);\
    }\
    return Py_BuildValue("i", 0);\
}


qgd_Circuit_Wrapper_add_one_qubit_gate(u1,U1)

qgd_Circuit_Wrapper_add_one_qubit_gate(u3,U3)

qgd_Circuit_Wrapper_add_one_qubit_gate(u2,U2)

qgd_Circuit_Wrapper_add_one_qubit_gate(rx,RX)

qgd_Circuit_Wrapper_add_one_qubit_gate(ry,RY)

qgd_Circuit_Wrapper_add_one_qubit_gate(rz,RZ)

qgd_Circuit_Wrapper_add_one_qubit_gate(r,R)

qgd_Circuit_Wrapper_add_one_qubit_gate(h, H)

qgd_Circuit_Wrapper_add_one_qubit_gate(x, X)

qgd_Circuit_Wrapper_add_one_qubit_gate(y, Y)

qgd_Circuit_Wrapper_add_one_qubit_gate(z, Z)

qgd_Circuit_Wrapper_add_one_qubit_gate(sx, SX)

qgd_Circuit_Wrapper_add_one_qubit_gate(sxdg, SXdg)

qgd_Circuit_Wrapper_add_one_qubit_gate(s, S)

qgd_Circuit_Wrapper_add_one_qubit_gate(sdg, Sdg)

qgd_Circuit_Wrapper_add_one_qubit_gate(t, T)

qgd_Circuit_Wrapper_add_one_qubit_gate(tdg, Tdg)

qgd_Circuit_Wrapper_add_two_qubit_gate(cnot, CNOT)

qgd_Circuit_Wrapper_add_two_qubit_gate(cz, CZ)

qgd_Circuit_Wrapper_add_two_qubit_gate(ch, CH)

qgd_Circuit_Wrapper_add_two_qubit_gate(cu, CU)

qgd_Circuit_Wrapper_add_two_qubit_gate(syc, SYC)

qgd_Circuit_Wrapper_add_two_qubit_gate(cry, CRY)

qgd_Circuit_Wrapper_add_two_qubit_gate(crz, CRZ)

qgd_Circuit_Wrapper_add_two_qubit_gate(crx, CRX)

// SWAP gate now uses vector-based interface
static PyObject *
qgd_Circuit_Wrapper_add_SWAP(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"target_qbits", NULL};
    PyObject* target_qbits_py = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &target_qbits_py))
        return Py_BuildValue("i", -1);

    if (target_qbits_py != NULL && PyList_Check(target_qbits_py)) {
        std::vector<int> target_qbits;
        Py_ssize_t list_size = PyList_Size(target_qbits_py);
        for (Py_ssize_t i = 0; i < list_size; i++) {
            PyObject* item = PyList_GetItem(target_qbits_py, i);
            target_qbits.push_back(PyLong_AsLong(item));
        }
        self->circuit->add_swap(target_qbits);
        
    }

    return Py_BuildValue("i", 0);
}

static PyObject *
qgd_Circuit_Wrapper_add_RXX(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"target_qbits", NULL};
    PyObject* target_qbits_py = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &target_qbits_py))
        return Py_BuildValue("i", -1);

    if (target_qbits_py != NULL && PyList_Check(target_qbits_py)) {
        std::vector<int> target_qbits;
        Py_ssize_t list_size = PyList_Size(target_qbits_py);
        for (Py_ssize_t i = 0; i < list_size; i++) {
            PyObject* item = PyList_GetItem(target_qbits_py, i);
            target_qbits.push_back(PyLong_AsLong(item));
        }
        self->circuit->add_rxx(target_qbits);

    }

    return Py_BuildValue("i", 0);
}

static PyObject *
qgd_Circuit_Wrapper_add_RYY(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"target_qbits", NULL};
    PyObject* target_qbits_py = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &target_qbits_py))
        return Py_BuildValue("i", -1);

    if (target_qbits_py != NULL && PyList_Check(target_qbits_py)) {
        std::vector<int> target_qbits;
        Py_ssize_t list_size = PyList_Size(target_qbits_py);
        for (Py_ssize_t i = 0; i < list_size; i++) {
            PyObject* item = PyList_GetItem(target_qbits_py, i);
            target_qbits.push_back(PyLong_AsLong(item));
        }
        self->circuit->add_ryy(target_qbits);

    }

    return Py_BuildValue("i", 0);
}

static PyObject *
qgd_Circuit_Wrapper_add_RZZ(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"target_qbits", NULL};
    PyObject* target_qbits_py = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &target_qbits_py))
        return Py_BuildValue("i", -1);

    if (target_qbits_py != NULL && PyList_Check(target_qbits_py)) {
        std::vector<int> target_qbits;
        Py_ssize_t list_size = PyList_Size(target_qbits_py);
        for (Py_ssize_t i = 0; i < list_size; i++) {
            PyObject* item = PyList_GetItem(target_qbits_py, i);
            target_qbits.push_back(PyLong_AsLong(item));
        }
        self->circuit->add_rzz(target_qbits);

    }

    return Py_BuildValue("i", 0);
}

qgd_Circuit_Wrapper_add_two_qubit_gate(cp, CP)

qgd_Circuit_Wrapper_add_two_qubit_gate(cr, CR)

qgd_Circuit_Wrapper_add_two_qubit_gate(crot, CROT)

qgd_Circuit_Wrapper_add_two_qubit_gate(adaptive, adaptive)

/**
@brief Wrapper function to add a CCX gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: target_qbit (int), control_qbits (list of ints)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_CCX(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit", (char*)"control_qbits", NULL};

    // initiate variables for input arguments
    int  target_qbit = -1;
    PyObject* control_qbits_py = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO", kwlist,
                                     &target_qbit, &control_qbits_py))
        return Py_BuildValue("i", -1);

    // adding CCX gate to the end of the gate structure
    if (target_qbit != -1 && control_qbits_py != NULL && PyList_Check(control_qbits_py)) {
        std::vector<int> control_qbits;
        Py_ssize_t list_size = PyList_Size(control_qbits_py);
        for (Py_ssize_t i = 0; i < list_size; i++) {
            PyObject* item = PyList_GetItem(control_qbits_py, i);
            if (PyLong_Check(item)) {
                control_qbits.push_back(PyLong_AsLong(item));
            }
        }
        if (control_qbits.size() >= 2) {
            self->circuit->add_ccx(target_qbit, control_qbits);
        }
    }

    return Py_BuildValue("i", 0);

}


/**
@brief Wrapper function to add a CSWAP gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: target_qbits (list of ints), control_qbits (list of ints)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_CSWAP(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbits", (char*)"control_qbits", NULL};

    // initiate variables for input arguments
    PyObject* target_qbits_py = NULL;
    PyObject* control_qbits_py = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist,
                                     &target_qbits_py, &control_qbits_py))
        return Py_BuildValue("i", -1);

    // adding CSWAP gate to the end of the gate structure
    if (target_qbits_py != NULL && PyList_Check(target_qbits_py) &&
        control_qbits_py != NULL && PyList_Check(control_qbits_py)) {

        std::vector<int> target_qbits;
        Py_ssize_t target_size = PyList_Size(target_qbits_py);
        for (Py_ssize_t i = 0; i < target_size; i++) {
            PyObject* item = PyList_GetItem(target_qbits_py, i);
            if (PyLong_Check(item)) {
                target_qbits.push_back(PyLong_AsLong(item));
            }
        }

        std::vector<int> control_qbits;
        Py_ssize_t control_size = PyList_Size(control_qbits_py);
        for (Py_ssize_t i = 0; i < control_size; i++) {
            PyObject* item = PyList_GetItem(control_qbits_py, i);
            if (PyLong_Check(item)) {
                control_qbits.push_back(PyLong_AsLong(item));
            }
        }

        if (target_qbits.size() >= 2 && control_qbits.size() >= 1) {
            self->circuit->add_cswap(target_qbits, control_qbits);
        }
    }

    return Py_BuildValue("i", 0);

}

/**
@brief Wrapper function to add a block of operations to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: Py_qgd_Circuit_Wrapper (PyObject)
Py_qgd_Circuit_Wrapper: an instance of qgd_Circuit_Wrapper containing the custom gate structure
*/
static PyObject *
qgd_Circuit_Wrapper_add_Circuit(qgd_Circuit_Wrapper *self, PyObject *args)
{

    // initiate variables for input arguments
    PyObject *Py_Circuit; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O",
                                     &Py_Circuit))
        return Py_BuildValue("i", -1);


    qgd_Circuit_Wrapper* qgd_op_block = (qgd_Circuit_Wrapper*) Py_Circuit;


    // adding general gate to the end of the gate structure
    self->circuit->add_gate( static_cast<Gate*>( qgd_op_block->circuit->clone() ) );

    return Py_BuildValue("i", 0);

}

#ifdef __DFE__

static PyObject*
DFEgateQGD_to_Python(DFEgate_kernel_type* DFEgates, int gatesNum)
{
    PyObject* o = PyList_New(0);
    for (int i = 0; i < gatesNum; i++) {
        PyList_Append(o, Py_BuildValue("iiibbbb", DFEgates[i].ThetaOver2, DFEgates[i].Phi,
            DFEgates[i].Lambda, DFEgates[i].target_qbit, DFEgates[i].control_qbit,
            DFEgates[i].gate_type, DFEgates[i].metadata));
    }
    delete [] DFEgates;
    return o;
}

static DFEgate_kernel_type*
DFEgatePython_to_QGD(PyObject* obj)
{
    Py_ssize_t gatesNum = PyList_Size(obj); //assert type is list
    DFEgate_kernel_type* DFEgates = new DFEgate_kernel_type[gatesNum];
    for (Py_ssize_t i = 0; i < gatesNum; i++) {
        PyObject* t = PyList_GetItem(obj, i);
        //assert type is tuple and PyTuple_Size(t) == 7
        DFEgates[i].ThetaOver2 = PyLong_AsLong(PyTuple_GetItem(t, 0));
        DFEgates[i].Phi = PyLong_AsLong(PyTuple_GetItem(t, 1));
        DFEgates[i].Lambda = PyLong_AsLong(PyTuple_GetItem(t, 2));
        DFEgates[i].target_qbit = PyLong_AsLong(PyTuple_GetItem(t, 3));
        DFEgates[i].control_qbit = PyLong_AsLong(PyTuple_GetItem(t, 4));
        DFEgates[i].gate_type = PyLong_AsLong(PyTuple_GetItem(t, 5));
        DFEgates[i].metadata = PyLong_AsLong(PyTuple_GetItem(t, 6));
    }
    return DFEgates;
}

static PyObject *
qgd_Circuit_Wrapper_convert_to_DFE_gates_with_derivates(qgd_Circuit_Wrapper *self, PyObject *args)
{
    bool only_derivates = false;
    PyArrayObject* parameters_mtx_np = NULL;

    if (!PyArg_ParseTuple(args, "|Ob",
                                     &parameters_mtx_np, &only_derivates))
        return Py_BuildValue("");

    if ( parameters_mtx_np == NULL ) {
        return Py_BuildValue("");
    }

    PyArrayObject* parameters_mtx = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_mtx_np, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(parameters_mtx) ) {
        std::cout << "parameters_mtx is not memory contiguous" << std::endl;
    }

    // create QGD version of the parameters_mtx
    Matrix_real parameters_mtx_mtx = numpy2matrix_real(parameters_mtx);
        
    int gatesNum = -1, gateSetNum = -1, redundantGateSets = -1;
    DFEgate_kernel_type* ret = self->circuit->convert_to_DFE_gates_with_derivates(parameters_mtx_mtx, gatesNum, gateSetNum, redundantGateSets, only_derivates);
    return Py_BuildValue("Oii", DFEgateQGD_to_Python(ret, gatesNum), gateSetNum, redundantGateSets);
}

static PyObject *
qgd_Circuit_Wrapper_adjust_parameters_for_derivation(qgd_Circuit_Wrapper *self, PyObject *args)
{
    int gatesNum = -1;
    PyObject* dfegates = NULL;
    if (!PyArg_ParseTuple(args, "|Oi",    
                                     &dfegates, &gatesNum))
        return Py_BuildValue("");
    int gate_idx = -1, gate_set_index = -1;
    DFEgate_kernel_type* dfegates_qgd = DFEgatePython_to_QGD(dfegates);
    self->circuit->adjust_parameters_for_derivation(dfegates_qgd, gatesNum, gate_idx, gate_set_index);    
    return Py_BuildValue("Oii", DFEgateQGD_to_Python(dfegates_qgd, gatesNum), gate_idx, gate_set_index);
}

/*static PyObject *
qgd_Circuit_Wrapper_convert_to_DFE_gates(qgd_Circuit_Wrapper *self, PyObject *args)
{
    PyObject* parameters_mtx_np = NULL;
    if (!PyArg_ParseTuple(args, "|O",
                                     &parameters_mtx_np))
        return Py_BuildValue("");

    if ( parameters_mtx_np == NULL ) return Py_BuildValue("");
    PyObject* parameters_mtx = PyArray_FROM_OTF(parameters_mtx_np, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(parameters_mtx) ) {
        std::cout << "parameters_mtx is not memory contiguous" << std::endl;
    }

    // create QGD version of the parameters_mtx
    Matrix_real parameters_mtx_mtx = numpy2matrix_real(parameters_mtx);
        
    int gatesNum = -1;
    DFEgate_kernel_type* ret = self->circuit->convert_to_DFE_gates(parameters_mtx_mtx, gatesNum);
    return DFEgateQGD_to_Python(ret, gatesNum);
}*/

static PyObject *
qgd_Circuit_Wrapper_convert_to_DFE_gates(qgd_Circuit_Wrapper *self, PyObject *args)
{
    int start_index = -1;
    PyArrayObject* parameters_mtx_np = NULL;
    PyObject* dfegates = NULL;

    if (!PyArg_ParseTuple(args, "|OOi",
                                     &parameters_mtx_np, &dfegates, &start_index))
        return Py_BuildValue("");


    if ( parameters_mtx_np == NULL ) {
        return Py_BuildValue("");
    }

    PyArrayObject* parameters_mtx = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)parameters_mtx_np, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(parameters_mtx) ) {
        std::cout << "parameters_mtx is not memory contiguous" << std::endl;
    }

    // create QGD version of the parameters_mtx
    Matrix_real parameters_mtx_mtx = numpy2matrix_real(parameters_mtx);
    DFEgate_kernel_type* dfegates_qgd = DFEgatePython_to_QGD(dfegates);

    Py_ssize_t gatesNum = PyList_Size(dfegates);

    self->circuit->convert_to_DFE_gates(parameters_mtx_mtx, dfegates_qgd, start_index);

    return DFEgateQGD_to_Python(dfegates_qgd, gatesNum);
}

#endif

/**
@brief Extract the optimized parameters and return the matrix representation of the gate circuit
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@param args A tuple of the input arguments: start_index (integer, optional) - the index of the first inverse gate
@return Returns a numpy array containing the matrix representation of the circuit
*/
static PyObject *
qgd_Circuit_Wrapper_get_Matrix( qgd_Circuit_Wrapper *self, PyObject *args ) {

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


    // get the C++ wrapper around the data
    Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );


    Matrix mtx = self->circuit->get_matrix( parameters_mtx );
    
    // convert to numpy array
    mtx.set_owner(false);
    PyObject *mtx_py = matrix_to_numpy( mtx );


    Py_DECREF(parameters_arr);

    return mtx_py;
}



/**
@brief Get the number of free parameters in the gate structure used for the decomposition
*/
static PyObject *
qgd_Circuit_Wrapper_get_Parameter_Num( qgd_Circuit_Wrapper *self ) {

    int parameter_num = self->circuit->get_parameter_num();

    return Py_BuildValue("i", parameter_num);
}



/**
@brief Apply the gate circuit operation on the input matrix
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@param args A tuple of the input arguments: parameters_arr (numpy array), unitary_arg (numpy array), parallel (int, optional)
@param kwds A tuple of keywords
@return Returns 0 on success
*/
static PyObject *
qgd_Circuit_Wrapper_apply_to( qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds ) {

    PyArrayObject * parameters_arr = NULL;
    PyArrayObject * unitary_arg = NULL;
    
    int parallel = 1;
    
    static char *kwlist[] = {(char*)"", (char*)"", (char*)"parallel", NULL};


    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i", kwlist, &parameters_arr, &unitary_arg, &parallel )) {
        PyErr_SetString(PyExc_Exception, "Unable to parse input");
        return NULL;
    } 
        
        

    if ( unitary_arg == NULL ) {
        PyErr_SetString(PyExc_Exception, "Input matrix was not given");
        return NULL;
    }


    if ( parameters_arr == NULL ) {
        PyErr_SetString(PyExc_Exception, "Parameters were not given");
        return NULL;
    }



    if ( PyArray_TYPE(parameters_arr) != NPY_DOUBLE ) {
        PyErr_SetString(PyExc_Exception, "Parameter vector should be real typed");
        return NULL;
    }
    
    
    if ( PyArray_TYPE(unitary_arg) != NPY_COMPLEX128 ) {
        PyErr_SetString(PyExc_Exception, "input matrix or state should be complex typed");
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



    PyArrayObject* unitary = (PyArrayObject*)PyArray_FROM_OTF( (PyObject*)unitary_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(unitary) ) {
        PyErr_SetString(PyExc_Exception, "input matrix is not memory contiguous");
        return NULL;
    }


    // create QGD version of the input matrix
    Matrix unitary_mtx = numpy2matrix(unitary);

    try {
        self->circuit->apply_to( parameters_mtx, unitary_mtx, parallel );
    }
    catch (std::string err) {
    
        Py_DECREF(parameters_arr);
        Py_DECREF(unitary);
    
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    catch(...) {
    
        Py_DECREF(parameters_arr);
        Py_DECREF(unitary);
    
        std::string err( "Invalid pointer to circuit class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    
    if (unitary_mtx.data != PyArray_DATA(unitary)) {
        memcpy(PyArray_DATA(unitary), unitary_mtx.data, unitary_mtx.size() * sizeof(QGD_Complex16));
    }

    Py_DECREF(parameters_arr);
    Py_DECREF(unitary);

    return Py_BuildValue("i", 0);
}



/**
@brief Wrapper function to evaluate the second Rényi entropy of a quantum circuit at a specific parameter set
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@param args A tuple of the input arguments: parameters_arr (numpy array, optional), input_state_arg (numpy array, optional), qubit_list_arg (list, optional)
@return Returns the second Rényi entropy value as a double, or -1 on error
*/
static PyObject *
qgd_Circuit_Wrapper_get_Second_Renyi_Entropy( qgd_Circuit_Wrapper *self, PyObject *args)
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
        PyErr_SetString(PyExc_Exception, "input matrix is not memory contiguous");
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
        entropy = self->circuit->get_second_Renyi_entropy( parameters_mtx, input_state_mtx, qbit_list_mtx );
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


    Py_DECREF(parameters_arr);
    Py_DECREF(input_state);



    PyObject* p = Py_BuildValue("d", entropy);

    return p;
}



/**
@brief Call to retrieve the number of qubits in the circuit
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@return Returns the number of qubits as an integer
*/
static PyObject *
qgd_Circuit_Wrapper_get_Qbit_Num( qgd_Circuit_Wrapper *self ) {

    int qbit_num = 0;

    try {
        qbit_num = self->circuit->get_qbit_num();
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        std::cout << err << std::endl;
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to circuit class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    return Py_BuildValue("i", qbit_num );
    
}




/**
@brief Call to set the number of qubits in the circuit
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@param args A tuple of the input arguments: qbit_num (integer, optional)
@return Returns None on success
*/
static PyObject *
qgd_Circuit_Wrapper_set_Qbit_Num( qgd_Circuit_Wrapper *self,  PyObject *args ) {

    int qbit_num = 0;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|i", &qbit_num )) {
        std::string err( "Unable to parse arguments");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    try {
        self->circuit->set_qbit_num( qbit_num );
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        std::cout << err << std::endl;
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to circuit class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    return Py_BuildValue("");
    
}



/**
@brief Call to retrieve the list of qubits involved in the circuit
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@return Returns a Python list containing the qubit indices involved in the circuit
*/
static PyObject *
qgd_Circuit_Wrapper_get_Qbits( qgd_Circuit_Wrapper *self ) {

    PyObject* ret = PyList_New(0);

    try {
        std::vector<int>&& qbits =  self->circuit->get_involved_qubits();
        for (size_t idx = 0; idx < qbits.size(); idx++) {
            PyList_Append(ret, Py_BuildValue("i", qbits[idx] ) );
        }

    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        std::cout << err << std::endl;
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to circuit class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    return Py_BuildValue("O", ret );
    
}


static PyObject *
qgd_Circuit_Wrapper_set_Min_Fusion( qgd_Circuit_Wrapper *self,  PyObject *args ) {

    int min_fusion = -1;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|i", &min_fusion )) {
        std::string err( "Unable to parse arguments");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    try {
        self->circuit->set_min_fusion( min_fusion );
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        std::cout << err << std::endl;
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to circuit class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    return Py_BuildValue("");
    
}


/**
@brief Call to remap the qubits in the circuit
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@param args A tuple of the input arguments: qbit_map_arg (dictionary)
@return Returns 0 on success, -1 on error
*/
static PyObject *
qgd_Circuit_Wrapper_Remap_Qbits( qgd_Circuit_Wrapper *self, PyObject *args ) {


    PyObject* qbit_map_arg = NULL;
    int qbit_num = 0;


    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|Oi", &qbit_map_arg, &qbit_num )) 
        return Py_BuildValue("i", -1);


    // parse qbit map and create C++ version of the map

    bool is_dict = (PyDict_Check( qbit_map_arg ) != 0);
    if (!is_dict) {
        printf("Qubit map object must be a python dictionary!\n");
        return Py_BuildValue("i", -1);
    }

    // integer type config metadata utilized during the optimization
    std::map<int, int> qbit_map;


    // keys and values of the config dict (borrowed references)
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(qbit_map_arg, &pos, &key, &value)) {
       

        if ( PyLong_Check( value ) && PyLong_Check( key ) ) { 
            int key_Cpp = (int)PyLong_AsLongLong( key );
            qbit_map[ key_Cpp ] = (int)PyLong_AsLongLong( value );
        }
        else {
            std::string err( "Key and value in the qbit_map should be integers");
            PyErr_SetString(PyExc_Exception, err.c_str());
            return NULL;
        }

    }


    Gates_block* remapped_circuit = NULL;

    try {
        remapped_circuit = self->circuit->create_remapped_circuit( qbit_map, qbit_num);

    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        std::cout << err << std::endl;
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to circuit class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }



        // import gate operation modules
        PyObject* qgd_circuit  = PyImport_ImportModule("squander.gates.qgd_Circuit");

        if ( qgd_circuit == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_Circuit" );
            return NULL;
        }

        PyObject* qgd_circuit_Dict  = PyModule_GetDict( qgd_circuit );    

        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_circuit_class = PyDict_GetItemString( qgd_circuit_Dict, "qgd_Circuit");

        PyObject* circuit_input = Py_BuildValue("(O)", Py_BuildValue("i", qbit_num) );
        PyObject* py_circuit    = PyObject_CallObject(py_circuit_class, circuit_input);


        // replace dummy data with real gate data
        qgd_Circuit_Wrapper* py_circuit_C = reinterpret_cast<qgd_Circuit_Wrapper*>( py_circuit );

        delete( py_circuit_C->circuit );
        py_circuit_C->circuit = remapped_circuit;

        Py_DECREF( qgd_circuit );            
        Py_DECREF( circuit_input );


    return py_circuit;
    
}


#define get_gate_template_two_qubit(GATE_NAME) \
    else if (gate->get_type() == GATE_NAME##_OPERATION) { \
        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate ); \
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, #GATE_NAME); \
        PyObject* gate_input = Py_BuildValue("(OOO)", qbit_num, target_qbit, control_qbit); \
        py_gate              = PyObject_CallObject(py_gate_class, gate_input); \
        qgd_Gate* py_gate_C = reinterpret_cast<qgd_Gate*>( py_gate ); \
        delete( py_gate_C->gate ); \
        py_gate_C->gate = static_cast<Gate*>( gate->clone() ); \
        Py_DECREF( qgd_gate ); \
        Py_DECREF( gate_input ); \
    }

#define get_gate_template_one_qubit(GATE_NAME) \
    else if (gate->get_type() == GATE_NAME##_OPERATION) { \
        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate ); \
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, #GATE_NAME); \
        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit); \
        py_gate              = PyObject_CallObject(py_gate_class, gate_input); \
        qgd_Gate* py_gate_C = reinterpret_cast<qgd_Gate*>( py_gate ); \
        delete( py_gate_C->gate ); \
        py_gate_C->gate = static_cast<Gate*>( gate->clone() ); \
        Py_DECREF( qgd_gate ); \
        Py_DECREF( gate_input ); \
    }


/**
@brief Call to get the metadata organized into Python dictionary of the idx-th gate
@param circuit A pointer pointing to an instance of the class Gates_block
@param idx Labels the idx-th gate (passed by reference, may be modified)
@return Returns a Python dictionary containing the metadata of the idx-th gate, or NULL on error
*/
static PyObject *
get_gate( Gates_block* circuit, int &idx ) {


    Gate* gate = circuit->get_gate( idx );

    // create dummy gate parameters to instantiate dummy object, which are then filled up with valid data
    PyObject* qbit_num     = Py_BuildValue("i",  gate->get_qbit_num() );
    PyObject* target_qbit  = Py_BuildValue("i",  gate->get_target_qbit() );
    PyObject* control_qbit = Py_BuildValue("i",  gate->get_control_qbit() );

    // The python instance of the gate
    PyObject* py_gate = NULL;

    // import gate operation modules
    PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.gates_Wrapper");

    if ( qgd_gate == NULL ) {
        PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.gates_Wrapper" );
        return NULL;
    }
    if (gate->get_type() == CNOT_OPERATION) {


        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "CNOT"); 

        PyObject* gate_input = Py_BuildValue("(OOO)", qbit_num, target_qbit, control_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_Gate* py_gate_C = reinterpret_cast<qgd_Gate*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<Gate*>( gate->clone() );

        Py_DECREF( qgd_gate );        
        Py_DECREF( gate_input );


    }
    get_gate_template_two_qubit(CZ)
    get_gate_template_two_qubit(CH)
    get_gate_template_two_qubit(CU)
    get_gate_template_two_qubit(SYC)
    get_gate_template_two_qubit(CRY)
    get_gate_template_two_qubit(CRX)
    get_gate_template_two_qubit(CRZ)
    get_gate_template_two_qubit(CR)
    get_gate_template_two_qubit(CROT)
    get_gate_template_two_qubit(CP)
    else if (gate->get_type() == SWAP_OPERATION){
        // SWAP now uses vector-based interface
        std::vector<int> target_qbits_vec = gate->get_target_qbits();
        PyObject* target_qbits_list = PyList_New((Py_ssize_t)target_qbits_vec.size());
        for (size_t i = 0; i < target_qbits_vec.size(); i++) {
            PyList_SetItem(target_qbits_list, (Py_ssize_t)i, Py_BuildValue("i", target_qbits_vec[i]));
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "SWAP");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbits_list);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_Gate* py_gate_C = reinterpret_cast<qgd_Gate*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<Gate*>( gate->clone() );

        Py_DECREF( qgd_gate );
        Py_DECREF( gate_input );
        Py_DECREF( target_qbits_list );
    }

    else if (gate->get_type() == RXX_OPERATION){
        // RXX now uses vector-based interface
        std::vector<int> target_qbits_vec = gate->get_target_qbits();
        PyObject* target_qbits_list = PyList_New((Py_ssize_t)target_qbits_vec.size());
        for (size_t i = 0; i < target_qbits_vec.size(); i++) {
            PyList_SetItem(target_qbits_list, (Py_ssize_t)i, Py_BuildValue("i", target_qbits_vec[i]));
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "RXX");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbits_list);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_Gate* py_gate_C = reinterpret_cast<qgd_Gate*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<Gate*>( gate->clone() );

        Py_DECREF( qgd_gate );
        Py_DECREF( gate_input );
        Py_DECREF( target_qbits_list );
    }

    else if (gate->get_type() == RYY_OPERATION){
        // RYY uses vector-based interface
        std::vector<int> target_qbits_vec = gate->get_target_qbits();
        PyObject* target_qbits_list = PyList_New((Py_ssize_t)target_qbits_vec.size());
        for (size_t i = 0; i < target_qbits_vec.size(); i++) {
            PyList_SetItem(target_qbits_list, (Py_ssize_t)i, Py_BuildValue("i", target_qbits_vec[i]));
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "RYY");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbits_list);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_Gate* py_gate_C = reinterpret_cast<qgd_Gate*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<Gate*>( gate->clone() );

        Py_DECREF( qgd_gate );
        Py_DECREF( gate_input );
        Py_DECREF( target_qbits_list );
    }

    else if (gate->get_type() == RZZ_OPERATION){
        // RZZ uses vector-based interface
        std::vector<int> target_qbits_vec = gate->get_target_qbits();
        PyObject* target_qbits_list = PyList_New((Py_ssize_t)target_qbits_vec.size());
        for (size_t i = 0; i < target_qbits_vec.size(); i++) {
            PyList_SetItem(target_qbits_list, (Py_ssize_t)i, Py_BuildValue("i", target_qbits_vec[i]));
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "RZZ");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbits_list);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_Gate* py_gate_C = reinterpret_cast<qgd_Gate*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<Gate*>( gate->clone() );

        Py_DECREF( qgd_gate );
        Py_DECREF( gate_input );
        Py_DECREF( target_qbits_list );
    }

    get_gate_template_one_qubit(U1)
    get_gate_template_one_qubit(U2)
    get_gate_template_one_qubit(U3)
    get_gate_template_one_qubit(RX)
    get_gate_template_one_qubit(RY)
    get_gate_template_one_qubit(RZ)
    get_gate_template_one_qubit(R)
    get_gate_template_one_qubit(H)
    get_gate_template_one_qubit(X)
    get_gate_template_one_qubit(Y)
    get_gate_template_one_qubit(Z)
    get_gate_template_one_qubit(T)
    get_gate_template_one_qubit(S)
    get_gate_template_one_qubit(SX)
    else if (gate->get_type() == SDG_OPERATION){


        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "Sdg");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_Gate* py_gate_C = reinterpret_cast<qgd_Gate*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<Gate*>( gate->clone() );

        Py_DECREF( qgd_gate );               
        Py_DECREF( gate_input );


    }
    else if (gate->get_type() == SXDG_OPERATION){


        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "SXdg");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_Gate* py_gate_C = reinterpret_cast<qgd_Gate*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<Gate*>( gate->clone() );

        Py_DECREF( qgd_gate );               
        Py_DECREF( gate_input );


    }
    else if (gate->get_type() == TDG_OPERATION){


        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "Tdg");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_Gate* py_gate_C = reinterpret_cast<qgd_Gate*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<Gate*>( gate->clone() );

        Py_DECREF( qgd_gate );               
        Py_DECREF( gate_input );


    }
    else if (gate->get_type() == CCX_OPERATION){
        // CCX now uses vector-based interface
        std::vector<int> control_qbits_vec = gate->get_control_qbits();
        PyObject* control_qbits_list = PyList_New((Py_ssize_t)control_qbits_vec.size());
        for (size_t i = 0; i < control_qbits_vec.size(); i++) {
            PyList_SetItem(control_qbits_list, (Py_ssize_t)i, Py_BuildValue("i", control_qbits_vec[i]));
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "CCX");

        PyObject* gate_input = Py_BuildValue("(OOO)", qbit_num, target_qbit, control_qbits_list);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_Gate* py_gate_C = reinterpret_cast<qgd_Gate*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<Gate*>( gate->clone() );

        Py_DECREF( qgd_gate );
        Py_DECREF( gate_input );
        Py_DECREF( control_qbits_list );

        Py_XDECREF(qbit_num);
        Py_XDECREF(target_qbit);
        Py_XDECREF(control_qbit);

        return py_gate;

    }
    else if (gate->get_type() == CSWAP_OPERATION){
        // CSWAP now uses vector-based interface
        std::vector<int> target_qbits_vec = gate->get_target_qbits();
        PyObject* target_qbits_list = PyList_New((Py_ssize_t)target_qbits_vec.size());
        for (size_t i = 0; i < target_qbits_vec.size(); i++) {
            PyList_SetItem(target_qbits_list, (Py_ssize_t)i, Py_BuildValue("i", target_qbits_vec[i]));
        }

        std::vector<int> control_qbits_vec = gate->get_control_qbits();
        PyObject* control_qbits_list = PyList_New((Py_ssize_t)control_qbits_vec.size());
        for (size_t i = 0; i < control_qbits_vec.size(); i++) {
            PyList_SetItem(control_qbits_list, (Py_ssize_t)i, Py_BuildValue("i", control_qbits_vec[i]));
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "CSWAP");

        PyObject* gate_input = Py_BuildValue("(OOO)", qbit_num, target_qbits_list, control_qbits_list);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_Gate* py_gate_C = reinterpret_cast<qgd_Gate*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<Gate*>( gate->clone() );

        Py_DECREF( qgd_gate );
        Py_DECREF( gate_input );
        Py_DECREF( target_qbits_list );
        Py_DECREF( control_qbits_list );

        Py_XDECREF(qbit_num);
        Py_XDECREF(target_qbit);
        Py_XDECREF(control_qbit);

        return py_gate;

    }
    else if (gate->get_type() == BLOCK_OPERATION) {

        // import gate operation modules
        PyObject* qgd_circuit  = PyImport_ImportModule("squander.gates.qgd_Circuit");

        if ( qgd_circuit == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_Circuit" );
            return NULL;
        }

        PyObject* qgd_circuit_Dict  = PyModule_GetDict( qgd_circuit );    

        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_circuit_class = PyDict_GetItemString( qgd_circuit_Dict, "qgd_Circuit");

        PyObject* circuit_input = Py_BuildValue("(O)", qbit_num );
        py_gate    = PyObject_CallObject(py_circuit_class, circuit_input);

        // replace dummy data with real gate data
        qgd_Circuit_Wrapper* py_gate_C = reinterpret_cast<qgd_Circuit_Wrapper*>( py_gate );

        Gates_block* circuit = reinterpret_cast<Gates_block*>( gate );
        delete( py_gate_C->circuit );
        py_gate_C->circuit = circuit->clone();

        Py_DECREF( qgd_circuit );            
        Py_DECREF( circuit_input );

    }
    else {

            Py_DECREF( qgd_gate );    
            Py_XDECREF(qbit_num);
            Py_XDECREF(target_qbit);
            Py_XDECREF(control_qbit);
            PyErr_SetString(PyExc_Exception, "qgd_Circuit_Wrapper::get_gate: unimplemented gate type" );
            return NULL;
    }

    Py_XDECREF(qbit_num);
    Py_XDECREF(target_qbit);
    Py_XDECREF(control_qbit);

    return py_gate;

}



/**
@brief Wrapper function to get a gate from the circuit
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@param args A tuple of the input arguments: idx (integer) - the index of the gate to retrieve
@return Returns a Python dictionary containing the gate metadata, or a gate object
*/
static PyObject *
qgd_Circuit_Wrapper_get_gate( qgd_Circuit_Wrapper *self, PyObject *args ) {

    // initiate variables for input arguments
    int  idx; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|i", &idx )) return Py_BuildValue("i", -1);


    return get_gate( self->circuit, idx );


}


/**
@brief Call to get the counts of individual gates in the circuit
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@return Returns a Python dictionary mapping gate type names to their counts in the circuit
*/
static PyObject *
qgd_Circuit_Wrapper_get_Gate_Nums( qgd_Circuit_Wrapper *self ) {

    std::map< std::string, int > gate_nums;
    
    try {
        gate_nums = self->circuit->get_gate_nums();
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


    PyObject* gate_nums_py = PyDict_New();
    if( gate_nums_py == NULL ) {
        std::string err( "Failed to create dictionary");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;    
    }
    
    for( auto it = gate_nums.begin(); it != gate_nums.end(); it++ ) {
    
        PyObject* key = Py_BuildValue( "s", it->first.c_str() );
        PyObject* val = Py_BuildValue("i", it->second );
    
        PyDict_SetItem(gate_nums_py, key, val);
    }
    
    return gate_nums_py;

}




/**
@brief Call to get the incorporated gates in a Python list
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@return Returns a Python list containing gate objects representing all gates in the circuit
*/
static PyObject *
qgd_Circuit_Wrapper_get_gates( qgd_Circuit_Wrapper *self ) {




    // get the number of gates
    int op_num = self->circuit->get_gate_num();

    // preallocate Python tuple for the output
    PyObject* ret = PyTuple_New( (Py_ssize_t) op_num );



    // iterate over the gates to get the gate list
    for (int idx = 0; idx < op_num; idx++ ) {

        // get metadata about the idx-th gate
        PyObject* gate = get_gate( self->circuit, idx );

        // adding gate information to the tuple
        PyTuple_SetItem( ret, (Py_ssize_t) idx, gate );

    }


    return ret;

}



/**
@brief Wrapper function to get the indices of parent gates
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@param args A tuple of the input arguments: idx (integer) - the index of the gate for which we are retrieving the parents
@return Returns a Python list containing the indices of parent gates
*/
static PyObject *
qgd_Circuit_Wrapper_get_parents( qgd_Circuit_Wrapper *self, PyObject *args ) {

    // the gate for which we look for the parents
    PyObject* py_gate = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &py_gate )) return Py_BuildValue("i", -1);


    if( py_gate == NULL ) {
        return PyTuple_New( 0 );
    }

    qgd_Gate* gate_struct = reinterpret_cast<qgd_Gate*>( py_gate );
    std::vector<Gate*> parents = gate_struct->gate->get_parents();

    // preallocate tuple for the output 
    PyObject* parent_tuple = PyTuple_New( (Py_ssize_t) parents.size() );

    std::vector<Gate*>&& gates = self->circuit->get_gates();


    // find the indices of the parents
    for(size_t idx=0; idx<parents.size(); idx++) {

        Gate* parent_gate = parents[idx];

        // find the index of the parent_gate
        int parent_idx = -1;
        for( size_t jdx=0; jdx<gates.size(); jdx++ ) {

            Gate* gate = gates[jdx];

            if( parent_gate == gate ) {
                parent_idx = static_cast<int>(jdx);
                break;
            }

            if( jdx == static_cast<size_t>(gates.size()-1) ) {
                std::string err( "Parent gate did not found in the circuit. May be the gate is not in the circuit");
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }

        }

        // adding parent_idx the tuple
        PyTuple_SetItem( parent_tuple, (Py_ssize_t) idx, Py_BuildValue("i", parent_idx) );
        
       
    }


    return parent_tuple;


}




/**
@brief Wrapper function to get the indices of children gates
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@param args A tuple of the input arguments: idx (integer) - the index of the gate for which we are retrieving the children
@return Returns a Python list containing the indices of children gates
*/
static PyObject *
qgd_Circuit_Wrapper_get_children( qgd_Circuit_Wrapper *self, PyObject *args ) {

    // the gate for which we look for the children
    PyObject* py_gate = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &py_gate )) return Py_BuildValue("i", -1);


    if( py_gate == NULL ) {
        return PyTuple_New( 0 );
    }

    qgd_Gate* gate_struct = reinterpret_cast<qgd_Gate*>( py_gate );
    std::vector<Gate*> children = gate_struct->gate->get_children();

    // preallocate tuple for the output 
    PyObject* children_tuple = PyTuple_New( (Py_ssize_t) children.size() );

    std::vector<Gate*>&& gates = self->circuit->get_gates();


    // find the indices of the children
    for(size_t idx=0; idx<children.size(); idx++) {

        Gate* child_gate = children[idx];

        // find the index of the child_gate
        int child_idx = -1;
        for( size_t jdx=0; jdx<gates.size(); jdx++ ) {

            Gate* gate = gates[jdx];

            if( child_gate == gate ) {
                child_idx = static_cast<int>(jdx);
                break;
            }

            if( jdx == static_cast<size_t>(gates.size()-1) ) {
                std::string err( "Child gate did not found in the circuit. May be the gate is not in the circuit");
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }

        }

        // adding child_idx the tuple
        PyTuple_SetItem( children_tuple, (Py_ssize_t) idx, Py_BuildValue("i", child_idx) );
        
       
    }


    return children_tuple;


}



/**
@brief Call to extract the parameters corresponding to the gate from a parameter array associated with the circuit in which the gate is embedded
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@param args A tuple of the input arguments: parameters_arr (numpy array, optional)
@return Returns a numpy array containing the extracted parameters, or -1 on error
*/
static PyObject *
qgd_Circuit_Wrapper_Extract_Parameters( qgd_Circuit_Wrapper *self, PyObject *args ) {

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

    // get the C++ wrapper around the data
    Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );


    
    Matrix_real extracted_parameters;

    try {
        extracted_parameters = self->circuit->extract_parameters( parameters_mtx );
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


    // convert to numpy array
    extracted_parameters.set_owner(false);
    PyObject *extracted_parameters_py = matrix_real_to_numpy( extracted_parameters );
   

    return extracted_parameters_py;
}



/**
@brief Method to generate a flat circuit. A flat circuit does not contain subcircuits: there are no Gates_block instances (containing subcircuits) in the resulting circuit. If the original circuit contains subcircuits, the gates in the subcircuits are directly incorporated in the resulting flat circuit
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@return Returns a new qgd_Circuit_Wrapper instance representing the flat circuit
*/
static PyObject *
qgd_Circuit_Wrapper_get_Flat_Circuit( qgd_Circuit_Wrapper *self ) {

    Gates_block* flat_circuit = self->circuit->get_flat_circuit();
    int qbit_num = flat_circuit->get_qbit_num();

    // import gate operation modules
    PyObject* qgd_circuit  = PyImport_ImportModule("squander.gates.qgd_Circuit");

    if ( qgd_circuit == NULL ) {
        PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_Circuit" );
        return NULL;
    }

    PyObject* qgd_circuit_Dict  = PyModule_GetDict( qgd_circuit );    

    // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
    PyObject* py_circuit_class = PyDict_GetItemString( qgd_circuit_Dict, "qgd_Circuit");

    PyObject* circuit_input = Py_BuildValue("(O)", Py_BuildValue("i", qbit_num) );
    PyObject* py_circuit    = PyObject_CallObject(py_circuit_class, circuit_input);

    // replace dummy data with real gate data
    qgd_Circuit_Wrapper* py_circuit_C = reinterpret_cast<qgd_Circuit_Wrapper*>( py_circuit );
    delete( py_circuit_C->circuit );
    py_circuit_C->circuit = flat_circuit;


    Py_DECREF( qgd_circuit  );                
    Py_DECREF( circuit_input );
  
    return py_circuit;
}




/**
@brief Method to extract the stored quantum circuit in a human-readable data serialized and pickle-able format
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@return Returns a Python dictionary containing the serialized circuit state
*/
static PyObject *
qgd_Circuit_Wrapper_getstate( qgd_Circuit_Wrapper *self ) {


    // get the number of gates
    int op_num = self->circuit->get_gate_num();

    // preallocate Python tuple for the output
    PyObject* ret = PyTuple_New( (Py_ssize_t) op_num+1 );




    // add qbit num value to the return tuple
    PyObject* qbit_num_dict = PyDict_New();

    if( qbit_num_dict == NULL ) {
        std::string err( "Failed to create dictionary");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;    
    }
    
    int qbit_num = self->circuit->get_qbit_num();
    PyObject* qbit_num_key = Py_BuildValue( "s", "qbit_num" );
    PyObject* qbit_num_val = Py_BuildValue("i", qbit_num );    

    PyDict_SetItem(qbit_num_dict, qbit_num_key, qbit_num_val);

    PyTuple_SetItem( ret, 0, qbit_num_dict );

    Py_DECREF( qbit_num_key );
    Py_DECREF( qbit_num_val );
    //Py_DECREF( qbit_num_dict );


    PyObject* method_name = Py_BuildValue("s", "__getstate__");

    // iterate over the gates to get the gate list
    for (int idx = 0; idx < op_num; idx++ ) {

        // get metadata about the idx-th gate
        PyObject* gate = get_gate( self->circuit, idx );

        PyObject* gate_state  = PyObject_CallMethodObjArgs( gate, method_name, NULL );   


        // remove the field qbit_num from gate dict sice this will be redundant information
        if ( PyDict_Contains(gate_state, qbit_num_key) == 1 ) {

            if ( PyDict_DelItem(gate_state, qbit_num_key) != 0 ) {
                std::string err( "Failed to delete item qbit_num from gate state");
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;    
            }

        }


        // adding gate information to the tuple
        PyTuple_SetItem( ret, (Py_ssize_t) idx+1, gate_state );


        
        Py_DECREF( gate );
        //Py_DECREF( gate_state );

    }

    Py_DECREF( method_name );
    
    return ret;
}



/**
@brief Call to set the state of a quantum circuit from a human-readable data serialized and pickle-able format
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@param args A tuple of the input arguments: state (dictionary) - the serialized circuit state
@return Returns None on success
*/
static PyObject *
qgd_Circuit_Wrapper_setstate( qgd_Circuit_Wrapper *self, PyObject *args ) {

    PyObject* state = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O", &state )) {
        std::string err( "Unable to parse state argument");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;    
    }

    if ( PyTuple_Size(state) == 0 ) {
        std::string err( "State should contain at least one element");
        PyErr_SetString(PyExc_Exception, err.c_str());

        Py_DECREF( state );
        return NULL;
    }

    PyObject* qbit_num_dict = PyTuple_GetItem( state, 0); // borrowed reference

    
    PyObject* qbit_num_key = Py_BuildValue( "s", "qbit_num" );

    if ( PyDict_Contains(qbit_num_dict, qbit_num_key) == 0 ) {
        std::string err( "The first entry of the circuit state should be the number of qubits");
        PyErr_SetString(PyExc_Exception, err.c_str());

        Py_DECREF( qbit_num_key );
        Py_DECREF( state );
        return NULL;
    }

    PyObject* qbit_num_py = PyDict_GetItem(qbit_num_dict, qbit_num_key); // borrowed reference

    if( !PyLong_Check(qbit_num_py) ) {
        std::string err( "The number of qubits should be an integer value");
        PyErr_SetString(PyExc_Exception, err.c_str());

        Py_DECREF( qbit_num_key );
        Py_DECREF( state );
        return NULL;
    } 


    int qbit_num = (int)PyLong_AsLong( qbit_num_py );


    // import gate operation modules
    PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.gates_Wrapper");
    
    if ( qgd_gate == NULL ) {
        PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.gates_Wrapper" );
        Py_DECREF( qbit_num_key );
        Py_DECREF( state );
        return NULL;
    }

    PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate ); // borrowed reference ???
    PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "Gate");  // borrowed reference 
    PyObject* setstate_name = Py_BuildValue( "s", "__setstate__" );
    PyObject* dummy_target_qbit = Py_BuildValue( "i", 0 );

    // now build up the quantum circuit
    try {

        self->circuit->release_gates();
        self->circuit->set_qbit_num( qbit_num );

        int gates_idx_max = (int) PyTuple_Size(state);

        for( int gate_idx=1; gate_idx < gates_idx_max; gate_idx++ ) {


            // get gate state as python dictionary
            PyObject* gate_state_dict = PyTuple_GetItem( state, gate_idx); // borrowed reference 

            if( !PyDict_Check( gate_state_dict ) ) {
                std::string err( "Gate state should be given by a dictionary");
                PyErr_SetString(PyExc_Exception, err.c_str());

                Py_DECREF( qgd_gate );
                Py_DECREF( qgd_gate_Dict );
                Py_DECREF( qbit_num_key );
                Py_DECREF( state );
                Py_DECREF( setstate_name );
                Py_DECREF( dummy_target_qbit );
                return NULL;
            }   

            PyDict_SetItem(gate_state_dict, qbit_num_key, qbit_num_py);  

            PyObject* gate_input = Py_BuildValue( "(O)", qbit_num_py );
            PyObject* py_gate    = PyObject_CallObject(py_gate_class, gate_input);

            // turn the generic gate into a specific gate
            PyObject_CallMethodObjArgs( py_gate, setstate_name, gate_state_dict, NULL );
            
            Gate* gate_loc = static_cast<Gate*>( ((qgd_Gate*)py_gate)->gate->clone() );
            self->circuit->add_gate( gate_loc );
            
            
            Py_DECREF( gate_input );
            Py_DECREF( py_gate );



        }


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




    Py_DECREF( qgd_gate );    
    //Py_DECREF( qgd_gate_Dict );
    Py_DECREF( qbit_num_key );
    Py_DECREF( setstate_name );
    Py_DECREF( dummy_target_qbit );
    

    return Py_BuildValue("");
}



/**
@brief Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper
@return Returns the starting index as an integer
*/
static PyObject *
qgd_Circuit_Wrapper_get_Parameter_Start_Index( qgd_Circuit_Wrapper *self ) {

    int start_index = self->circuit->get_parameter_start_idx();

    return Py_BuildValue("i", start_index);

}


static PyMethodDef qgd_Circuit_Wrapper_Methods[] = {
    {"add_U1", (PyCFunction) qgd_Circuit_Wrapper_add_U1, METH_VARARGS | METH_KEYWORDS,
     "Call to add a U1 gate to the front of the gate structure"
    },
    {"add_U2", (PyCFunction) qgd_Circuit_Wrapper_add_U2, METH_VARARGS | METH_KEYWORDS,
     "Call to add a U2 gate to the front of the gate structure"
    },
    {"add_U3", (PyCFunction) qgd_Circuit_Wrapper_add_U3, METH_VARARGS | METH_KEYWORDS,
     "Call to add a U3 gate to the front of the gate structure"
    },
    {"add_RX", (PyCFunction) qgd_Circuit_Wrapper_add_RX, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RX gate to the front of the gate structure"
    },
    {"add_RXX", (PyCFunction) qgd_Circuit_Wrapper_add_RXX, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RXX gate to the front of the gate structure"
    },
    {"add_RYY", (PyCFunction) qgd_Circuit_Wrapper_add_RYY, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RYY gate to the front of the gate structure"
    },
    {"add_RZZ", (PyCFunction) qgd_Circuit_Wrapper_add_RZZ, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RZZ gate to the front of the gate structure"
    },
    {"add_R", (PyCFunction) qgd_Circuit_Wrapper_add_R, METH_VARARGS | METH_KEYWORDS,
     "Call to add a R gate to the front of the gate structure"
    },
    {"add_RY", (PyCFunction) qgd_Circuit_Wrapper_add_RY, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RY gate to the front of the gate structure"
    },
    {"add_RZ", (PyCFunction) qgd_Circuit_Wrapper_add_RZ, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RZ gate to the front of the gate structure"
    },
    {"add_CNOT", (PyCFunction) qgd_Circuit_Wrapper_add_CNOT, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CNOT gate to the front of the gate structure"
    },
    {"add_CZ", (PyCFunction) qgd_Circuit_Wrapper_add_CZ, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CZ gate to the front of the gate structure"
    },
    {"add_CU", (PyCFunction) qgd_Circuit_Wrapper_add_CU, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CU gate to the front of the gate structure"
    },
    {"add_CH", (PyCFunction) qgd_Circuit_Wrapper_add_CH, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CH gate to the front of the gate structure"
    },
    {"add_SYC", (PyCFunction) qgd_Circuit_Wrapper_add_SYC, METH_VARARGS | METH_KEYWORDS,
     "Call to add a Sycamore gate to the front of the gate structure"
    },
    {"add_H", (PyCFunction) qgd_Circuit_Wrapper_add_H, METH_VARARGS | METH_KEYWORDS,
     "Call to add a Hadamard gate to the front of the gate structure"
    },
    {"add_X", (PyCFunction) qgd_Circuit_Wrapper_add_X, METH_VARARGS | METH_KEYWORDS,
     "Call to add a X gate to the front of the gate structure"
    },
    {"add_Y", (PyCFunction) qgd_Circuit_Wrapper_add_Y, METH_VARARGS | METH_KEYWORDS,
     "Call to add a Y gate to the front of the gate structure"
    },
    {"add_Z", (PyCFunction) qgd_Circuit_Wrapper_add_Z, METH_VARARGS | METH_KEYWORDS,
     "Call to add a Z gate to the front of the gate structure"
    },
    {"add_SX", (PyCFunction) qgd_Circuit_Wrapper_add_SX, METH_VARARGS | METH_KEYWORDS,
     "Call to add a SX gate to the front of the gate structure"
    },
    {"add_SXdg", (PyCFunction) qgd_Circuit_Wrapper_add_SXdg, METH_VARARGS | METH_KEYWORDS,
     "Call to add a SXdg gate to the front of the gate structure"
    },
    {"add_S", (PyCFunction) qgd_Circuit_Wrapper_add_S, METH_VARARGS | METH_KEYWORDS,
     "Call to add a S gate to the front of the gate structure"
    },
    {"add_Sdg", (PyCFunction) qgd_Circuit_Wrapper_add_Sdg, METH_VARARGS | METH_KEYWORDS,
     "Call to add a Sdg gate to the front of the gate structure"
    },
    {"add_T", (PyCFunction) qgd_Circuit_Wrapper_add_T, METH_VARARGS | METH_KEYWORDS,
     "Call to add a T gate to the front of the gate structure"
    },
    {"add_Tdg", (PyCFunction) qgd_Circuit_Wrapper_add_Tdg, METH_VARARGS | METH_KEYWORDS,
     "Call to add a Tdg gate to the front of the gate structure"
    },
    {"add_CRY", (PyCFunction) qgd_Circuit_Wrapper_add_CRY, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CRY gate to the front of the gate structure"
    },
    {"add_CRX", (PyCFunction) qgd_Circuit_Wrapper_add_CRX, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CRY gate to the front of the gate structure"
    },
    {"add_CRZ", (PyCFunction) qgd_Circuit_Wrapper_add_CRZ, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CRY gate to the front of the gate structure"
    },
    {"add_CP", (PyCFunction) qgd_Circuit_Wrapper_add_CP, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CRY gate to the front of the gate structure"
    },
    {"add_CCX", (PyCFunction) qgd_Circuit_Wrapper_add_CCX, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CCX gate to the front of the gate structure"
    },
    {"add_SWAP", (PyCFunction) qgd_Circuit_Wrapper_add_SWAP, METH_VARARGS | METH_KEYWORDS,
     "Call to add a SWAP gate to the front of the gate structure"
    },
    {"add_CSWAP", (PyCFunction) qgd_Circuit_Wrapper_add_CSWAP, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CSWAP gate to the front of the gate structure"
    },
    {"add_CROT", (PyCFunction) qgd_Circuit_Wrapper_add_CROT, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CROT gate to the front of the gate structure"
    },
    {"add_CR", (PyCFunction) qgd_Circuit_Wrapper_add_CR, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CR gate to the front of the gate structure"
    },
    {"add_adaptive", (PyCFunction) qgd_Circuit_Wrapper_add_adaptive, METH_VARARGS | METH_KEYWORDS,
     "Call to add an adaptive gate to the front of the gate structure"
    },
    {"add_Circuit", (PyCFunction) qgd_Circuit_Wrapper_add_Circuit, METH_VARARGS,
     "Call to add a block of operations to the front of the gate structure."
    },
#ifdef __DFE__
    {"convert_to_DFE_gates_with_derivates", (PyCFunction) qgd_Circuit_Wrapper_convert_to_DFE_gates_with_derivates, METH_VARARGS,
     "Call to convert to DFE gates with derivates."
    },
    {"adjust_parameters_for_derivation", (PyCFunction) qgd_Circuit_Wrapper_adjust_parameters_for_derivation, METH_VARARGS,
     "Call to adjust parameters for derivation."
    },
    {"convert_to_DFE_gates", (PyCFunction) qgd_Circuit_Wrapper_convert_to_DFE_gates, METH_VARARGS,
     "Call to convert to DFE gates."
    },
    {"convert_to_DFE_gates", (PyCFunction) qgd_Circuit_Wrapper_convert_to_DFE_gates, METH_VARARGS,
     "Call to convert to DFE gates."
    },
#endif
    {"get_Matrix", (PyCFunction) qgd_Circuit_Wrapper_get_Matrix, METH_VARARGS,
     "Method to get the matrix of the operation."
    },
    {"get_Parameter_Num", (PyCFunction) qgd_Circuit_Wrapper_get_Parameter_Num, METH_NOARGS,
     "Call to get the number of free parameters in the circuit"
    },
    {"apply_to", (PyCFunction) qgd_Circuit_Wrapper_apply_to, METH_VARARGS | METH_KEYWORDS,
     "Call to apply the gate on the input matrix (or state)."
    },
    {"get_Second_Renyi_Entropy", (PyCFunction) qgd_Circuit_Wrapper_get_Second_Renyi_Entropy, METH_VARARGS,
     "Wrapper function to evaluate the second Rényi entropy of a quantum circuit at a specific parameter set."
    },
    {"get_Qbit_Num", (PyCFunction) qgd_Circuit_Wrapper_get_Qbit_Num, METH_NOARGS,
     "Call to get the number of qubits in the circuit"
    },
    {"set_Qbit_Num", (PyCFunction) qgd_Circuit_Wrapper_set_Qbit_Num, METH_VARARGS,
     "Call to set the number of qubits in the circuit"
    },
    {"get_Qbits", (PyCFunction) qgd_Circuit_Wrapper_get_Qbits, METH_NOARGS,
     "Call to get the list of qubits involved in the circuit"
    },
    {"set_min_fusion", (PyCFunction) qgd_Circuit_Wrapper_set_Min_Fusion, METH_VARARGS,
     "Call to set the min fusion in the circuit"
    },
    {"Remap_Qbits", (PyCFunction) qgd_Circuit_Wrapper_Remap_Qbits, METH_VARARGS,
     "Call to remap the qubits in the circuit."
    },
    {"get_Gate", (PyCFunction) qgd_Circuit_Wrapper_get_gate, METH_VARARGS,
     "Method to get the i-th decomposing gates."
    },
    {"get_Gates", (PyCFunction) qgd_Circuit_Wrapper_get_gates, METH_NOARGS,
     "Method to get the tuple of decomposing gates."
    },
    {"get_Gate_Nums", (PyCFunction) qgd_Circuit_Wrapper_get_Gate_Nums, METH_NOARGS,
     "Method to get statistics on the gate counts in the circuit."
    },   
    {"get_Parameter_Start_Index", (PyCFunction) qgd_Circuit_Wrapper_get_Parameter_Start_Index, METH_NOARGS,
     "Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated."
    },
    {"Extract_Parameters", (PyCFunction) qgd_Circuit_Wrapper_Extract_Parameters, METH_VARARGS,
     "Call to extract the parameters corresponding to the gate from a parameter array associated with the circuit in which the gate is embedded."
    },
    {"get_Flat_Circuit", (PyCFunction) qgd_Circuit_Wrapper_get_Flat_Circuit, METH_NOARGS,
     "Method to generate a flat circuit. A flat circuit does not contain subcircuits: there are no Gates_block instances (containing subcircuits) in the resulting circuit. If the original circuit contains subcircuits, the gates in the subcircuits are directly incorporated in the resulting flat circuit."
    },
    {"get_Parents", (PyCFunction) qgd_Circuit_Wrapper_get_parents, METH_VARARGS,
     "Method to get the list of parent gate indices. Then the parent gates can be obtained from the list of gates involved in the circuit."
    },
    {"get_Children", (PyCFunction) qgd_Circuit_Wrapper_get_children, METH_VARARGS,
     "Method to get the list of child gate indices. Then the children gates can be obtained from the list of gates involved in the circuit."
    },
    {"__getstate__", (PyCFunction) qgd_Circuit_Wrapper_getstate, METH_NOARGS,
     "Method to extract the stored quantum circuit in a human-readable data serialized and pickle-able format."
    },
    {"__setstate__", (PyCFunction) qgd_Circuit_Wrapper_setstate, METH_VARARGS,
     "Call to set the state of a quantum circuit from a human-readable data serialized and pickle-able format."
    },


    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class Circuit.
*/
static PyTypeObject qgd_Circuit_Wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_Circuit_Wrapper.qgd_Circuit_Wrapper", /*tp_name*/
  sizeof(qgd_Circuit_Wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_Circuit_Wrapper_dealloc, /*tp_dealloc*/
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
  "Object to represent a qgd_Circuit_Wrapper class of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  qgd_Circuit_Wrapper_Methods, /*tp_methods*/
  qgd_Circuit_Wrapper_Members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_Circuit_Wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_Circuit_Wrapper_new, /*tp_new*/
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
static PyModuleDef qgd_Circuit_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    "qgd_Circuit_Wrapper",
    "Python binding for QGD Circuit class",
    -1,
};

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_Circuit_Wrapper(void)
{

    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&qgd_Circuit_Wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&qgd_Circuit_Wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&qgd_Circuit_Wrapper_Type);
    if (PyModule_AddObject(m, "qgd_Circuit_Wrapper", (PyObject *) &qgd_Circuit_Wrapper_Type) < 0) {
        Py_DECREF(&qgd_Circuit_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
