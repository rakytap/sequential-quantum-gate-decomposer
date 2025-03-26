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
\file qgd_Operation_BLock.cpp
\brief Python interface for the Gates_block class
*/

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "structmember.h"
#include "Gates_block.h"
#include "CZ.h"
#include "CH.h"
#include "CNOT.h"
#include "U3.h"
#include "RX.h"
#include "RY.h"
#include "CRY.h"
#include "RZ.h"
#include "H.h"
#include "X.h"
#include "Y.h"
#include "Z.h"
#include "SX.h"
#include "SYC.h"
#include "UN.h"
#include "ON.h"
#include "Adaptive.h"
#include "Composite.h"

#include "numpy_interface.h"

#ifdef __DFE__
#include <numpy/arrayobject.h>
#include "numpy_interface.h"
#endif


/**
@brief Type definition of the qgd_U3_Wrapper Python class of the qgd_U3_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the U3 gate
    U3* gate;
} qgd_U3_Wrapper;

/**
@brief Type definition of the qgd_RX_Wrapper Python class of the qgd_RX_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the RX gate
    RX* gate;
} qgd_RX_Wrapper;

/**
@brief Type definition of the qgd_RY_Wrapper Python class of the qgd_RX_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the RX gate
    RY* gate;
} qgd_RY_Wrapper;


/**
@brief Type definition of the qgd_CRY_Wrapper Python class of the qgd_CRY_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the CRY gate
    CRY* gate;
} qgd_CRY_Wrapper;


/**
@brief Type definition of the qgd_RX_Wrapper Python class of the qgd_RX_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the RX gate
    RZ* gate;
} qgd_RZ_Wrapper;



/**
@brief Type definition of the qgd_H_Wrapper Python class of the qgd_H_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the X gate
    X* gate;
} qgd_X_Wrapper;


/**
@brief Type definition of the qgd_RX_Wrapper Python class of the qgd_RX_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the RX gate
    Y* gate;
} qgd_Y_Wrapper;



/**
@brief Type definition of the qgd_RX_Wrapper Python class of the qgd_RX_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the RX gate
    Z* gate;
} qgd_Z_Wrapper;




/**
@brief Type definition of the qgd_H_Wrapper Python class of the qgd_H_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the X gate
    H* gate;
} qgd_H_Wrapper;


/**
@brief Type definition of the qgd_RX_Wrapper Python class of the qgd_RX_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the RX gate
    SX* gate;
} qgd_SX_Wrapper;


/**
@brief Type definition of the  qgd_CNOT_Wrapper Python class of the  qgd_CNOT_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the CNOT gate
    CNOT* gate;
} qgd_CNOT_Wrapper;


/**
@brief Type definition of the  qgd_CZ_Wrapper Python class of the  qgd_CZ_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the CZ gate
    CZ* gate;
} qgd_CZ_Wrapper;

/**
@brief Type definition of the  qgd_CH_Wrapper Python class of the  qgd_CH_Wrapper module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the CH gate
    CH* gate;
} qgd_CH_Wrapper;

/**
@brief Type definition of the  qgd_SYC Python class of the  qgd_SYC module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the SYC gate
    SYC* gate;
} qgd_SYC;


/**
@brief Type definition of the  qgd_Gate Python class of the  qgd_Gate module
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
    Gates_block* circuit;
} qgd_Circuit_Wrapper;


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param qbit_num Number of qubits spanning the unitary
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
Gates_block* 
create_Circuit( int qbit_num ) {

    return new Gates_block(qbit_num);
}

/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
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
    qgd_Circuit_Wrapper *self;
    self = (qgd_Circuit_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}
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
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"qbit_num", NULL};

    // initiate variables for input arguments
    int  qbit_num = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,
                                     &qbit_num))
        return -1;

    // create instance of class Circuit
    if (qbit_num > 0 ) {
        self->circuit = create_Circuit( qbit_num );
    }
    return 0;
}


/**
@brief Structure containing metadata about the members of class qgd_Circuit_Wrapper.
*/
static PyMemberDef qgd_Circuit_Wrapper_Members[] = {
    {NULL}  /* Sentinel */
};



/**
@brief Wrapper function to add a U3 gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit.
@param args A tuple of the input arguments: target_qbit (int), Theta (bool), Phi (bool), Lambda (bool)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_U3(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit", (char*)"Theta", (char*)"Phi", (char*)"Lambda", NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 
    bool Theta = true;
    bool Phi = true;
    bool Lambda = true;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ibbb", kwlist,
                                     &target_qbit, &Theta, &Phi, &Lambda))
        return Py_BuildValue("i", -1);

    // adding U3 gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_u3(target_qbit, Theta, Phi, Lambda);
    }

    return Py_BuildValue("i", 0);

}



/**
@brief Wrapper function to add a RX gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_RX(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit", NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,
                                     &target_qbit))
        return Py_BuildValue("i", -1);

    // adding U3 gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_rx(target_qbit);
    }

    return Py_BuildValue("i", 0);

}




/**
@brief Wrapper function to add a RY gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_RY(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit", NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,
                                     &target_qbit))
        return Py_BuildValue("i", -1);

    // adding U3 gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_ry(target_qbit);
    }

    return Py_BuildValue("i", 0);

}




/**
@brief Wrapper function to add a RZ gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_RZ(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit", NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,
                                     &target_qbit))
        return Py_BuildValue("i", -1);

    // adding U3 gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_rz(target_qbit);
    }

    return Py_BuildValue("i", 0);

}


/**
@brief Wrapper function to add a CNOT gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: control_qbit (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_CNOT(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit",  (char*)"control_qbit", NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 
    int  control_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,
                                     &target_qbit, &control_qbit))
        return Py_BuildValue("i", -1);


    // adding CNOT gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_cnot(target_qbit, control_qbit);
    }

    return Py_BuildValue("i", 0);

}



/**
@brief Wrapper function to add a CZ gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: control_qbit (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_CZ(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit",  (char*)"control_qbit", NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 
    int  control_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,
                                     &target_qbit, &control_qbit))
        return Py_BuildValue("i", -1);


    // adding CZ gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_cz(target_qbit, control_qbit);
    }

    return Py_BuildValue("i", 0);

}




/**
@brief Wrapper function to add a CH gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: control_qbit (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_CH(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit", (char*)"control_qbit",  NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 
    int  control_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,
                                     &target_qbit, &control_qbit))
        return Py_BuildValue("i", -1);


    // adding CZ gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_ch(target_qbit, control_qbit);
    }

    return Py_BuildValue("i", 0);

}





/**
@brief Wrapper function to add a Sycamore gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: control_qbit (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_SYC(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit", (char*)"control_qbit",  NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 
    int  control_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,
                                     &target_qbit, &control_qbit))
        return Py_BuildValue("i", -1);


    // adding Sycamore gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_syc(target_qbit, control_qbit);
    }

    return Py_BuildValue("i", 0);

}

/**
@brief Wrapper function to add a Hadamard gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class Circuit.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_H(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit",  NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,
                                     &target_qbit))
        return Py_BuildValue("i", -1);


    // adding X gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_h(target_qbit);
    }

    return Py_BuildValue("i", 0);

}

/**
@brief Wrapper function to add a X gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class Circuit.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_X(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit",  NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,
                                     &target_qbit))
        return Py_BuildValue("i", -1);


    // adding X gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_x(target_qbit);
    }

    return Py_BuildValue("i", 0);

}


/**
@brief Wrapper function to add a X gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class Circuit.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_Y(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit",  NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,
                                     &target_qbit))
        return Py_BuildValue("i", -1);


    // adding X gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_y(target_qbit);
    }

    return Py_BuildValue("i", 0);

}


/**
@brief Wrapper function to add a X gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class Circuit.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_Z(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit",  NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,
                                     &target_qbit))
        return Py_BuildValue("i", -1);


    // adding X gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_z(target_qbit);
    }

    return Py_BuildValue("i", 0);

}



/**
@brief Wrapper function to add a SX gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class Circuit.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_SX(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit",  NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,
                                     &target_qbit))
        return Py_BuildValue("i", -1);


    // adding SX gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_sx(target_qbit);
    }

    return Py_BuildValue("i", 0);

}


/**
@brief Wrapper function to add an adaptive gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_CRY(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit", (char*)"control_qbit", NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 
    int  control_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,
                                     &target_qbit, &control_qbit))
        return Py_BuildValue("i", -1);

    // adding U3 gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_cry(target_qbit, control_qbit);
    }

    return Py_BuildValue("i", 0);

}




/**
@brief Wrapper function to add an adaptive gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_Wrapper_add_adaptive(qgd_Circuit_Wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"target_qbit", (char*)"control_qbit", NULL};

    // initiate variables for input arguments
    int  target_qbit = -1; 
    int  control_qbit = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,
                                     &target_qbit, &control_qbit))
        return Py_BuildValue("i", -1);

    // adding U3 gate to the end of the gate structure
    if (target_qbit != -1 ) {
        self->circuit->add_adaptive(target_qbit, control_qbit);
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
    PyObject* parameters_mtx_np = NULL;
    if (!PyArg_ParseTuple(args, "|Ob",
                                     &parameters_mtx_np, &only_derivates))
        return Py_BuildValue("");

    if ( parameters_mtx_np == NULL ) return Py_BuildValue("");
    PyObject* parameters_mtx = PyArray_FROM_OTF(parameters_mtx_np, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

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
    PyObject* parameters_mtx_np = NULL, *dfegates = NULL;
    if (!PyArg_ParseTuple(args, "|OOi",
                                     &parameters_mtx_np, &dfegates, &start_index))
        return Py_BuildValue("");

    if ( parameters_mtx_np == NULL ) return Py_BuildValue("");
    PyObject* parameters_mtx = PyArray_FROM_OTF(parameters_mtx_np, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

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
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
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
@brief Call to apply the gate operation on the inut matrix
*/
static PyObject *
qgd_Circuit_Wrapper_apply_to( qgd_Circuit_Wrapper *self, PyObject *args ) {

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

    int parallel = 1;
    self->circuit->apply_to( parameters_mtx, unitary_mtx, parallel );
    
    if (unitary_mtx.data != PyArray_DATA(unitary)) {
        memcpy(PyArray_DATA(unitary), unitary_mtx.data, unitary_mtx.size() * sizeof(QGD_Complex16));
    }

    Py_DECREF(parameters_arr);
    Py_DECREF(unitary);

    return Py_BuildValue("i", 0);
}



/**
@brief Wrapper function to evaluate the second Rényi entropy of a quantum circuit at a specific parameter set.
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
        std::string err( "Invalid pointer to decomposition class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


    return Py_BuildValue("i", qbit_num );
    
}




/**
@brief Call to get the metadata organised into Python dictionary of the idx-th gate
@param circuit A pointer pointing to an instance of the class Gates_block.
@param idx Labels the idx-th gate.
@return Returns with a python dictionary containing the metadata of the idx-th gate
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

    if (gate->get_type() == CNOT_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_CNOT");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_CNOT" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_CNOT"); 

        PyObject* gate_input = Py_BuildValue("(OOO)", qbit_num, target_qbit, control_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_CNOT_Wrapper* py_gate_C = reinterpret_cast<qgd_CNOT_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<CNOT*>( gate->clone() );

        Py_DECREF( qgd_gate );        
        Py_DECREF( gate_input );


    }
    else if (gate->get_type() == CZ_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_CZ");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_CZ" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_CZ");

        PyObject* gate_input = Py_BuildValue("(OOO)", qbit_num, target_qbit, control_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_CZ_Wrapper* py_gate_C = reinterpret_cast<qgd_CZ_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<CZ*>( gate->clone() );


        Py_DECREF( qgd_gate );                
        Py_DECREF( gate_input );


    }
    else if (gate->get_type() == CH_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_CH");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_CH" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_CH");

        PyObject* gate_input = Py_BuildValue("(OOO)", qbit_num, target_qbit, control_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_CH_Wrapper* py_gate_C = reinterpret_cast<qgd_CH_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<CH*>( gate->clone() );

        Py_DECREF( qgd_gate );                
        Py_DECREF( gate_input );

    }
    else if (gate->get_type() == SYC_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_SYC");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_SYC" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_SYC");

        PyObject* gate_input = Py_BuildValue("(OOO)", qbit_num, target_qbit, control_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_SYC* py_gate_C = reinterpret_cast<qgd_SYC*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<SYC*>( gate->clone() );

        Py_DECREF( qgd_gate );                
        Py_DECREF( gate_input );


    }
    else if (gate->get_type() == U3_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_U3");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_U3" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_U3");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit );
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_U3_Wrapper* py_gate_C = reinterpret_cast<qgd_U3_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<U3*>( gate->clone() );


        Py_DECREF( qgd_gate );                
        Py_DECREF( gate_input );


    }
    else if (gate->get_type() == RX_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_RX");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_RX" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_RX");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_RX_Wrapper* py_gate_C = reinterpret_cast<qgd_RX_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<RX*>( gate->clone() );

        Py_DECREF( qgd_gate );                
        Py_DECREF( gate_input );

    }
    else if (gate->get_type() == RY_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_RY");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_RY" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_RY");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_RY_Wrapper* py_gate_C = reinterpret_cast<qgd_RY_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<RY*>( gate->clone() );

        Py_DECREF( qgd_gate );               
        Py_DECREF( gate_input );

    }
    else if (gate->get_type() == CRY_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_CRY");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_CRY" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_CRY");

        PyObject* gate_input = Py_BuildValue("(OOO)", qbit_num, target_qbit, control_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_CH_Wrapper* py_gate_C = reinterpret_cast<qgd_CH_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<CH*>( gate->clone() );

        Py_DECREF( qgd_gate );                
        Py_DECREF( gate_input );

/*

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_CRY");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_CRY" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_CRY");

        PyObject* gate_input = Py_BuildValue("(OOO)", qbit_num, target_qbit, control_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_CRY_Wrapper* py_gate_C = reinterpret_cast<qgd_CRY_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<CRY*>( gate->clone() );

        Py_DECREF( qgd_gate );               
        Py_DECREF( gate_input );
*/
    }
    else if (gate->get_type() == RZ_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_RZ");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_RZ" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_RZ");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_RZ_Wrapper* py_gate_C = reinterpret_cast<qgd_RZ_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<RZ*>( gate->clone() );

        Py_DECREF( qgd_gate );        
        Py_DECREF( gate_input );
        
    }
    else if (gate->get_type() == H_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_H");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_H" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_H");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_H_Wrapper* py_gate_C = reinterpret_cast<qgd_H_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<H*>( gate->clone() );

        Py_DECREF( qgd_gate );        
        Py_DECREF( gate_input );
        

    }    
    else if (gate->get_type() == X_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_X");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_X" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_X");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_X_Wrapper* py_gate_C = reinterpret_cast<qgd_X_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<X*>( gate->clone() );

        Py_DECREF( qgd_gate );                
        Py_DECREF( gate_input );


    }
        else if (gate->get_type() == Y_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_Y");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_Y" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_Y");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_Y_Wrapper* py_gate_C = reinterpret_cast<qgd_Y_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<Y*>( gate->clone() );

        Py_DECREF( qgd_gate );                
        Py_DECREF( gate_input );


    }
    else if (gate->get_type() == Z_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_Z");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_Z" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_Z");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_Z_Wrapper* py_gate_C = reinterpret_cast<qgd_Z_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<Z*>( gate->clone() );

        Py_DECREF( qgd_gate );                
        Py_DECREF( gate_input );


    }
    else if (gate->get_type() == SX_OPERATION) {

        // import gate operation modules
        PyObject* qgd_gate  = PyImport_ImportModule("squander.gates.qgd_SX");

        if ( qgd_gate == NULL ) {
            PyErr_SetString(PyExc_Exception, "Module import error: squander.gates.qgd_SX" );
            return NULL;
        }

        PyObject* qgd_gate_Dict  = PyModule_GetDict( qgd_gate );
        // PyDict_GetItemString creates a borrowed reference to the item in the dict. Reference counting is not increased on this element, dont need to decrease the reference counting at the end
        PyObject* py_gate_class = PyDict_GetItemString( qgd_gate_Dict, "qgd_SX");

        PyObject* gate_input = Py_BuildValue("(OO)", qbit_num, target_qbit);
        py_gate              = PyObject_CallObject(py_gate_class, gate_input);

        // replace dummy data with real gate data
        qgd_SX_Wrapper* py_gate_C = reinterpret_cast<qgd_SX_Wrapper*>( py_gate );
        delete( py_gate_C->gate );
        py_gate_C->gate = static_cast<SX*>( gate->clone() );

        Py_DECREF( qgd_gate );               
        Py_DECREF( gate_input );


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
        py_gate_C->circuit->combine( circuit );

        Py_DECREF( qgd_circuit );            
        Py_DECREF( circuit_input );

    }
    else {
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
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: idx (int)
idx: labels the idx-th gate.
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
@brief Call to get the incorporated gates in a python list
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
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: 
gate: the gate for which we are retriving the parents
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
    for(int idx=0; idx<parents.size(); idx++) {

        Gate* parent_gate = parents[idx];

        // find the index of the parent_gate
        int parent_idx = -1;
        for( int jdx=0; jdx<gates.size(); jdx++ ) {

            Gate* gate = gates[jdx];

            if( parent_gate == gate ) {
                parent_idx = jdx;
                break;
            }

            if( jdx == gates.size()-1 ) {
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
@param self A pointer pointing to an instance of the class qgd_Circuit_Wrapper.
@param args A tuple of the input arguments: 
gate: the gate for which we are retriving the children
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
    for(int idx=0; idx<children.size(); idx++) {

        Gate* child_gate = children[idx];

        // find the index of the child_gate
        int child_idx = -1;
        for( int jdx=0; jdx<gates.size(); jdx++ ) {

            Gate* gate = gates[jdx];

            if( child_gate == gate ) {
                child_idx = jdx;
                break;
            }

            if( jdx == gates.size()-1 ) {
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
@brief Call to extract the paramaters corresponding to the gate, from a parameter array associated to the circuit in which the gate is embedded.
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
@brief Method to generate a flat circuit. A flat circuit is a circuit does not containing subcircuits: there are no Gates_block instances (containing subcircuits) in the resulting circuit. If the original circuit contains subcircuits, the gates in the subcircuits are directly incorporated in the resulting flat circuit.
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
@brief Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated
@return Returns with the starting index
*/
static PyObject *
qgd_Circuit_Wrapper_get_Parameter_Start_Index( qgd_Circuit_Wrapper *self ) {

    int start_index = self->circuit->get_parameter_start_idx();

    return Py_BuildValue("i", start_index);

}




static PyMethodDef qgd_Circuit_Wrapper_Methods[] = {
    {"add_U3", (PyCFunction) qgd_Circuit_Wrapper_add_U3, METH_VARARGS | METH_KEYWORDS,
     "Call to add a U3 gate to the front of the gate structure"
    },
    {"add_RX", (PyCFunction) qgd_Circuit_Wrapper_add_RX, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RX gate to the front of the gate structure"
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
    {"add_CRY", (PyCFunction) qgd_Circuit_Wrapper_add_CRY, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CRY gate to the front of the gate structure"
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
    {"apply_to", (PyCFunction) qgd_Circuit_Wrapper_apply_to, METH_VARARGS,
     "Call to apply the gate on the input matrix."
    },
    {"get_Second_Renyi_Entropy", (PyCFunction) qgd_Circuit_Wrapper_get_Second_Renyi_Entropy, METH_VARARGS,
     "Wrapper function to evaluate the second Rényi entropy of a quantum circuit at a specific parameter set."
    },
    {"get_Qbit_Num", (PyCFunction) qgd_Circuit_Wrapper_get_Qbit_Num, METH_NOARGS,
     "Call to get the number of qubits in the circuit"
    },
    {"get_Gate", (PyCFunction) qgd_Circuit_Wrapper_get_gate, METH_VARARGS,
     "Method to get the i-th decomposing gates."
    },
    {"get_Gates", (PyCFunction) qgd_Circuit_Wrapper_get_gates, METH_NOARGS,
     "Method to get the tuple of decomposing gates."
    },
    {"get_Parameter_Start_Index", (PyCFunction) qgd_Circuit_Wrapper_get_Parameter_Start_Index, METH_NOARGS,
     "Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated."
    },
    {"Extract_Parameters", (PyCFunction) qgd_Circuit_Wrapper_Extract_Parameters, METH_VARARGS,
     "Call to extract the paramaters corresponding to the gate, from a parameter array associated to the circuit in which the gate is embedded."
    },
    {"get_Flat_Circuit", (PyCFunction) qgd_Circuit_Wrapper_get_Flat_Circuit, METH_NOARGS,
     "Method to generate a flat circuit. A flat circuit is a circuit does not containing subcircuits: there are no Gates_block instances (containing subcircuits) in the resulting circuit. If the original circuit contains subcircuits, the gates in the subcircuits are directly incorporated in the resulting flat circuit."
    },
    {"get_Parents", (PyCFunction) qgd_Circuit_Wrapper_get_parents, METH_VARARGS,
     "Method to get the list of parent gate indices. Then the parent gates can be obtained from the list of gates involved in the circuit."
    },
    {"get_Children", (PyCFunction) qgd_Circuit_Wrapper_get_children, METH_VARARGS,
     "Method to get the list of child gate indices. Then the children gates can be obtained from the list of gates involved in the circuit."
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
