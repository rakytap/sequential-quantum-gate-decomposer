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
#include "numpy_interface.h"

#ifdef __DFE__
#include <numpy/arrayobject.h>
#include "numpy_interface.h"
#endif

/**
@brief Type definition of the qgd_Operation_Block Python class of the qgd_Operation_Block module
*/
typedef struct qgd_Circuit {
    PyObject_HEAD
    Gates_block* gate;
} qgd_Circuit;


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
@brief Method called when a python instance of the class qgd_Circuit is destroyed
@param self A pointer pointing to an instance of class qgd_Circuit.
*/
static void
qgd_Circuit_dealloc(qgd_Circuit *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_Circuit( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class qgd_Circuit is allocated
@param type A pointer pointing to a structure describing the type of the class Circuit.
*/
static PyObject *
qgd_Circuit_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_Circuit *self;
    self = (qgd_Circuit *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_Circuit is initialized
@param self A pointer pointing to an instance of the class qgd_Circuit.
@param args A tuple of the input arguments: qbit_num (integer)
qbit_num: the number of qubits spanning the operations
@param kwds A tuple of keywords
*/
static int
qgd_Circuit_init(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate = create_Circuit( qbit_num );
    }
    return 0;
}


/**
@brief Structure containing metadata about the members of class qgd_Circuit.
*/
static PyMemberDef qgd_Circuit_Members[] = {
    {NULL}  /* Sentinel */
};



/**
@brief Wrapper function to add a U3 gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit.
@param args A tuple of the input arguments: target_qbit (int), Theta (bool), Phi (bool), Lambda (bool)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_add_U3(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_u3(target_qbit, Theta, Phi, Lambda);
    }

    return Py_BuildValue("i", 0);

}



/**
@brief Wrapper function to add a RX gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_add_RX(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_rx(target_qbit);
    }

    return Py_BuildValue("i", 0);

}




/**
@brief Wrapper function to add a RY gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_add_RY(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_ry(target_qbit);
    }

    return Py_BuildValue("i", 0);

}




/**
@brief Wrapper function to add a RZ gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_add_RZ(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_rz(target_qbit);
    }

    return Py_BuildValue("i", 0);

}


/**
@brief Wrapper function to add a CNOT gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit.
@param args A tuple of the input arguments: control_qbit (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_add_CNOT(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_cnot(target_qbit, control_qbit);
    }

    return Py_BuildValue("i", 0);

}



/**
@brief Wrapper function to add a CZ gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit.
@param args A tuple of the input arguments: control_qbit (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_add_CZ(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_cz(target_qbit, control_qbit);
    }

    return Py_BuildValue("i", 0);

}




/**
@brief Wrapper function to add a CH gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit.
@param args A tuple of the input arguments: control_qbit (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_add_CH(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_ch(target_qbit, control_qbit);
    }

    return Py_BuildValue("i", 0);

}





/**
@brief Wrapper function to add a Sycamore gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit.
@param args A tuple of the input arguments: control_qbit (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_add_SYC(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_syc(target_qbit, control_qbit);
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
qgd_Circuit_add_X(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_x(target_qbit);
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
qgd_Circuit_add_Y(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_y(target_qbit);
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
qgd_Circuit_add_Z(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_z(target_qbit);
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
qgd_Circuit_add_SX(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_sx(target_qbit);
    }

    return Py_BuildValue("i", 0);

}



/**
@brief Wrapper function to add an adaptive gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Circuit_add_adaptive(qgd_Circuit *self, PyObject *args, PyObject *kwds)
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
        self->gate->add_adaptive(target_qbit, control_qbit);
    }

    return Py_BuildValue("i", 0);

}


/**
@brief Wrapper function to add a block of operations to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Circuit.
@param args A tuple of the input arguments: Py_qgd_Circuit (PyObject)
Py_qgd_Circuit: an instance of qgd_Circuit containing the custom gate structure
*/
static PyObject *
qgd_Circuit_add_Circuit(qgd_Circuit *self, PyObject *args)
{

    // initiate variables for input arguments
    PyObject *Py_Circuit; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O",
                                     &Py_Circuit))
        return Py_BuildValue("i", -1);


    qgd_Circuit* qgd_op_block = (qgd_Circuit*) Py_Circuit;


    // adding general gate to the end of the gate structure
    self->gate->add_gate( static_cast<Gate*>( qgd_op_block->gate->clone() ) );

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
qgd_Circuit_convert_to_DFE_gates_with_derivates(qgd_Circuit *self, PyObject *args)
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
    DFEgate_kernel_type* ret = self->gate->convert_to_DFE_gates_with_derivates(parameters_mtx_mtx, gatesNum, gateSetNum, redundantGateSets, only_derivates);
    return Py_BuildValue("Oii", DFEgateQGD_to_Python(ret, gatesNum), gateSetNum, redundantGateSets);
}

static PyObject *
qgd_Circuit_adjust_parameters_for_derivation(qgd_Circuit *self, PyObject *args)
{
    int gatesNum = -1;
    PyObject* dfegates = NULL;
    if (!PyArg_ParseTuple(args, "|Oi",    
                                     &dfegates, &gatesNum))
        return Py_BuildValue("");
    int gate_idx = -1, gate_set_index = -1;
    DFEgate_kernel_type* dfegates_qgd = DFEgatePython_to_QGD(dfegates);
    self->gate->adjust_parameters_for_derivation(dfegates_qgd, gatesNum, gate_idx, gate_set_index);    
    return Py_BuildValue("Oii", DFEgateQGD_to_Python(dfegates_qgd, gatesNum), gate_idx, gate_set_index);
}

static PyObject *
qgd_Circuit_convert_to_DFE_gates(qgd_Circuit *self, PyObject *args)
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
    DFEgate_kernel_type* ret = self->gate->convert_to_DFE_gates(parameters_mtx_mtx, gatesNum);
    return DFEgateQGD_to_Python(ret, gatesNum);
}

static PyObject *
qgd_Circuit_convert_to_DFE_gates(qgd_Circuit *self, PyObject *args)
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
    self->gate->convert_to_DFE_gates(parameters_mtx_mtx, dfegates_qgd, start_index);
    return DFEgateQGD_to_Python(dfegates_qgd, gatesNum);
}

#endif

/**
@brief Extract the optimized parameters
@param start_index The index of the first inverse gate
*/
static PyObject *
qgd_Circuit_get_Matrix( qgd_Circuit *self, PyObject *args ) {

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


    // get the C++ wrapper around the data
    Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );


    Matrix mtx = self->gate->get_matrix( parameters_mtx );
    
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
qgd_Circuit_get_Parameter_Num( qgd_Circuit *self ) {

    int parameter_num = self->gate->get_parameter_num();

    return Py_BuildValue("i", parameter_num);
}



/**
@brief Call to apply the gate operation on the inut matrix
*/
static PyObject *
qgd_Circuit_apply_to( qgd_Circuit *self, PyObject *args ) {

    PyObject * parameters_arr = NULL;
    PyObject * unitary_arg = NULL;


    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|OO", &parameters_arr, &unitary_arg )) 
        return Py_BuildValue("i", -1);

    
    if ( PyArray_IS_C_CONTIGUOUS(parameters_arr) ) {
        Py_INCREF(parameters_arr);
    }
    else {
        parameters_arr = PyArray_FROM_OTF(parameters_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }

    // get the C++ wrapper around the data
    Matrix_real&& parameters_mtx = numpy2matrix_real( parameters_arr );


    // convert python object array to numpy C API array
    if ( unitary_arg == NULL ) {
        PyErr_SetString(PyExc_Exception, "Input matrix was not given");
        return NULL;
    }

    PyObject* unitary = PyArray_FROM_OTF(unitary_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    // test C-style contiguous memory allocation of the array
    if ( !PyArray_IS_C_CONTIGUOUS(unitary) ) {
        PyErr_SetString(PyExc_Exception, "input mtrix is not memory contiguous");
        return NULL;
    }


    // create QGD version of the input matrix
    Matrix unitary_mtx = numpy2matrix(unitary);


    self->gate->apply_to( parameters_mtx, unitary_mtx );
    
    if (unitary_mtx.data != PyArray_DATA(unitary)) {
        memcpy(PyArray_DATA(unitary), unitary_mtx.data, unitary_mtx.size() * sizeof(QGD_Complex16));
    }

    Py_DECREF(parameters_arr);
    Py_DECREF(unitary);

    return Py_BuildValue("i", 0);
}





static PyMethodDef qgd_Circuit_Methods[] = {
    {"add_U3", (PyCFunction) qgd_Circuit_add_U3, METH_VARARGS | METH_KEYWORDS,
     "Call to add a U3 gate to the front of the gate structure"
    },
    {"add_RX", (PyCFunction) qgd_Circuit_add_RX, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RX gate to the front of the gate structure"
    },
    {"add_RY", (PyCFunction) qgd_Circuit_add_RY, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RY gate to the front of the gate structure"
    },
    {"add_RZ", (PyCFunction) qgd_Circuit_add_RZ, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RZ gate to the front of the gate structure"
    },
    {"add_CNOT", (PyCFunction) qgd_Circuit_add_CNOT, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CNOT gate to the front of the gate structure"
    },
    {"add_CZ", (PyCFunction) qgd_Circuit_add_CZ, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CZ gate to the front of the gate structure"
    },
    {"add_CH", (PyCFunction) qgd_Circuit_add_CH, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CH gate to the front of the gate structure"
    },
    {"add_SYC", (PyCFunction) qgd_Circuit_add_SYC, METH_VARARGS | METH_KEYWORDS,
     "Call to add a Sycamore gate to the front of the gate structure"
    },
    {"add_X", (PyCFunction) qgd_Circuit_add_X, METH_VARARGS | METH_KEYWORDS,
     "Call to add a X gate to the front of the gate structure"
    },
    {"add_Y", (PyCFunction) qgd_Circuit_add_X, METH_VARARGS | METH_KEYWORDS,
     "Call to add a Y gate to the front of the gate structure"
    },
    {"add_Z", (PyCFunction) qgd_Circuit_add_X, METH_VARARGS | METH_KEYWORDS,
     "Call to add a Z gate to the front of the gate structure"
    },
    {"add_SX", (PyCFunction) qgd_Circuit_add_SX, METH_VARARGS | METH_KEYWORDS,
     "Call to add a SX gate to the front of the gate structure"
    },
    {"add_adaptive", (PyCFunction) qgd_Circuit_add_adaptive, METH_VARARGS | METH_KEYWORDS,
     "Call to add an adaptive gate to the front of the gate structure"
    },
    {"add_Circuit", (PyCFunction) qgd_Circuit_add_Circuit, METH_VARARGS,
     "Call to add a block of operations to the front of the gate structure."
    },
#ifdef __DFE__
    {"convert_to_DFE_gates_with_derivates", (PyCFunction) qgd_Circuit_convert_to_DFE_gates_with_derivates, METH_VARARGS,
     "Call to convert to DFE gates with derivates."
    },
    {"adjust_parameters_for_derivation", (PyCFunction) qgd_Circuit_adjust_parameters_for_derivation, METH_VARARGS,
     "Call to adjust parameters for derivation."
    },
    {"convert_to_DFE_gates", (PyCFunction) qgd_Circuit_convert_to_DFE_gates, METH_VARARGS,
     "Call to convert to DFE gates."
    },
    {"convert_to_DFE_gates", (PyCFunction) qgd_Circuit_convert_to_DFE_gates, METH_VARARGS,
     "Call to convert to DFE gates."
    },
#endif
    {"get_Matrix", (PyCFunction) qgd_Circuit_get_Matrix, METH_VARARGS,
     "Method to get the matrix of the operation."
    },
    {"get_Parameter_Num", (PyCFunction) qgd_Circuit_get_Parameter_Num, METH_NOARGS,
     "Call to get the number of free parameters in the circuit"
    },
    {"apply_to", (PyCFunction) qgd_Circuit_apply_to, METH_VARARGS,
     "Call to apply the gate on the input matrix."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class Circuit.
*/
static PyTypeObject qgd_Circuit_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_Circuit.qgd_Circuit", /*tp_name*/
  sizeof(qgd_Circuit), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_Circuit_dealloc, /*tp_dealloc*/
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
  "Object to represent a qgd_Circuit class of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  qgd_Circuit_Methods, /*tp_methods*/
  qgd_Circuit_Members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_Circuit_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_Circuit_new, /*tp_new*/
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
static PyModuleDef qgd_Circuit_Module = {
    PyModuleDef_HEAD_INIT,
    "qgd_Circuit",
    "Python binding for QGD Circuit class",
    -1,
};

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_Circuit(void)
{

    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&qgd_Circuit_Type) < 0)
        return NULL;

    m = PyModule_Create(&qgd_Circuit_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&qgd_Circuit_Type);
    if (PyModule_AddObject(m, "qgd_Circuit", (PyObject *) &qgd_Circuit_Type) < 0) {
        Py_DECREF(&qgd_Circuit_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
