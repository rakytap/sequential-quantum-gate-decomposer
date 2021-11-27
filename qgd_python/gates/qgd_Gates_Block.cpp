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
\file qgd_Operation_BLock.cpp
\brief Python interface for the Gates_block class
*/

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "structmember.h"
#include "Gates_block.h"



/**
@brief Type definition of the qgd_Operation_Block Python class of the qgd_Operation_Block module
*/
typedef struct qgd_Gates_Block {
    PyObject_HEAD
    Gates_block* gate;
} qgd_Gates_Block;


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param qbit_num Number of qubits spanning the unitary
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
Gates_block* 
create_Gates_block( int qbit_num ) {

    return new Gates_block(qbit_num);
}

/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_Gates_block( Gates_block*  instance ) {
    delete instance;
    return;
}





extern "C"
{

/**
@brief Method called when a python instance of the class qgd_Gates_Block is destroyed
@param self A pointer pointing to an instance of class qgd_Gates_Block.
*/
static void
qgd_Gates_Block_dealloc(qgd_Gates_Block *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_Gates_block( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class qgd_Gates_Block is allocated
@param type A pointer pointing to a structure describing the type of the class Gates_Block.
*/
static PyObject *
qgd_Gates_Block_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_Gates_Block *self;
    self = (qgd_Gates_Block *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_Gates_Block is initialized
@param self A pointer pointing to an instance of the class qgd_Gates_Block.
@param args A tuple of the input arguments: qbit_num (integer)
qbit_num: the number of qubits spanning the operations
@param kwds A tuple of keywords
*/
static int
qgd_Gates_Block_init(qgd_Gates_Block *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"qbit_num", NULL};

    // initiate variables for input arguments
    int  qbit_num = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,
                                     &qbit_num))
        return -1;

    // create instance of class Gates_block
    if (qbit_num > 0 ) {
        self->gate = create_Gates_block( qbit_num );
    }
    return 0;
}


/**
@brief Structure containing metadata about the members of class qgd_Gates_Block.
*/
static PyMemberDef qgd_Gates_Block_Members[] = {
    {NULL}  /* Sentinel */
};



/**
@brief Wrapper function to add a U3 gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Gates_Block.
@param args A tuple of the input arguments: target_qbit (int), Theta (bool), Phi (bool), Lambda (bool)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Gates_Block_add_U3(qgd_Gates_Block *self, PyObject *args, PyObject *kwds)
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
@param self A pointer pointing to an instance of the class qgd_Gates_Block.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Gates_Block_add_RX(qgd_Gates_Block *self, PyObject *args, PyObject *kwds)
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
@param self A pointer pointing to an instance of the class qgd_Gates_Block.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Gates_Block_add_RY(qgd_Gates_Block *self, PyObject *args, PyObject *kwds)
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
@param self A pointer pointing to an instance of the class qgd_Gates_Block.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Gates_Block_add_RZ(qgd_Gates_Block *self, PyObject *args, PyObject *kwds)
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
@param self A pointer pointing to an instance of the class qgd_Gates_Block.
@param args A tuple of the input arguments: control_qbit (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Gates_Block_add_CNOT(qgd_Gates_Block *self, PyObject *args, PyObject *kwds)
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
@param self A pointer pointing to an instance of the class qgd_Gates_Block.
@param args A tuple of the input arguments: control_qbit (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Gates_Block_add_CZ(qgd_Gates_Block *self, PyObject *args, PyObject *kwds)
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
@param self A pointer pointing to an instance of the class qgd_Gates_Block.
@param args A tuple of the input arguments: control_qbit (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Gates_Block_add_CH(qgd_Gates_Block *self, PyObject *args, PyObject *kwds)
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
@param self A pointer pointing to an instance of the class qgd_Gates_Block.
@param args A tuple of the input arguments: control_qbit (int), target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Gates_Block_add_SYC(qgd_Gates_Block *self, PyObject *args, PyObject *kwds)
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
@param self A pointer pointing to an instance of the class Gates_Block.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Gates_Block_add_X(qgd_Gates_Block *self, PyObject *args, PyObject *kwds)
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
@brief Wrapper function to add a SX gate to the front of the gate structure.
@param self A pointer pointing to an instance of the class Gates_Block.
@param args A tuple of the input arguments: target_qbit (int)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_Gates_Block_add_SX(qgd_Gates_Block *self, PyObject *args, PyObject *kwds)
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
@brief Wrapper function to add a block of operations to the front of the gate structure.
@param self A pointer pointing to an instance of the class qgd_Gates_Block.
@param args A tuple of the input arguments: Py_qgd_Gates_Block (PyObject)
Py_qgd_Gates_Block: an instance of qgd_Gates_Block containing the custom gate structure
*/
static PyObject *
qgd_Gates_Block_add_Gates_Block(qgd_Gates_Block *self, PyObject *args)
{

    // initiate variables for input arguments
    PyObject *Py_Gates_Block; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|O",
                                     &Py_Gates_Block))
        return Py_BuildValue("i", -1);


    qgd_Gates_Block* qgd_op_block = (qgd_Gates_Block*) Py_Gates_Block;


    // adding general gate to the end of the gate structure
    self->gate->add_gate( static_cast<Gate*>( qgd_op_block->gate->clone() ) );

    return Py_BuildValue("i", 0);

}

static PyMethodDef qgd_Gates_Block_Methods[] = {
    {"add_U3", (PyCFunction) qgd_Gates_Block_add_U3, METH_VARARGS | METH_KEYWORDS,
     "Call to add a U3 gate to the front of the gate structure"
    },
    {"add_RX", (PyCFunction) qgd_Gates_Block_add_RX, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RX gate to the front of the gate structure"
    },
    {"add_RY", (PyCFunction) qgd_Gates_Block_add_RY, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RY gate to the front of the gate structure"
    },
    {"add_RZ", (PyCFunction) qgd_Gates_Block_add_RZ, METH_VARARGS | METH_KEYWORDS,
     "Call to add a RZ gate to the front of the gate structure"
    },
    {"add_CNOT", (PyCFunction) qgd_Gates_Block_add_CNOT, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CNOT gate to the front of the gate structure"
    },
    {"add_CZ", (PyCFunction) qgd_Gates_Block_add_CZ, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CZ gate to the front of the gate structure"
    },
    {"add_CH", (PyCFunction) qgd_Gates_Block_add_CH, METH_VARARGS | METH_KEYWORDS,
     "Call to add a CH gate to the front of the gate structure"
    },
    {"add_SYC", (PyCFunction) qgd_Gates_Block_add_SYC, METH_VARARGS | METH_KEYWORDS,
     "Call to add a Sycamore gate to the front of the gate structure"
    },
    {"add_X", (PyCFunction) qgd_Gates_Block_add_X, METH_VARARGS | METH_KEYWORDS,
     "Call to add a X gate to the front of the gate structure"
    },
    {"add_SX", (PyCFunction) qgd_Gates_Block_add_SX, METH_VARARGS | METH_KEYWORDS,
     "Call to add a SX gate to the front of the gate structure"
    },
    {"add_Gates_Block", (PyCFunction) qgd_Gates_Block_add_Gates_Block, METH_VARARGS,
     "Call to add a block of operations to the front of the gate structure."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class Gates_Block.
*/
static PyTypeObject qgd_Gates_Block_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_Gates_Block.qgd_Gates_Block", /*tp_name*/
  sizeof(qgd_Gates_Block), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_Gates_Block_dealloc, /*tp_dealloc*/
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
  "Object to represent a qgd_Gates_block class of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  qgd_Gates_Block_Methods, /*tp_methods*/
  qgd_Gates_Block_Members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_Gates_Block_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_Gates_Block_new, /*tp_new*/
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
static PyModuleDef qgd_Gates_Block_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "qgd_Gates_Block",
    .m_doc = "Python binding for QGD Gates_block class",
    .m_size = -1,
};

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_Gates_Block(void)
{
    PyObject *m;
    if (PyType_Ready(&qgd_Gates_Block_Type) < 0)
        return NULL;

    m = PyModule_Create(&qgd_Gates_Block_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&qgd_Gates_Block_Type);
    if (PyModule_AddObject(m, "qgd_Gates_Block", (PyObject *) &qgd_Gates_Block_Type) < 0) {
        Py_DECREF(&qgd_Gates_Block_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
