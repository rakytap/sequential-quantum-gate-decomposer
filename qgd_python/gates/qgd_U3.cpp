#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include "structmember.h"
#include "U3.h"





/**
@brief Type definition of the qgd_U3 Python class of the qgd_U3 module
*/
typedef struct {
    PyObject_HEAD
    /// Pointer to the C++ class of the U3 gate
    U3* gate;
} qgd_U3;


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param qbit_num The number of qubits spanning the operation.
@param target_qbit The 0<=ID<qbit_num of the target qubit.
@param Theta logical value indicating whether the matrix creation takes an argument theta.
@param Phi logical value indicating whether the matrix creation takes an argument phi
@param Lambda logical value indicating whether the matrix creation takes an argument lambda
*/
U3* 
create_U3( int qbit_num, int target_qbit, bool Theta, bool Phi, bool Lambda ) {

    return new U3( qbit_num, target_qbit, Theta, Phi, Lambda );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_U3( U3*  instance ) {
    delete instance;
    return;
}





extern "C"
{


/**
@brief Method called when a python instance of the class qgd_U3 is destroyed
@param self A pointer pointing to an instance of class qgd_U3.
*/
static void
qgd_U3_dealloc(qgd_U3 *self)
{

    // release the U3 gate
    release_U3( self->gate );

    Py_TYPE(self)->tp_free((PyObject *) self);
}


/**
@brief Method called when a python instance of the class qgd_U3 is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_U3.
*/
static PyObject *
qgd_U3_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_U3 *self;
    self = (qgd_U3 *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_U3 is initialized
@param self A pointer pointing to an instance of the class qgd_U3.
@param args A tuple of the input arguments: qbit_num (int), target_qbit (int), Theta (bool) , Phi (bool), Lambda (bool)
@param kwds A tuple of keywords
*/
static int
qgd_U3_init(qgd_U3 *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"qbit_num", (char*)"target_qbit", (char*)"Theta", (char*)"Phi", (char*)"Lambda", NULL};
    int  qbit_num = -1; 
    int target_qbit = -1;
    bool Theta = false;
    bool Phi = false;
    bool Lambda = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iibbb", kwlist,
                                     &qbit_num, &target_qbit, &Theta, &Phi, &Lambda))
        return -1;

    if (qbit_num != -1 && target_qbit != -1) {
        self->gate = create_U3( qbit_num, target_qbit, Theta, Phi, Lambda );
    }
    return 0;
}


/**
@brief Structure containing metadata about the members of class qgd_U3.
*/
static PyMemberDef qgd_U3_members[] = {
    {NULL}  /* Sentinel */
};


/**
@brief Structure containing metadata about the methods of class qgd_U3.
*/
static PyMethodDef qgd_U3_methods[] = {
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class qgd_U3.
*/
static PyTypeObject  qgd_U3_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_U3.qgd_U3", /*tp_name*/
  sizeof(qgd_U3), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_U3_dealloc, /*tp_dealloc*/
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
  "Object to represent a U3 gate of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  qgd_U3_methods, /*tp_methods*/
  qgd_U3_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_U3_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_U3_new, /*tp_new*/
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
static PyModuleDef  qgd_U3_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "qgd_U3",
    .m_doc = "Python binding for QGD U3 gate",
    .m_size = -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_U3(void)
{
    PyObject *m;
    if (PyType_Ready(& qgd_U3_Type) < 0)
        return NULL;

    m = PyModule_Create(& qgd_U3_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(& qgd_U3_Type);
    if (PyModule_AddObject(m, "qgd_U3", (PyObject *) & qgd_U3_Type) < 0) {
        Py_DECREF(& qgd_U3_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}




}
