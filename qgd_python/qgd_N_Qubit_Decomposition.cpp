#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <stdio.h>

#include "N_Qubit_Decomposition.h"
#include "Sub_Matrix_Decomposition_Custom.h"





/**
@brief Type definition of the qgd_N_Qubit_Decomposition Python class of the qgd_N_Qubit_Decomposition module
*/
typedef struct qgd_N_Qubit_Decomposition {
    PyObject_HEAD
    /// pointer to the unitary to be decomposed to keep it alive
    PyObject *Umtx;
    /// An object to decompose the unitary
    N_Qubit_Decomposition<Sub_Matrix_Decomposition_Custom>* decomp;

} qgd_N_Qubit_Decomposition;



/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a pointer pointing to the class instance (C++ linking is needed)
@param Umtx An instance of class Matrix containing the unitary to be decomposed
@param qbit_num Number of qubits spanning the unitary
@param optimize_layer_num Logical value. Set true to optimize the number of decomposing layers during the decomposition procedure, or false otherwise.
@param initial_guess Type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
N_Qubit_Decomposition<Sub_Matrix_Decomposition_Custom>* 
create_N_Qubit_Decomposition( Matrix& Umtx, int qbit_num, bool optimize_layer_num, guess_type initial_guess ) {

    return new N_Qubit_Decomposition<Sub_Matrix_Decomposition_Custom>( Umtx, qbit_num, optimize_layer_num, initial_guess );
}


/**
@brief Call to deallocate an instance of N_Qubit_Decomposition class
@param ptr A pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void
release_N_Qubit_Decomposition( N_Qubit_Decomposition<Sub_Matrix_Decomposition_Custom>*  instance ) {
    delete instance;
    return;
}




extern "C"
{


/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition is destroyed
@param self A pointer pointing to an instance of class qgd_N_Qubit_Decomposition.
*/
static void
qgd_N_Qubit_Decomposition_dealloc(qgd_N_Qubit_Decomposition *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_N_Qubit_Decomposition( self->decomp );

    // release the unitary to be decomposed
    Py_DECREF(self->Umtx);    
    
    Py_TYPE(self)->tp_free((PyObject *) self);

}

/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition is allocated
@param type A pointer pointing to a structure describing the type of the class qgd_N_Qubit_Decomposition.
*/
static PyObject *
qgd_N_Qubit_Decomposition_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    qgd_N_Qubit_Decomposition *self;
    self = (qgd_N_Qubit_Decomposition *) type->tp_alloc(type, 0);
    if (self != NULL) {}
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class qgd_N_Qubit_Decomposition is initialized
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition.
@param args A tuple of the input arguments: Umtx (numpy array), qbit_num (integer)
@param kwds A tuple of keywords
*/
static int
qgd_N_Qubit_Decomposition_init(qgd_N_Qubit_Decomposition *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"Umtx", (char*)"qbit_num", NULL};
 
    // initiate variables for input arguments
    PyObject *Umtx_arg = NULL;
    int  qbit_num = -1; 

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Oi", kwlist,
                                     &Umtx_arg, &qbit_num))
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
    
    // create an instance of the class N_Qubit_Decomposition
    if (qbit_num > 0 ) {
        self->decomp =  create_N_Qubit_Decomposition( Umtx_mtx, qbit_num, false, ZEROS);
    }
    else {
        std::cout << "The number of qubits should be given as a positive integer, " << qbit_num << "  was given" << std::endl;
        return -1;
    }

    return 0;
}

/**
@brief Wrapper function to call the start_decomposition method of C++ class N_Qubit_Decomposition
@param self A pointer pointing to an instance of the class qgd_N_Qubit_Decomposition.
@param args A tuple of the input arguments: finalize_decomp (bool), prepare_export (bool)
@param kwds A tuple of keywords
*/
static PyObject *
qgd_N_Qubit_Decomposition_Start_Decomposition(qgd_N_Qubit_Decomposition *self, PyObject *args, PyObject *kwds)
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
@brief Structure containing metadata about the members of class qgd_N_Qubit_Decomposition.
*/
static PyMemberDef qgd_N_Qubit_Decomposition_members[] = {
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the methods of class qgd_N_Qubit_Decomposition.
*/
static PyMethodDef qgd_N_Qubit_Decomposition_methods[] = {
    {"Start_Decomposition", (PyCFunction) qgd_N_Qubit_Decomposition_Start_Decomposition, METH_VARARGS | METH_KEYWORDS,
     "Wrapper method to start the decomposition."
    },
    {NULL}  /* Sentinel */
};

/**
@brief A structure describing the type of the class qgd_N_Qubit_Decomposition.
*/
static PyTypeObject qgd_N_Qubit_Decomposition_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "qgd_N_Qubit_Decomposition.qgd_N_Qubit_Decomposition", /*tp_name*/
  sizeof(qgd_N_Qubit_Decomposition), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) qgd_N_Qubit_Decomposition_dealloc, /*tp_dealloc*/
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
  qgd_N_Qubit_Decomposition_methods, /*tp_methods*/
  qgd_N_Qubit_Decomposition_members, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) qgd_N_Qubit_Decomposition_init, /*tp_init*/
  0, /*tp_alloc*/
  qgd_N_Qubit_Decomposition_new, /*tp_new*/
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
static PyModuleDef qgd_N_Qubit_Decomposition_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "qgd_N_Qubit_Decomposition",
    .m_doc = "Python binding for QGD N_Qubit_Decomposition class",
    .m_size = -1,
};


/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_qgd_N_Qubit_Decomposition(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&qgd_N_Qubit_Decomposition_Type) < 0)
        return NULL;

    m = PyModule_Create(&qgd_N_Qubit_Decomposition_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&qgd_N_Qubit_Decomposition_Type);
    if (PyModule_AddObject(m, "qgd_N_Qubit_Decomposition", (PyObject *) &qgd_N_Qubit_Decomposition_Type) < 0) {
        Py_DECREF(&qgd_N_Qubit_Decomposition_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}


} //extern C

