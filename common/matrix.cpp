#include "qgd/matrix.h"




/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
Matrix_base::Matrix_base() {

  // The number of rows
  rows = 0;
  // The number of columns
  cols = 0;
  // pointer to the stored data
  data = NULL;
  // logical variable indicating whether the Matrix needs to be conjugated in CBLAS operations
  conjugated = false;
  // logical variable indicating whether the Matrix needs to be transposed in CBLAS operations
  transposed = false;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  owner = false;
  // mutual exclusion to count the references for class instances refering to the same data.
  reference_mutex = new tbb::spin_mutex;
  // the number of the current references of the present object
  references = new int64_t;
  (*references)=1;

}


/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param rows_in The number of rows in the stored Matrix
@param cols_in The number of columns in the stored Matrix
@return Returns with the instance of the class.
*/
Matrix_base::Matrix_base( QGD_Complex16* data_in, size_t rows_in, size_t cols_in) {

  // The number of rows
  rows = rows_in;
  // The number of columns
  cols = cols_in;
  // pointer to the stored data
  data = data_in;
  // logical variable indicating whether the Matrix needs to be conjugated in CBLAS operations
  conjugated = false;
  // logical variable indicating whether the Matrix needs to be transposed in CBLAS operations
  transposed = false;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  owner = false;
  // mutual exclusion to count the references for class instances refering to the same data.
  reference_mutex = new tbb::spin_mutex;
  // the number of the current references of the present object
  references = new int64_t;
  (*references)=1;

}



/**
@brief Constructor of the class. Allocates data for Matrix rows_in times cols_in. By default the created instance would be the owner of the stored data.
@param rows_in The number of rows in the stored Matrix
@param cols_in The number of columns in the stored Matrix
@return Returns with the instance of the class.
*/
Matrix_base::Matrix_base( size_t rows_in, size_t cols_in) {

  // The number of rows
  rows = rows_in;
  // The number of columns
  cols = cols_in;
  // pointer to the stored data
  data = (QGD_Complex16*)scalable_aligned_malloc( rows*cols*sizeof(QGD_Complex16), CACHELINE);
  assert(data);
  // logical variable indicating whether the Matrix needs to be conjugated in CBLAS operations
  conjugated = false;
  // logical variable indicating whether the Matrix needs to be transposed in CBLAS operations
  transposed = false;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  owner = true;
  // mutual exclusion to count the references for class instances refering to the same data.
  reference_mutex = new tbb::spin_mutex;
  // the number of the current references of the present object
  references = new int64_t;
  (*references)=1;

}


/**
@brief Copy constructor of the class. The new instance shares the stored memory with the input Matrix. (Needed for TBB calls)
@param An instance of class Matrix to be copied.
*/
Matrix_base::Matrix_base(const Matrix_base &in) {

    data = in.data;
    rows = in.rows;
    cols = in.cols;
    transposed = in.transposed;
    conjugated = in.conjugated;
    owner = in.owner;

    reference_mutex = in.reference_mutex;
    references = in.references;

    {
      tbb::spin_mutex::scoped_lock my_lock{*reference_mutex};
      (*references)++;
    }


}



/**
@brief Destructor of the class
*/
Matrix_base::~Matrix_base() {
  release_data();
}

/**
@brief Call to get whether the Matrix should be conjugated in CBLAS functions or not.
@return Returns with true if the Matrix should be conjugated in CBLAS functions or false otherwise.
*/
bool
Matrix_base::is_conjugated() {
  return conjugated;
}

/**
@brief Call to conjugate (or un-conjugate) the Matrix for CBLAS functions.
*/
void
Matrix_base::conjugate() {

  conjugated = !conjugated;

}


/**
@brief Call to get whether the Matrix should be conjugated in CBLAS functions or not.
@return Returns with true if the Matrix should be conjugated in CBLAS functions or false otherwise.
*/
bool
Matrix_base::is_transposed() {

  return transposed;

}

/**
@brief Call to transpose (or un-transpose) the Matrix for CBLAS functions.
*/
void
Matrix_base::transpose()  {

  transposed = !transposed;

}


/**
@brief Call to get the pointer to the stored data
*/
QGD_Complex16*
Matrix_base::get_data() {

  return data;

}


/**
@brief Call to replace the stored data by an another data array. If the class was the owner of the original data array, then it is released.
@param data_in The data array to be set as a new storage.
@param owner_in Set true to set the current class instance to be the owner of the data array, or false otherwise.
*/
void
Matrix_base::replace_data( QGD_Complex16* data_in, bool owner_in) {

    release_data();
    data = data_in;
    owner = owner_in;

    reference_mutex = new tbb::spin_mutex;
    references = new int64_t;
    (*references)=1;

}


/**
@brief Call to release the data stored by the Matrix. (If the class instance was not the owner of the data, then the data pointer is simply set to NULL pointer.)
*/
void
Matrix_base::release_data() {

    if (references==NULL) return;
    bool call_delete = false;

{

    tbb::spin_mutex::scoped_lock my_lock{*reference_mutex};

    if (references==NULL) return;
    call_delete = ((*references)==1);


    if (call_delete) {
      // release the data when Matrix is the owner
      if (owner) {
        scalable_aligned_free(data);
      }
      delete references;
    }
    else {
        (*references)--;
    }

    data = NULL;
    references = NULL;

}

  if ( call_delete ) {
    delete reference_mutex;
  }

}



/**
@brief Call to set the current class instance to be (or not to be) the owner of the stored data array.
@param owner_in Set true to set the current class instance to be the owner of the data array, or false otherwise.
*/
void
Matrix_base::set_owner( bool owner_in)  {

    owner=owner_in;

}

/**
@brief Assignment operator.
@param mtx An instance of class Matrix
@return Returns with the instance of the class.
*/
void
Matrix_base::operator= (const Matrix_base& mtx ) {

  // releasing the containing data
  release_data();

  // The number of rows
  rows = mtx.rows;
  // The number of columns
  cols = mtx.cols;
  // pointer to the stored data
  data = mtx.data;
  // logical variable indicating whether the Matrix needs to be conjugated in CBLAS operations
  conjugated = mtx.conjugated;
  // logical variable indicating whether the Matrix needs to be transposed in CBLAS operations
  transposed = mtx.transposed;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  owner = mtx.owner;

  reference_mutex = mtx.reference_mutex;
  references = mtx.references;

  {
      tbb::spin_mutex::scoped_lock my_lock{*reference_mutex};
      (*references)++;
  }

}


/**
@brief Operator [] to access elements in array style (does not check the boundaries of the stored array)
@param idx the index of the element
@return Returns with a reference to the idx-th element.
*/
QGD_Complex16&
Matrix_base::operator[](size_t idx) {
    return data[idx];
}




/**
@brief Call to create a copy of the Matrix
@return Returns with the instance of the class.
*/
Matrix_base
Matrix_base::copy() {

  Matrix_base ret = Matrix_base(rows, cols);

  // logical variable indicating whether the Matrix needs to be conjugated in CBLAS operations
  ret.conjugated = conjugated;
  // logical variable indicating whether the Matrix needs to be transposed in CBLAS operations
  ret.transposed = transposed;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  ret.owner = true;

  memcpy( ret.data, data, rows*cols*sizeof(QGD_Complex16));

  return ret;

}



/**
@brief Call to get the number of the allocated elements
@return Returns with the number of the allocated elements (rows*cols)
*/
size_t
Matrix_base::size() {

  return rows*cols;

}


/**
@brief Call to prints the stored Matrix on the standard output
*/
void
Matrix_base::print_matrix() {
    std::cout << std::endl << "The stored Matrix:" << std::endl;
    for ( size_t row_idx=0; row_idx < rows; row_idx++ ) {
        for ( size_t col_idx=0; col_idx < cols; col_idx++ ) {
            size_t element_idx = row_idx*cols + col_idx;
            QGD_Complex16 element = data[element_idx];
              std::cout << " " << element.real << " + i*" << element.imag;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl << std::endl;

}





/**
@brief Call to create a copy of the Matrix
@return Returns with the instance of the class.
*/
Matrix
Matrix::copy() {

  Matrix ret = Matrix(rows, cols);

  // logical variable indicating whether the Matrix needs to be conjugated in CBLAS operations
  ret.conjugated = conjugated;
  // logical variable indicating whether the Matrix needs to be transposed in CBLAS operations
  ret.transposed = transposed;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  ret.owner = true;

  memcpy( ret.data, data, rows*cols*sizeof(QGD_Complex16));

  return ret;

}












