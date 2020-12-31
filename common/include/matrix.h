#ifndef Matrix_H
#define Matrix_H

#include <cstring>
#include <iostream>
#include <tbb/scalable_allocator.h>
#include <tbb/tbb.h>
#include "qgd/QGDTypes.h"


/**
@brief Class to store data of complex arrays and its properties.
*/
class Matrix_base {

public:
  /// The number of rows
  size_t rows;
  /// The number of columns
  size_t cols;
  /// pointer to the stored data
  QGD_Complex16* data;

protected:

  /// logical variable indicating whether the Matrix needs to be conjugated in CBLAS operations
  bool conjugated;
  /// logical variable indicating whether the Matrix needs to be transposed in CBLAS operations
  bool transposed;
  /// logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  bool owner;
  /// mutual exclusion to count the references for class instances referring to the same data.
  tbb::spin_mutex* reference_mutex;
  /// the number of the current references of the present object
  int64_t* references;
  /// padding bytes. Useful to avoid false sharing when a concurrent list of Matrix objects is about to used.
  //uint8_t padding[CACHELINE-48];



public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
Matrix_base();


/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param rows_in The number of rows in the stored Matrix
@param cols_in The number of columns in the stored Matrix
@return Returns with the instance of the class.
*/
Matrix_base( QGD_Complex16* data_in, size_t rows_in, size_t cols_in);



/**
@brief Constructor of the class. Allocates data for Matrix rows_in times cols_in. By default the created instance would be the owner of the stored data.
@param rows_in The number of rows in the stored Matrix
@param cols_in The number of columns in the stored Matrix
@return Returns with the instance of the class.
*/
Matrix_base( size_t rows_in, size_t cols_in);


/**
@brief Copy constructor of the class. The new instance shares the stored memory with the input Matrix. (Needed for TBB calls)
@param An instance of class Matrix to be copied.
*/

Matrix_base(const Matrix_base &in);



/**
@brief Destructor of the class
*/
~Matrix_base();

/**
@brief Call to get whether the Matrix should be conjugated in CBLAS functions or not.
@return Returns with true if the Matrix should be conjugated in CBLAS functions or false otherwise.
*/
bool is_conjugated();

/**
@brief Call to conjugate (or un-conjugate) the Matrix for CBLAS functions.
*/
void conjugate();


/**
@brief Call to get whether the Matrix should be conjugated in CBLAS functions or not.
@return Returns with true if the Matrix should be conjugated in CBLAS functions or false otherwise.
*/
bool is_transposed();

/**
@brief Call to transpose (or un-transpose) the Matrix for CBLAS functions.
*/
void transpose();


/**
@brief Call to get the pointer to the stored data
*/
QGD_Complex16* get_data();



/**
@brief Call to replace the stored data by an another data array. If the class was the owner of the original data array, then it is released.
@param data_in The data array to be set as a new storage.
@param owner_in Set true to set the current class instance to be the owner of the data array, or false otherwise.
*/
void replace_data( QGD_Complex16* data_in, bool owner_in);


/**
@brief Call to release the data stored by the Matrix. (If the class instance was not the owner of the data, then the data pointer is simply set to NULL pointer.)
*/
void release_data();



/**
@brief Call to set the current class instance to be (or not to be) the owner of the stored data array.
@param owner_in Set true to set the current class instance to be the owner of the data array, or false otherwise.
*/
void set_owner( bool owner_in);

/**
@brief Assignment operator.
@param mtx An instance of class Matrix
@return Returns with the instance of the class.
*/
void operator= (const Matrix_base& mtx );


/**
@brief Operator [] to access elements in array style (does not check the boundaries of the stored array)
@param idx the index of the element
@return Returns with a reference to the idx-th element.
*/
QGD_Complex16& operator[](size_t idx);


/**
@brief Call to create a copy of the Matrix
@return Returns with the instance of the class.
*/
Matrix_base copy();



/**
@brief Call to get the number of the allocated elements
@return Returns with the number of the allocated elements (rows*cols)
*/
size_t size();


/**
@brief Call to prints the stored Matrix on the standard output
*/
void print_matrix();




}; //Matrix


/**
@brief Class to store data of complex arrays and its properties.
*/

class Matrix : public Matrix_base {

private:
  /// padding bytes. Useful to avoid false/true sharing between Matrix objects in concurrent containers.
  uint8_t padding[CACHELINE-sizeof(Matrix_base)];


public:
  // inherit all the constructors of Matrix_base
  using Matrix_base::Matrix_base;


/**
@brief Call to create a copy of the Matrix
@return Returns with the instance of the class.
*/
Matrix copy();


}; //Matrix


#endif
