/**
 * Copyright 2021 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GRAYCODE_BASE_H
#define GRAYCODE_BASE_H

#include "matrix_base.hpp"




/**
@brief Class to store one-dimensional state vectors and their additional properties.
*/
template <typename intType>
class GrayCode_base : public matrix_base<intType> {

public:

#if CACHELINE>=64
private:
    /// padding class object to cache line borders
    uint8_t padding[CACHELINE-sizeof(matrix_base<intType>)-sizeof(intType)];
#endif

    /// the limits of the gray code digits
    matrix_base<intType> n_ary_limits;


public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GrayCode_base() : matrix_base<intType>() {
    this->rows = 0;
    this->cols = 0;

    n_ary_limits = matrix_base<intType>();
}

/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
GrayCode_base( intType* data_in, matrix_base<intType> n_ary_limits_) : matrix_base<intType>(data_in, 1, n_ary_limits_.size()) {

    n_ary_limits = n_ary_limits_;

}


/**
@brief Constructor of the class. Allocates data for the state of elements cols. By default the created instance would be the owner of the stored data.
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
GrayCode_base( matrix_base<intType> n_ary_limits_) : matrix_base<intType>(1, n_ary_limits_.size()) {

    n_ary_limits = n_ary_limits_;

}


/**
@brief Constructor of the class. Allocates data for the state of elements cols and set the values to value. By default the created instance would be the owner of the stored data.
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
GrayCode_base( intType value, matrix_base<intType>& n_ary_limits_) : matrix_base<intType>(1, n_ary_limits_.size()) {

    if (value == 0) {
        memset(this->data, value, n_ary_limits_.size()*sizeof(intType));
    }
    else {
        for (size_t idx=0; idx < this->size(); idx ++) {
            this->data[idx] = value;
        }
    }

    n_ary_limits = n_ary_limits_;

}


/**
@brief Copy constructor of the class. The new instance shares the stored memory with the input matrix. (Needed for TBB calls)
@param An instance of class matrix to be copied.
*/
GrayCode_base(const GrayCode_base &in) : matrix_base<intType>(in) {

    n_ary_limits = in.n_ary_limits;

}

/**
@brief Operator to compare two keys made of PicState_base class instances.
@param state An instance of an implemented class of PicState_base
@return Returns with true if the two keys are equal, or false otherwise
*/
bool
operator==( const GrayCode_base &gcode) const {

    if (this->cols != gcode.cols) {
        return false;
    }

    intType *data_this = this->data;
    intType *data = gcode.data;

    for (size_t idx=0; idx<this->cols; idx++) {
        if ( data_this[idx] != data[idx] ) {
            return false;
        }
    }

    n_ary_limits = gcode.n_ary_limits;

    return true;

}


/**
@brief Operator [] to access elements in array style.
@param idx the index of the element
@return Returns with a reference to the idx-th element.
*/
intType&
operator[](size_t idx) {
    return this->data[idx];
}

/**
@brief Operator [] to access a constant element in array style of constant instance.
@param idx the index of the element
@return Returns with a reference to the idx-th element.
*/
const intType&
operator[](size_t idx) const {
    return this->data[idx];
}



/**
@brief Overloaded assignment operator to create a copy of the state
@param state An instance of PicState_base
@return Returns with the instance of the class.
*/
void
operator= (const GrayCode_base &gcode ) {

    matrix_base<intType>::operator=( gcode );

    n_ary_limits = gcode.n_ary_limits;

}


/**
@brief Call to create a copy of the state. By default the created instance would be the owner of the stored array.
@return Returns with the instance of the class.
*/
GrayCode_base
copy() const {

     GrayCode_base ret = GrayCode_base(this->n_ary_limits);

    // logical variable indicating whether the matrix needs to be conjugated in CBLAS operations
    ret.conjugated = this->conjugated;
    // logical variable indicating whether the matrix needs to be transposed in CBLAS operations
    ret.transposed = this->transposed;
    // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
    ret.owner = true;

    memcpy( ret.data, this->data, this->rows*this->cols*sizeof(intType));


    ret.n_ary_limits = n_ary_limits.copy();

    return ret;

}





};



#endif // GRAYCODE_BASE_HPP
