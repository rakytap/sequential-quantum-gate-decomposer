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

#ifndef n_aryGrayCodeCounter_H
#define n_aryGrayCodeCounter_H

#include "matrix_base.h"


/**
@brief Class iterate over n-ary reflected gray codes
*/
class n_aryGrayCodeCounter {


protected:

    /// the current gray code associated to the offset value
    matrix_base<int> gray_code;
    /// The maximal value of the individual gray code elements
    matrix_base<int> n_ary_limits;
    /// The incremental counter chain associated to the gray code
    matrix_base<int> counter_chain;
    /// the maximal offset in the counter offset = prod( n_ary_limits[i] )
    int64_t offset_max;
    /// the current offset in the counter 0<= offset <= offset_max
    int64_t offset;

public:


/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
n_aryGrayCodeCounter();

/**
@brief Constructor of the class. 
@param n_ary_limits_in The maximal value of the individual gray code elements
*/
n_aryGrayCodeCounter( matrix_base<int>& n_ary_limits_in);


/**
@brief Constructor of the class. 
@param n_ary_limits_in The maximal value of the individual gray code elements
@param initial_offset The initial offset of the counter 0<=initial_offset<=offset_max
*/
n_aryGrayCodeCounter( matrix_base<int>& n_ary_limits_in, int64_t initial_offset);



/**
@brief Initialize the gray counter by zero offset
*/
void initialize();


/**
@brief Initialize the gray counter to specific initial offset
@param initial_offset The initial offset of the counter 0<= initial_offset <= offset_max
*/
void initialize( int64_t initial_offset );


/**
@brief Get the current gray code counter value
@return Returns with the current gray code associated with the current offset.
*/
matrix_base<int> get();


/**
@brief Iterate the counter to the next value
*/
int next();


/**
@brief Iterate the counter to the next value
@param changed_index The index of the gray code element where change occured.
*/
int next( int& changed_index);


/**
@brief Iterate the counter to the next value
@param changed_index The index of the gray code element where change occured.
@param value_prev The previous value of the gray code element that changed.
@param value The new value of the gray code element.
*/
int next( int& changed_index, int& value_prev, int& value);


void set_offset_max( const int64_t& value );


}; //n_aryGrayCodeCounter




#endif
