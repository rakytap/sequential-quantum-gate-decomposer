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

#include "n_aryGrayCodeCounter.h"
#include <cstring>
#include <iostream>



/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
n_aryGrayCodeCounter::n_aryGrayCodeCounter()  {


   offset_max = 0;
   offset = 0;

}


/**
@brief Constructor of the class. 
@param n_ary_limits_in The maximal value of the individual gray code elements
*/
n_aryGrayCodeCounter::n_aryGrayCodeCounter( matrix_base<int>& n_ary_limits_in) {

    n_ary_limits = n_ary_limits_in.copy();

    if ( n_ary_limits.size() == 0 ) {
        offset_max = 0;
        offset = 0;
        return;
    }

    offset_max = n_ary_limits[0];
    for (size_t idx=1; idx<n_ary_limits.size(); idx++) {
        offset_max *= n_ary_limits[idx];
    }

    offset_max--;
    offset = 0;

    // initialize the counter
    initialize(0);

}


/**
@brief Constructor of the class. 
@param n_ary_limits_in The maximal value of the individual gray code elements
@param initial_offset The initial offset of the counter 0<=initial_offset<=offset_max
*/
n_aryGrayCodeCounter::n_aryGrayCodeCounter( matrix_base<int>& n_ary_limits_in, int64_t initial_offset) {

    n_ary_limits = n_ary_limits_in.copy();

    if ( n_ary_limits.size() == 0 ) {
        offset_max = 0;
        offset = 0;
        return;
    }

    offset_max = n_ary_limits[0];
    for (size_t idx=1; idx<n_ary_limits.size(); idx++) {
        offset_max *= n_ary_limits[idx];
    }

    offset_max--;
    offset = initial_offset;

    // initialize the counter
    initialize(initial_offset);

}




/**
@brief Initialize the gray counter by zero offset
*/
void 
n_aryGrayCodeCounter::initialize() {

    initialize(0);

}


/**
@brief Initialize the gray counter to specific initial offset
@param initial_offset The initial offset of the counter 0<= initial_offset <= offset_max
*/
void 
n_aryGrayCodeCounter::initialize( int64_t initial_offset ) {

    if ( initial_offset < 0 || initial_offset > offset_max ) {
        std::string error("n_aryGrayCodeCounter::initialize:  Wrong value of initial_offset");
        throw error;
    }

    // generate counter chain
    counter_chain = matrix_base<int>( 1, n_ary_limits.size() );

    for (size_t idx = 0; idx < n_ary_limits.size(); idx++) {
        counter_chain[idx] = initial_offset % n_ary_limits[idx];
        initial_offset /= n_ary_limits[idx]; 
    }

    // determine the initial gray code corresponding to the given offset
    gray_code = matrix_base<int>( 1, n_ary_limits.size() );
    int parity = 0;
    for (unsigned long long jdx = n_ary_limits.size()-1; jdx != ~0ULL; jdx--) {
        gray_code[jdx] = parity ? n_ary_limits[jdx] - 1 - counter_chain[jdx] : counter_chain[jdx];
        parity = parity ^ (gray_code[jdx] & 1);
    }



}



/**
@brief Get the current gray code counter value
@return Returns with the current gray code associated with the current offset.
*/
matrix_base<int>  
n_aryGrayCodeCounter::get() {


    return gray_code;

}


/**
@brief Iterate the counter to the next value
*/
int  
n_aryGrayCodeCounter::next() {

    int changed_index;

    int&& ret = next(changed_index);
    return ret;

}


/**
@brief Iterate the counter to the next value
@param changed_index The index of the gray code element where change occured.
*/
int  
n_aryGrayCodeCounter::next( int& changed_index) {


    int value_prev, value;
    int&& ret = next( changed_index, value_prev, value);
    return ret;

}

/**
@brief Iterate the counter to the next value
@param changed_index The index of the gray code element where change occured.
@param value_prev The previous value of the gray code element that changed.
@param value The new value of the gray code element.
*/
int   
n_aryGrayCodeCounter::next( int& changed_index, int& value_prev, int& value) {


    // determine the index which is about to modify
    changed_index = 0;

    if ( offset >= offset_max ) {
        return 1;
    }


    bool update_counter = true;
    int counter_chain_idx = 0;
    while( update_counter ) {

        if ( counter_chain[counter_chain_idx] < n_ary_limits[counter_chain_idx]-1 ) {
            counter_chain[counter_chain_idx]++;
            update_counter = false;
        }
        else if ( counter_chain[counter_chain_idx] == n_ary_limits[counter_chain_idx]-1 ) {
            counter_chain[counter_chain_idx] = 0;
            update_counter = true;
        }

        counter_chain_idx++;

    }


    // determine the updated gray code
    int parity = 0;
    for (size_t jdx = n_ary_limits.size()-1; jdx != ~0ULL; jdx--) {
        int gray_code_new_val = parity ? n_ary_limits[jdx] - 1 - counter_chain[jdx] : counter_chain[jdx];
        parity = parity ^ (gray_code_new_val & 1);

        if ( gray_code_new_val != gray_code[jdx] ) {
            value_prev = gray_code[jdx];
            value = gray_code_new_val;
            gray_code[jdx] = gray_code_new_val;
            changed_index = jdx;
            break;
        }
    }

    offset++;

    return 0;

}



void  
n_aryGrayCodeCounter::set_offset_max( const int64_t& value ) {


    offset_max = value;

}



