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
    gray_code = GrayCode( n_ary_limits );
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
GrayCode  
n_aryGrayCodeCounter::get() {


    return gray_code.copy();

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

/**
@brief Advance the n-ary Gray code counter by incrementing the digit at a specified position.
@param counter_pos The index of the digit to attempt to increment (least-significant digit = 0).
@return The number of states skipped in the mixed-radix sequence, or 0 if no forward state remains.

This method advances the counter by trying to increment the digit at @p counter_pos.
If that digit is already at its maximum, a carry is propagated rightward until a higher
digit can be incremented. When no further forward state exists within the current
lexicographic slab, the method returns 0.

After a successful advance, the mixed-radix offset is recomputed and the internal
state (counter chain and Gray code) is reinitialized to match the new offset.
*/
int64_t n_aryGrayCodeCounter::advance(int counter_pos) {
    const int L = (int)n_ary_limits.size();
    if (L == 0) return 0;
    if (counter_pos < 0) counter_pos = 0;
    if (counter_pos >= L) counter_pos = L - 1;

    // Try to bump digit at counter_pos; if not possible, carry left
    int p = counter_pos;
    if (counter_chain[p] + 1 < n_ary_limits[p]) {
        counter_chain[p] += 1;
        if (p > 0) std::fill(counter_chain.data, counter_chain.data + p, 0);
    } else {
        // carry left: find the rightmost position < p that can be increased
        int r = p + 1;
        while (r < L && counter_chain[r] + 1 >= n_ary_limits[r]) ++r;
        if (r < L) {
            counter_chain[r] += 1;
            std::fill(counter_chain.data, counter_chain.data + r, 0);
        } else {
            // no forward state remains in this lex-slab
            return 0;
        }
    }

    // Compute the new mixed-radix rank (offset) from digits d (LSD at index 0).
    int64_t new_offset = 0;
    int64_t mul = 1;
    for (int j = 0; j < L; ++j) {
        new_offset += mul * (int64_t)counter_chain[j];
        mul *= (int64_t)n_ary_limits[j];
    }

    //printf("Advancing from offset %lld to offset %lld max %lld\n", offset, new_offset, offset_max);
    // If somehow not moving forward, do nothing
    if (new_offset <= offset || new_offset > offset_max) return 0;

    // Reinitialize counter to the new offset (rebuilds counter_chain & gray_code)
    initialize(new_offset);

    // Return exact number of states skipped
    return new_offset - offset; // note: initialize() set offset=new_offset
}

void  
n_aryGrayCodeCounter::set_offset_max( const int64_t& value ) {


    offset_max = value;

}



