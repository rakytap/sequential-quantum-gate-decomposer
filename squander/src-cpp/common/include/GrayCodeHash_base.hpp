/*
Created on Fri Jun 26 14:13:26 2020
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
/*! \file GrayCodeHash_base.hpp
    \brief Header file for Gray code hashing
*/

#ifndef GRAYCODEHASH_BASE_HPP
#define GRAYCODEHASH_BASE_HPP

#include "GrayCode_base.hpp"



/**
@brief Class to hash function operator for GrayCode_base keys in unordered maps and unordered sets
*/
template <typename intType>
class GrayCodeHash_base {

protected:

public:

/**
@brief Constructor of the class.
@return Returns with the instance of the class.
*/
GrayCodeHash_base() {

}

/**
@brief Operator to generate hash key for class instance PicState_base<intType>
@param key An instance of class PicState
@return Returns with the calculated hash value.
*/
size_t
operator()(const GrayCode_base<intType> &key) const {

    GrayCode_base<intType> &key_loc = const_cast<GrayCode_base<intType> &>(key);
    intType *data = key_loc.get_data();
    size_t hash_val = 0;

    matrix_base<intType>&& n_ary_limits = key.get_Limits();
    size_t pow = 1;

    for (size_t idx=0; idx<key.cols; idx++) {
        hash_val = hash_val + data[idx]*pow;
        pow = pow*n_ary_limits[idx];
    }

    return hash_val;
}


}; //GrayCodeHash_base





#endif // GRAYCODEHASH_BASE_HPP
