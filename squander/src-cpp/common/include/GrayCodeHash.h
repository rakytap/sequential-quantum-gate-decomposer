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
/*! \file GrayCodeHash.h
    \brief Header file for Gray code hashing
*/
#ifndef GRAYCODEHASH_H
#define GRAYCODEHASH_H

#include "GrayCodeHash_base.hpp"



/// alias for class to hash function operator for GrayCodeHash_int64
using GrayCodeHash_int64 = GrayCodeHash_base<int64_t>;


/// alias for class to hash function operator for GrayCodeHash
using GrayCodeHash = GrayCodeHash_base<int>;



#endif
