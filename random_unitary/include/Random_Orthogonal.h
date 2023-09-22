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
/*! \file qgd/Random_Unitary.h
    \brief Header file for a class and methods to cerate random unitary matrices
*/

#pragma once
#include <map>
#include <cstdlib>
#include <time.h>
#include <ctime>
#include "common.h"
#include "matrix.h"
#include "matrix_real.h"
#include "logging.h"



/**
@brief A class to cerate general random unitary matrix according to arXiv:1303:5904v1
*/
class Random_Orthogonal : public logging {


public:
    /// The number of rows in the created unitary
    int dim;



public:

/**
@brief Constructor of the class.
@param dim_in The number of rows in the random unitary to be ceated.
@return An instance of the class
*/
Random_Orthogonal( int dim_in );


/**
@brief Call to create a random unitary
@return Returns with a pointer to the created random unitary
*/
Matrix Construct_Orthogonal_Matrix();

/**
@brief Generates a unitary matrix from parameters vartheta, varphi, varkappa according to arXiv:1303:5904v1
@param vartheta array of dim*(dim-1)/2 elements
@param varphi array of dim*(dim-1)/2 elements
@param varkappa array of dim-1 elements
@return Returns with a pointer to the generated unitary
*/
Matrix Construct_Orthogonal_Matrix( Matrix_real &vargamma );

};




