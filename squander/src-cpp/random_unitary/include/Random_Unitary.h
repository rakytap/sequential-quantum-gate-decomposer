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
#include "U3.h"
#include "CNOT.h"
#include "matrix.h"
#include "matrix_float.h"
#include "matrix_real_float.h"
#include "logging.h"


/**
@brief Call to create a random unitary constructed by CNOT operation between randomly chosen qubits and by random U3 operations.
@param qbit_num The number of qubits spanning the unitary.
@param cnot_num The number of CNOT gates composing the random unitary.
@return Returns with the the constructed random unitary.
*/
Matrix few_CNOT_unitary( int qbit_num, int cnot_num);



/**
@brief A class to create general random unitary matrices according to arXiv:1303.5904v1.

Internal helpers (Omega, M, Q, E_alpha_beta, I_alpha_beta, gamma, kronecker,
convert_indexes) are implementation details and live in an anonymous namespace
in the .cpp; they are not part of the public API.
*/
class Random_Unitary : public logging {

public:
    /// The number of rows in the created unitary
    int dim;

    /**
    @brief Constructor.
    @param dim_in The number of rows in the random unitary to be created.
    */
    Random_Unitary( int dim_in );

    /**
    @brief Construct a random unitary with internally generated float64 parameters.
    @return The constructed random unitary.
    */
    Matrix Construct_Unitary_Matrix();

    /**
    @brief Construct a float64 unitary from explicit parameter arrays (arXiv:1303.5904v1).
    @param vartheta array of dim*(dim-1)/2 elements
    @param varphi   array of dim*(dim-1)/2 elements
    @param varkappa array of dim-1 elements
    @return The constructed float64 unitary.
    */
    Matrix Construct_Unitary_Matrix( double* vartheta, double* varphi, double* varkappa );

    /**
    @brief Construct a float32 unitary from explicit parameter arrays (native float32,
           no precision conversion).
    @param vartheta array of dim*(dim-1)/2 elements
    @param varphi   array of dim*(dim-1)/2 elements
    @param varkappa array of dim-1 elements
    @return The constructed float32 unitary.
    */
    Matrix_float Construct_Unitary_Matrix( float* vartheta, float* varphi, float* varkappa );

    /**
    @brief Construct a float64 unitary from a packed parameter array of
           (dim+1)*(dim-1) elements (vartheta | varphi | varkappa contiguous).
    @param parameters array of (dim+1)*(dim-1) elements
    @return The constructed float64 unitary.
    */
    Matrix Construct_Unitary_Matrix( double* parameters );

};




