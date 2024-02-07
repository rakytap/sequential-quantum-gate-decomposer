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
#include "logging.h"


/**
@brief Call to create a random unitary constructed by CNOT operation between randomly chosen qubits and by random U3 operations.
@param qbit_num The number of qubits spanning the unitary.
@param cnot_num The number of CNOT gates composing the random unitary.
@return Returns with the the constructed random unitary.
*/
Matrix few_CNOT_unitary( int qbit_num, int cnot_num);



/**
@brief A class to cerate general random unitary matrix according to arXiv:1303:5904v1
*/
class Random_Unitary : public logging {


public:
    /// The number of rows in the created unitary
    int dim;



public:

/**
@brief Constructor of the class.
@param dim_in The number of rows in the random unitary to be ceated.
@return An instance of the class
*/
Random_Unitary( int dim_in );


/**
@brief Call to create a random unitary
@return Returns with a pointer to the created random unitary
*/
Matrix Construct_Unitary_Matrix();

/**
@brief Generates a unitary matrix from parameters vartheta, varphi, varkappa according to arXiv:1303:5904v1
@param vartheta array of dim*(dim-1)/2 elements
@param varphi array of dim*(dim-1)/2 elements
@param varkappa array of dim-1 elements
@return Returns with a pointer to the generated unitary
*/
Matrix Construct_Unitary_Matrix( double* vartheta, double* varphi, double* varkappa );


/**
@brief Calculates an index from paramaters varalpha and varbeta
@param varalpha An integer
@param varbeta An integer
@return Returns with the calculated index.
*/
int  convert_indexes( int varalpha, int varbeta );

/**
@brief Generates a unitary matrix from parameters parameters according to arXiv:1303:5904v1
@param parameters array of (dim+1)*(dim-1) elements
@return The constructed unitary
*/
Matrix Construct_Unitary_Matrix( double* parameters );

/**
@brief Eq (6) of arXiv:1303:5904v1
@param varalpha An integer
@param varbeta An integer
@param x A complex number
@param y A complex number
@return Return with a pointer to the calculated Omega matrix of Eq. (6) of arXiv:1303:5904v1
*/
Matrix Omega(int varalpha, int varbeta, QGD_Complex16 x, QGD_Complex16 y );


/**
@brief Implements Eq (8) of arXiv:1303:5904v1
@param varalpha An integer
@param varbeta An integer
@param s A complex number
@param t A complex number
@return Return with a pointer to the calculated M matrix of Eq. (8) of arXiv:1303:5904v1
*/
Matrix M( int varalpha, int varbeta, QGD_Complex16 s, QGD_Complex16 t );

/**
@brief Implements Eq (9) of arXiv:1303:5904v1
@param u1 A complex number
@param u2 A complex number
@return Return with a pointer to the calculated Q matrix of Eq. (9) of arXiv:1303:5904v1
*/
Matrix Q(  QGD_Complex16 u1, QGD_Complex16 u2 );


/**
@brief Implements matrix I below Eq (7) of arXiv:1303:5904v1
@param varalpha An integer
@param varbeta An integer
@return Return with a pointer to the calculated E matrix of Eq. (7) of arXiv:1303:5904v1
*/
Matrix E_alpha_beta( int varalpha, int varbeta );


/**
@brief Implements matrix I below Eq (7) of arXiv:1303:5904v1
@param varalpha An integer
@param varbeta An integer
@return Return with a pointer to the calculated I matrix of Eq. (7) of arXiv:1303:5904v1
*/
Matrix I_alpha_beta( int varalpha, int varbeta );

/**
@brief Implements Eq (11) of arXiv:1303:5904v1
@return Returns eith the value of gamma
*/
double gamma();

/**
@brief Kronecker delta
@param a An integer
@param b An integer
@return Returns with the Kronecker delta value of a and b.
*/
double kronecker( int a, int b );


};




