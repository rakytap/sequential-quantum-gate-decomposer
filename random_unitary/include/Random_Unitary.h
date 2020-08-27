/*
Created on Fri Jun 26 14:13:26 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/

//
// @brief A base class responsible for constructing matrices of C-NOT gates
// gates acting on the N-qubit space

#pragma once
#include <map>
#include <cstdlib>
#include <time.h> 
#include <ctime>
#include "qgd/common.h"




////
// @brief A class containing basic methods for the decomposition process.
class Random_Unitary {


public:

    int dim;



public:

//// Contructor of the class
//> @brief Constructor of the class.
//> @param ??????????????????
//> @return An instance of the class
Random_Unitary( int dim_in );   
 

//// Construct_Unitary_Matrix
//> @brief Constructor of the class.
//> @param parameters array of (dim+1)*(dim-1) elements
//> @return The constructed matrix
MKL_Complex16* Construct_Unitary_Matrix();    
    
//// Construct_Unitary_Matrix
//> @brief Constructor of the class.
//> @param vartheta array of dim*(dim-1)/2 elements
//> @param varphi array of dim*(dim-1)/2 elements
//> @param varkappa array of dim-1 elements
//> @return The constructed matrix
MKL_Complex16* Construct_Unitary_Matrix( double* vartheta, double* varphi, double* varkappa );

int  convert_indexes( int varalpha, int varbeta );
    
//// Construct_Unitary_Matrix
//> @brief Constructor of the class.
//> @param parameters array of (dim+1)*(dim-1) elements
//> @return The constructed matrix
MKL_Complex16* Construct_Unitary_Matrix( double* parameters );
    
//// Omega
//> @brief Eq (6)
MKL_Complex16* Omega(int varalpha, int varbeta, MKL_Complex16 x, MKL_Complex16 y );  
    
    
//// M
//> @brief Eq (8)
MKL_Complex16* M( int varalpha, int varbeta, MKL_Complex16 s, MKL_Complex16 t );
    
//// Q
//> @brief Eq (9)
MKL_Complex16* Q(  MKL_Complex16 u1, MKL_Complex16 u2 );
    
    
//// E_n_m
//> @brief below Eq (7)
MKL_Complex16* E_alpha_beta( int varalpha, int varbeta );


//// I_n
//> @brief below Eq (7)
MKL_Complex16* I_alpha_beta( int varalpha, int varbeta );
    
//// 
//> @brief Eq (11)
double gamma();
    
////
//> @brief Eq (11)
double kronecker( int a, int b );


};
    



