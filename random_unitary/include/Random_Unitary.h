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
/*! \file qgd/Random_Unitary.h
    \brief Header file for a class and methods to cerate random unitary matrices
*/

#pragma once
#include <map>
#include <cstdlib>
#include <time.h> 
#include <ctime>
#include "qgd/common.h"
#include "qgd/U3.h"
#include "qgd/CNOT.h"



/**
@brief Call to create a random unitary constructed by CNOT operation between randomly chosen qubits and by random U3 operations.
@param qbit_num The number of qubits spanning the unitary.
@param cnot_num The number of CNOT gates composing the random unitary.
@param mtx The preallocated array for the constructed unitary.
*/
void few_CNOT_unitary( int qbit_num, int cnot_num, QGD_Complex16* mtx);



/**
@brief A class to cerate general random unitary matrix according to arXiv:1303:5904v1
*/
class Random_Unitary {


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
QGD_Complex16* Construct_Unitary_Matrix();    
    
/**
@brief Generates a unitary matrix from parameters vartheta, varphi, varkappa according to arXiv:1303:5904v1
@param vartheta array of dim*(dim-1)/2 elements
@param varphi array of dim*(dim-1)/2 elements
@param varkappa array of dim-1 elements
@return Returns with a pointer to the generated unitary
*/
QGD_Complex16* Construct_Unitary_Matrix( double* vartheta, double* varphi, double* varkappa );


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
QGD_Complex16* Construct_Unitary_Matrix( double* parameters );
    
/**
@brief Eq (6) of arXiv:1303:5904v1
@param varalpha An integer
@param varbeta An integer
@param x A complex number
@param y A complex number
@return Return with a pointer to the calculated Omega matrix of Eq. (6) of arXiv:1303:5904v1
*/
QGD_Complex16* Omega(int varalpha, int varbeta, QGD_Complex16 x, QGD_Complex16 y );  
    
    
/**
@brief Implements Eq (8) of arXiv:1303:5904v1
@param varalpha An integer
@param varbeta An integer
@param s A complex number
@param t A complex number
@return Return with a pointer to the calculated M matrix of Eq. (8) of arXiv:1303:5904v1
*/
QGD_Complex16* M( int varalpha, int varbeta, QGD_Complex16 s, QGD_Complex16 t );
    
/**
@brief Implements Eq (9) of arXiv:1303:5904v1 
@param u1 A complex number
@param u2 A complex number
@return Return with a pointer to the calculated Q matrix of Eq. (9) of arXiv:1303:5904v1
*/
QGD_Complex16* Q(  QGD_Complex16 u1, QGD_Complex16 u2 );
    
    
/**
@brief Implements matrix I below Eq (7) of arXiv:1303:5904v1 
@param varalpha An integer
@param varbeta An integer
@return Return with a pointer to the calculated E matrix of Eq. (7) of arXiv:1303:5904v1
*/
QGD_Complex16* E_alpha_beta( int varalpha, int varbeta );


/**
@brief Implements matrix I below Eq (7) of arXiv:1303:5904v1 
@param varalpha An integer
@param varbeta An integer
@return Return with a pointer to the calculated I matrix of Eq. (7) of arXiv:1303:5904v1
*/
QGD_Complex16* I_alpha_beta( int varalpha, int varbeta );
    
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
    



