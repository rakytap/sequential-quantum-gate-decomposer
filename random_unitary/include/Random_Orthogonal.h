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
#include "common.h"
#include "matrix.h"
#include "matrix_real.h"




/**
@brief A class to cerate general random unitary matrix according to arXiv:1303:5904v1
*/
class Random_Orthogonal {


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




