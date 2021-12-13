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
/*! \file RZ.h
    \brief Header file for a class representing a rotation gate around the Z axis.
*/

#ifndef RZ_H
#define RZ_H

#include "U3.h"
#include "matrix.h"
#include "matrix_real.h"
#include <math.h>


/**
@brief A class representing a RZ gate.
*/
class RZ: public U3 {


public:

/**
@brief NullaRZ constructor of the class.
*/
RZ();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
RZ(int qbit_num_in, int target_qbit_in);

/**
@brief Destructor of the class
*/
~RZ();


/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void apply_to( Matrix_real& parameters, Matrix& input );


/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void apply_from_right( Matrix_real& parameters, Matrix& input );



/**
@brief Call to set the final optimized parameters of the gate.
@param Phi Real parameter standing for the parameter phi.
*/
void set_optimized_parameters( double Phi );

/**
@brief Call to get the final optimized parameters of the gate.
@param parameters_in Preallocated pointer to store the parameters Theta, Phi and Lambda of the U3 gate.
*/
Matrix_real get_optimized_parameters();

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
RZ* clone();


};


#endif //U3

