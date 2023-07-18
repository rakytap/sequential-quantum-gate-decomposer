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
/*! \file SX.h
    \brief Header file for a class representing a rotation gate around the X axis.
*/

#ifndef SX_H
#define SX_H


#include "U3.h"
//#include "Gate.h"
#include "matrix.h"
#include <math.h>



/**
@brief A class representing a U3 gate.
*/
class SX: public U3 {


public:

/**
@brief NullaRX constructor of the class.
*/
SX();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
SX(int qbit_num_in, int target_qbit_in);

/**
@brief Destructor of the class
*/
~SX();

/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@return Returns with a matrix of the gate
*/
Matrix get_matrix( );


/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void apply_to( Matrix& input );


/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void apply_from_right( Matrix& input );


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
SX* clone();


/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
void set_qbit_num(int qbit_num_in);



/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void reorder_qubits( std::vector<int> qbit_list);


/**
@brief Set static values for matrix of the gates.
@param u3_1qbit Matrix parameter for the gate.

*/
void parameters_for_calc_one_qubit( Matrix& u3_1qbit);

};


#endif //U3

