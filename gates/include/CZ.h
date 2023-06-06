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
/*! \file CZ.h
    \brief Header file for a class representing a CZ operation.
*/

#ifndef CZ_H
#define CZ_H

#include "matrix.h"
#include "Gate.h"
#include <math.h>
#include "CNOT.h"


/**
@brief A class representing a CZ operation.
*/
class CZ: public CNOT {

protected:


public:

/**
@brief Nullary constructor of the class.
*/
CZ();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
CZ(int qbit_num_in, int target_qbit_in,  int control_qbit_in);

/**
@brief Destructor of the class
*/
~CZ();

/**
@brief Call to retrieve the operation matrix
@return Returns with the matrix of the operation
*/
Matrix get_matrix();

/**
@brief Call to apply the gate on the input array/matrix CZ*input
@param input The input array on which the gate is applied
*/
void apply_to( Matrix& input );

/**
@brief Call to apply the gate on the input array/matrix by input*CZ
@param input The input array on which the gate is applied
*/
void apply_from_right( Matrix& input );

/**
@brief Call to set the number of qubits spanning the matrix of the operation
@param qbit_num The number of qubits
*/
void set_qbit_num(int qbit_num);

/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void reorder_qubits( std::vector<int> qbit_list);


/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on a single qbit space.
@param Theta Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
void parameters_for_calc_one_qubit( Matrix& u3_1qbit, double& ThetaOver2, double& Phi, double& Lambda);

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CZ* clone();

};

#endif //CZ
