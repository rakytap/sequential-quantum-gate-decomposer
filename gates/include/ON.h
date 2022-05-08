/*
Created on Fri JON 26 14:13:26 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it ONder the terms of the GNU General Public License as published by
the Free Software FoONdation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file ON.h
    \brief Header file for a class for the representation of general gate operations on the first qbit_num-1 qubits.
*/

#ifndef ON_H
#define ON_H

#include <vector>
#include "common.h"
#include "matrix.h"
#include "matrix_real.h"
#include "Gate.h"

/**
@brief Base class for the representation of general gate operations.
*/
class ON : public Gate {


protected:

   /// Parameters theta, phi, lambda of the U3 gate after the decomposition of the ONitary is done
   Matrix_real parameters;

public:

/**
@brief Default constructor of the class.
@return An instance of the class
*/
ON();

/**
@brief Destructor of the class
*/
virtual ~ON();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the ONitaries
@return An instance of the class
*/
ON(int qbit_num_in);

/**
@brief Call to retrieve the operation matrix
@return Returns with a matrix of the operation
*/
Matrix get_matrix(Matrix_real& parameters);


/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
*/
void apply_to( Matrix_real& parameters, Matrix& input );


/**
@brief ?????
*/
Matrix get_submatrix( Matrix_real& parameters );

/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
void apply_from_right( Matrix_real& parameters, Matrix& input );



/**
@brief Set the number of qubits spanning the matrix of the operation
@param qbit_num_in The number of qubits spanning the matrix
*/
virtual void set_qbit_num( int qbit_num_in );

/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
virtual void reorder_qubits( std::vector<int> qbit_list );


/**
@brief Call to set the final optimized parameters of the gate.
@param parameters_ Real array of the optimized parameters
*/
void set_optimized_parameters( Matrix_real parameters_ );

/**
@brief Call to get the final optimized parameters of the gate.
*/
Matrix_real get_optimized_parameters();

/**
@brief Call to get the number of free parameters
@return Return with the number of the free parameters
*/
int get_parameter_num();


/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
gate_type get_type();

/**
@brief Call to get the number of qubits composing the ONitary
@return Return with the number of qubits composing the ONitary
*/
int get_qbit_num();

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
ON* clone();

};


#endif //OPERATION
