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
/*! \file qgd/Sub_Matrix_Decomposition.h
    \brief Header file for a class responsible for the disentanglement of one qubit from the others.
*/

#ifndef SUB_MATRIX_DECOMPOSITION_H
#define SUB_MATRIX_DECOMPOSITION_H

#include "Decomposition_Base.h"
#include "Sub_Matrix_Decomposition_Cost_Function.h"
#include "Functor_Cost_Function_Gradient.h"


/**
@brief A class responsible for the disentanglement of one qubit from the others.
*/
class Sub_Matrix_Decomposition : public Decomposition_Base {


public:

    /// logical value indicating whether the disentamglement of a qubit from the othetrs was done or not
    bool subdisentaglement_done;

    /// The subdecomposed matrix
    Matrix subdecomposed_mtx;

    /// logical value. Set true to optimize the minimum number of operation layers required in the decomposition, or false when the predefined maximal number of layer gates is used (ideal for general unitaries).
    bool optimize_layer_num;

    /// A map of <int n: int num> indicating that how many identical succesive blocks should be used in the disentanglement of the nth qubit from the others
    std::map<int,int> identical_blocks;

    /// Custom gate structure describing the gate structure used in the decomposition. The gate structure is repeated periodically in the decomposing gate structure
    Operation_block* unit_gate_structure;


public:

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
Sub_Matrix_Decomposition( );

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
Sub_Matrix_Decomposition( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, guess_type initial_guess_in );

/**
@brief Destructor of the class
*/
~Sub_Matrix_Decomposition();


/**
@brief Start the optimization process to disentangle the most significant qubit from the others. The optimized parameters and operations are stored in the attributes optimized_parameters and operations.
*/
void disentangle_submatrices();

/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
virtual void add_operation_layers();


/**
@brief Call to add default gate layers to the gate structure used in the subdecomposition.
*/
void add_default_operation_layers();


/**
@brief Call to set custom layers to the gate structure that are intended to be used in the subdecomposition.
@param block_in A pointer to the block of operations to be used in the decomposition
*/
void set_custom_operation_layers( Operation_block* block_in);



/**
@brief Call to solve layer by layer the optimization problem. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void solve_layer_optimization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl);




/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double optimization_problem( const double* parameters);



/**
@brief The optimization problem of the final optimization
@param parameters A GNU Scientific Library containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
static double optimization_problem( const gsl_vector* parameters, void* void_instance );


/**
@brief Calculate the approximate derivative (f-f0)/(x-x0) of the cost function with respect to the free parameters.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
*/
static void optimization_problem_grad( const gsl_vector* parameters, void* void_instance, gsl_vector* grad  );


/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
*/
static void optimization_problem_combined( const gsl_vector* parameters, void* void_instance, double* f0, gsl_vector* grad );

/**
@brief Set the number of identical successive blocks during the subdecomposition of the qbit-th qubit.
@param qbit The number of qubits for which the maximal number of layers should be used in the subdecomposition.
@param identical_blocks_in The number of successive identical layers used in the subdecomposition.
@return Returns with zero in case of success.
*/
int set_identical_blocks( int qbit, int identical_blocks_in );


/**
@brief Set the number of identical successive blocks during the subdecomposition of the n-th qubit.
@param identical_blocks_in An <int,int> map containing the number of successive identical layers used in the subdecompositions.
@return Returns with zero in case of success.
*/
int set_identical_blocks( std::map<int, int> identical_blocks_in );


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
Sub_Matrix_Decomposition* clone();


};


#endif //SUB_MATRIX_DECOMPOSITION
