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

@author: Peter Rakyta, Ph.D.
*/
/*! \file Sub_Matrix_Decomposition.h
    \brief Header file for a class responsible for the disentanglement of one qubit from the others.
*/

#ifndef SUB_MATRIX_DECOMPOSITION_H
#define SUB_MATRIX_DECOMPOSITION_H

#include "Decomposition_Base.h"



/**
@brief A class responsible for the disentanglement of one qubit from the others.
*/
class Sub_Matrix_Decomposition : public Decomposition_Base {


public:

    /// logical value indicating whether the disentamglement of a qubit from the othetrs was done or not
    bool subdisentaglement_done;

    /// The subdecomposed matrix
    Matrix subdecomposed_mtx;

    /// logical value. Set true to optimize the minimum number of gate layers required in the decomposition, or false when the predefined maximal number of layer gates is used (ideal for general unitaries).
    bool optimize_layer_num;

    /// A map of <int n: int num> indicating that how many identical succesive blocks should be used in the disentanglement of the nth qubit from the others
    std::map<int,int> identical_blocks;

    /// Custom gate structure describing the gate structure used in the decomposition. The gate structure is repeated periodically in the decomposing gate structure
    Gates_block* unit_gate_structure;

    /// maximal number of inner iterations
    long long max_inner_iterations;


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
Sub_Matrix_Decomposition( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, std::map<std::string, Config_Element>& config_in, guess_type initial_guess_in );

/**
@brief Destructor of the class
*/
~Sub_Matrix_Decomposition();


/**
@brief Start the optimization process to disentangle the most significant qubit from the others. The optimized parameters and gates are stored in the attributes optimized_parameters and gates.
*/
void disentangle_submatrices();

/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
virtual void add_gate_layers();


/**
@brief Call to add default gate layers to the gate structure used in the subdecomposition.
*/
void add_default_gate_layers();


/**
@brief Call to set custom layers to the gate structure that are intended to be used in the subdecomposition.
@param block_in A pointer to the block of gates to be used in the decomposition
*/
void set_custom_gate_layers( Gates_block* block_in);



/**
@brief Call to solve layer by layer the optimization problem. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void solve_layer_optimization_problem( int num_of_parameters, Matrix_real solution_guess_gsl);




/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double optimization_problem( double* parameters);



/**
@brief The optimization problem of the final optimization
@param parameters A GNU Scientific Library containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
static double optimization_problem( Matrix_real parameters, void* void_instance );


/**
@brief Calculate the approximate derivative (f-f0)/(x-x0) of the cost function with respect to the free parameters.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
*/
static void optimization_problem_grad( Matrix_real parameters, void* void_instance, Matrix_real& grad  );


/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
*/
static void optimization_problem_combined( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad );

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
