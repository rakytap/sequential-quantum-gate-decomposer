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
#include "qgd/Decomposition_Base.h"


////
// @brief A class containing basic methods for the decomposition process.

class Sub_Matrix_Decomposition : public Decomposition_Base {


public:

    // logical value indicating whether the quasi-unitarization of the submatrices was done or not 
    bool subdisentaglement_done;

    // The subunitarized matrix
    MKL_Complex16* subdecomposed_mtx;

    // logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
    bool optimize_layer_num;

    // The number of successive identical blocks in one leyer
    std::map<int,int> identical_blocks;


public:

//// Contructor of the class
// @brief Constructor of the class.
// @param Umtx The unitary matrix to be decomposed
// @param max_layer_num_in A map of <int n: int num> indicating that how many layers should be used in the subdecomposition process at the subdecomposing of n-th qubits.
// @param identical_blocks_in A map of <int n: int num> indicating that how many identical succesive blocks should be used in the disentanglement of the nth qubit from the others
// @param optimize_layer_num Optional logical value. If true, then the optimalization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimalization is performed for the maximal number of layers.
// @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: 'zeros' (deafult),'random', 'close_to_zero'
// @return An instance of the class
Sub_Matrix_Decomposition( MKL_Complex16*, int, std::map<int,int>, std::map<int,int>, bool, string );


////
// @brief Start the optimalization process to disentangle the most significant qubit from the others. The optimized parameters and operations are stored in the attributes @optimized_parameters and @operations.
void disentangle_submatrices();


////
// @brief This method can be used to solve a single sub-layer optimalization problem. The optimalized parameters are stored in attribute @optimized_parameters.
// @param 'solution_guess' Array of guessed parameters
// @param 'num_of_parameters' NUmber of free parameters to be optimized
void solve_layer_optimalization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl);




//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
double optimalization_problem( const double* );



//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
static double optimalization_problem( const gsl_vector*, void*  );

//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
static double optimalization_problem_deriv( double x, void*  );

////
// @brief This is an abstact def giving the cost def measuring the entaglement of the qubits. When the qubits are indepent, teh cost def should be zero.
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
static void optimalization_problem_grad( const gsl_vector* parameters, void*, gsl_vector*  );


////
// @brief This is an abstact def giving the cost def measuring the entaglement of the qubits. When the qubits are indepent, teh cost def should be zero.
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
static void optimalization_problem_grad( const gsl_vector* parameters, void*, gsl_vector*, double f0  );

////
// @brief This is an abstact def giving the cost def measuring the entaglement of the qubits. When the qubits are indepent, teh cost def should be zero.
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
static void optimalization_problem_combined( const gsl_vector* parameters, void* , double* , gsl_vector*  );


/*

// 
// @brief Check whether qubits are indepent or not
// @returns Return with true if qubits are disentangled, or false otherwise.
bool test_indepency();


//static double _evaluate( void *instance, const double *x, double *g, const int n, const double step );

//static int _progress(void *instance, const double *x, const double *g, const double fx, const double xnorm, const double gnorm, const double step, int n, int k, int ls);
*/
};
