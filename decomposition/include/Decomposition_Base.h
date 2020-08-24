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
#include "Operation_block.h"
#include "CNOT.h"
#include "U3.h"
#include <map>
#include <cstdlib>
#include <time.h> 
#include <ctime>
#include "gsl/gsl_multimin.h"
#include "gsl/gsl_deriv.h"
#include <gsl/gsl_statistics.h>


struct deriv {
    int idx;
    const gsl_vector* parameters;
    void* instance;
};


////
// @brief A class containing basic methods for the decomposition process.
class Decomposition_Base : public Operation_block {


public:
bool verbose;

    // number of operator blocks in one sub-layer of the optimalization process
    int optimalization_block;

    // default number of layers in the decomposition as a function of number of qubits
    static std::map<int,int> max_layer_num_def;

    // The maximal allowed error of the optimalization problem
    double optimalization_tolerance;

protected:



    //  number of layers in the decomposition as a function of number of qubits
    std::map<int,int> max_layer_num;

    // number of iteratrion loops in the optimalization
    std::map<int,int> iteration_loops;

    // The unitary to be decomposed
    MKL_Complex16* Umtx;

    // The corrent optimized parameters for the operations
    double* optimized_parameters;
        
    // logical value describing whether the decomposition was finalized or not
    bool decomposition_finalized; 
        
    // error of the unitarity of the final decomposition
    double decomposition_error;
        
    // number of finalizing (deterministic) opertaions counted from the top of the array of operations
    long finalizing_operations_num;
        
    // the number of the finalizing (deterministic) parameters counted from the top of the optimized_parameters list
    int finalizing_parameter_num;
        
    // The current minimum of the optimalization problem
    double current_minimum;
        
    // The global minimum of the optimalization problem
    double global_target_minimum;
        
    // logical value describing whether the optimalization problem was solved or not
    bool optimalization_problem_solved;
        
    // Maximal number of iteartions in the optimalization process
    long max_iterations;
        
    // method to guess initial values for the optimalization. POssible values: 'zeros', 'random', 'close_to_zero'
    string initial_guess;

    // current minimum evaluated by the LBFGS library
    double* m_x;


public:

//// Contructor of the class
// @brief Constructor of the class.
// @param Umtx The unitary matrix to be decomposed
// @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: 'zeros' (deafult),'random', 'close_to_zero'
// @return An instance of the class
Decomposition_Base( MKL_Complex16*, int, string );

//// 
// @brief Destructor of the class
~Decomposition_Base();


////   
// @brief Call to set the number of operation layers to optimize in one shot
// @param optimalization_block The number of operation blocks to optimize in one shot 
void set_optimalization_blocks( int);
        
////   
// @brief Call to set the maximal number of the iterations in the optimalization process
// @param max_iterations aximal number of iteartions in the optimalization process
void set_max_iteration( long);


//// 
// @brief After the main optimalization problem is solved, the indepent qubits can be rotated into state |0> by this def. The constructed operations are added to the array of operations needed to the decomposition of the input unitary.
void finalize_decomposition();


////
// @brief Lists the operations decomposing the initial unitary. (These operations are the inverse operations of the operations bringing the intial matrix into unity.)
// @param start_index The index of the first inverse operation
void list_operations( int );

////
// @brief This method determine the operations needed to rotate the indepent qubits into the state |0>
// @param mtx The unitary describing indepent qubits.
// @return [1] The operations needed to rotate the qubits into the state |0>
// @return [2] The parameters of the U3 operations needed to rotate the qubits into the state |0>
// @return [3] The resulted diagonalized matrix.
MKL_Complex16* get_finalizing_operations( MKL_Complex16* mtx, Operation_block* & finalizing_operations, double* &finalizing_parameters);


//// 
// @brief This method can be used to solve the main optimalization problem which is devidid into sub-layer optimalization processes. (The aim of the optimalization problem is to disentangle one or more qubits) The optimalized parameters are stored in attribute @optimized_parameters.
// @param varargin Cell array of optional parameters:
// @param 'optimalization_problem' def handle of the cost def to be optimalized
// @param 'solution_guess' Array of guessed parameters
void solve_optimalization_problem();

////
// @brief This method can be used to solve a single sub-layer optimalization problem. The optimalized parameters are stored in attribute @optimized_parameters.
// @param 'solution_guess' Array of guessed parameters
// @param 'num_of_parameters' NUmber of free parameters to be optimized
virtual void solve_layer_optimalization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl);



////
// @brief This is an abstact def giving the cost def measuring the entaglement of the qubits. When the qubits are indepent, teh cost def should be zero.
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
virtual double optimalization_problem( const double* parameters );

////
// @brief This is an abstact def giving the cost def measuring the entaglement of the qubits. When the qubits are indepent, teh cost def should be zero.
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
//double optimalization_problem_grad( const double* parameters, void*, gsl_vector* );

////
// @brief This is an abstact def giving the cost def measuring the entaglement of the qubits. When the qubits are indepent, teh cost def should be zero.
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
//double optimalization_problem_combined( const double* parameters, void*, gsl_vector* );


//// check_optimalization_solution
// @brief Checks the convergence of the optimalization problem.
// @return Returns with true if the target global minimum was reached during the optimalization process, or false otherwise.
bool check_optimalization_solution();


////
// @brief Calculate the list of cumulated gate operation matrices such that the i>0-th element in the result list is the product of the operations of all 0<=n<i operations from the input list and the 0th element in the result list is the identity.
// @param parameters An array containing the parameters of the U3 operations.
// @param operations Iterator pointing to the first element in a vector of operations to be considered in the multiplications.
// @param num_of_operations The number of operations counted from the first element of the operations.
// @return Returns with a vector of the product matrices.
std::vector<MKL_Complex16*> get_operation_products(double* , std::vector<Operation*>::iterator, long );


//
// @brief Call to get the unitary to be transformed
// @return Return with a pointer pointing to the unitary
MKL_Complex16* get_Umtx();

//
// @brief Call to get the size of the unitary to be transformed
// @return Return with the size of the unitary
int get_Umtx_size();

//
// @brief Call to get the optimized parameters
// @return Return with the pointer pointing to the array storing the optimized parameters
double* get_optimized_parameters();


////
// @brief Calculate the transformed matrix resulting by an array of operations on a given initial matrix.
// @param parameters An array containing the parameters of the U3 operations.
// @param operations The array of the operations to be applied on a unitary
// @param initial_matrix The initial matrix wich is transformed by the given operations. (by deafult it is set to the attribute @Umtx)
// @return Returns with the transformed matrix.
MKL_Complex16* get_transformed_matrix( const double* parameters, std::vector<Operation*>::iterator operations, long num_of_operations, MKL_Complex16* initial_matrix );


////
// @brief Calculate the transformed matrix resulting by an array of operations on a given initial matrix.
// @return Returns with the decomposed matrix.
MKL_Complex16* get_decomposed_matrix();

////
// @brief Gives an array of permutation indexes that can be used to permute the basis in the N-qubit unitary according to the flip in the qubit order.
// @param qbit_list A list of the permutation of the qubits (for example [1 3 0 2])
// @retrun Returns with the reordering indexes of the basis     
std::vector<int> get_basis_of_reordered_qubits( vector<int> );

////
// @brief Call to reorder the qubits in the unitary to be decomposed (the qubits become reordeerd in the operations a well)        
// @param qbit_list A list of the permutation of the qubits (for example [1 3 0 2])
void reorder_qubits( vector<int> );

////
// @brief Apply an operations on the input matrix
// @param operation_mtx The matrix of the operation.
// @param input_matrix The input matrix to be transformed.
// @return Returns with the transformed matrix
MKL_Complex16* apply_operation( MKL_Complex16*, MKL_Complex16*  );

////
// @briefinitializes default layer numbers
static void Init_max_layer_num();

//double evaluate(const double *, double *, const int, const double);


//int progress(const double *x, const double *g, const double fx, const double xnorm, const double gnorm, const double step, int n, int k, int ls);




//static double _evaluate( void *instance, const double *x, double *g, const int n, const double step );


    

//static int _progress(void *instance, const double *x, const double *g, const double fx, const double xnorm, const double gnorm, const double step, int n, int k, int ls);



};
