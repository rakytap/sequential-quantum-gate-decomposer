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
/*! \file qgd/Decomposition_Base.h
    \brief Header file for a class containing basic methods for the decomposition process.
*/

#ifndef DECOMPOSITION_BASE_H
#define DECOMPOSITION_BASE_H


#include "Operation_block.h"
#include "CNOT.h"
#include "U3.h"
#include <map>
#include <cstdlib>
#include <time.h>
#include <ctime>
#include "gsl/gsl_multimin.h"
#include "gsl/gsl_statistics.h"


/// @brief Type definition of the types of the initial guess
typedef enum guess_type {ZEROS, RANDOM, CLOSE_TO_ZERO} guess_type;



/**
@brief A class containing basic methods for the decomposition process.
*/
class Decomposition_Base : public Operation_block {


public:
    /// Logical variable. Set true for verbose mode, or to false to suppress output messages.
    bool verbose;

    /// number of operation blocks used in one shot of the optimization process
    int optimization_block;

    /// A map of <int n: int num> indicating that how many layers should be used in the subdecomposition process by default for the subdecomposing of the nth qubits.
    static std::map<int,int> max_layer_num_def;

    /// The maximal allowed error of the optimization problem
    double optimization_tolerance;

protected:

    ///  A map of <int n: int num> indicating that how many layers should be used in the subdecomposition process for the subdecomposing of the nth qubits.
    std::map<int,int> max_layer_num;

    /// A map of <int n: int num> indicating the number of iteration in each step of the decomposition.
    std::map<int,int> iteration_loops;

    /// The unitary to be decomposed
    Matrix Umtx;

    /// The optimized parameters for the operations
    double* optimized_parameters;

    /// logical value describing whether the decomposition was finalized or not (i.e. whether the decomposed qubits were rotated into the state |0> or not)
    bool decomposition_finalized;

    /// error of the unitarity of the final decomposition
    double decomposition_error;

    /// number of finalizing (deterministic) opertaions rotating the disentangled qubits into state |0>.
    int finalizing_operations_num;

    /// the number of the finalizing (deterministic) parameters of operations rotating the disentangled qubits into state |0>.
    int finalizing_parameter_num;

    /// The current minimum of the optimization problem
    double current_minimum;

    /// The global target minimum of the optimization problem
    double global_target_minimum;

    /// logical value describing whether the optimization problem was solved or not
    bool optimization_problem_solved;

    /// Maximal number of iterations allowed in the optimization process
    int max_iterations;

    /// type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
    guess_type initial_guess;

    /// Number of outer OpenMP threads. (During the calculations OpenMP multithreading is turned off.)
    int num_threads;

public:

/** Contructor of the class
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary to be decomposed.
@param initial_guess_in Type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
@return An instance of the class
*/
Decomposition_Base( Matrix Umtx_in, int qbit_num_in, guess_type initial_guess_in);

/**
@brief Destructor of the class
*/
virtual ~Decomposition_Base();


/**
@brief Call to set the number of operation blocks to be optimized in one shot
@param optimization_block_in The number of operation blocks to be optimized in one shot
*/
void set_optimization_blocks( int optimization_block_in );

/**
@brief Call to set the maximal number of the iterations in the optimization process
@param max_iterations_in maximal number of iteartions in the optimization process
*/
void set_max_iteration( int max_iterations_in);


/**
@brief After the main optimization problem is solved, the indepent qubits can be rotated into state |0> by this def. The constructed operations are added to the array of operations needed to the decomposition of the input unitary.
*/
void finalize_decomposition();


/**
@brief Call to print the operations decomposing the initial unitary. These operations brings the intial matrix into unity.
@param start_index The index of the first operation
*/
void list_operations( int start_index );

/**
@brief This method determine the operations needed to rotate the indepent qubits into the state |0>
@param mtx The unitary describing indepent qubits. The resulting matrix is returned by this pointer
@param finalizing_operations Pointer pointig to a block of operations containing the final operations.
@param finalizing_parameters Parameters corresponding to the finalizing operations.
@return Returns with the finalized matrix
*/
Matrix get_finalizing_operations( Matrix& mtx, Operation_block* finalizing_operations, double* finalizing_parameters);


/**
@brief This method can be used to solve the main optimization problem which is devidid into sub-layer optimization processes. (The aim of the optimization problem is to disentangle one or more qubits) The optimalized parameters are stored in attribute optimized_parameters.
@param solution_guess An array of the guessed parameters
@param solution_guess_num The number of guessed parameters. (not necessarily equal to the number of free parameters)
*/
void solve_optimization_problem( double* solution_guess, int solution_guess_num );

/**
@brief Abstarct function to be used to solve a single sub-layer optimization problem. The optimalized parameters are stored in attribute optimized_parameters.
@param 'num_of_parameters' The number of free parameters to be optimized
@param solution_guess_gsl A GNU Scientific Libarary vector containing the free parameters to be optimized.
*/
virtual void solve_layer_optimization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl);



/**
@brief This is an abstact definition of function giving the cost functions measuring the entaglement of the qubits. When the qubits are indepent, teh cost function should be zero.
@param parameters An array of the free parameters to be optimized. (The number of the free paramaters should be equal to the number of parameters in one sub-layer)
*/
virtual double optimization_problem( const double* parameters );

/** check_optimization_solution
@brief Checks the convergence of the optimization problem.
@return Returns with true if the target global minimum was reached during the optimization process, or false otherwise.
*/
bool check_optimization_solution();


/**
@brief Calculate the list of gate operation matrices such that the i>0-th element in the result list is the product of the operations of all 0<=n<i operations from the input list and the 0th element in the result list is the identity.
@param parameters An array containing the parameters of the U3 operations.
@param operations_it An iterator pointing to the forst operation.
@param num_of_operations The number of operations involved in the calculations
@return Returns with a vector of the product matrices.
*/
std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>> get_operation_products(double* parameters, std::vector<Operation*>::iterator operations_it, int num_of_operations);


/**
@brief Call to retrive a pointer to the unitary to be transformed
@return Return with the unitary Umtx
*/
Matrix get_Umtx();

/**
@brief Call to get the size of the unitary to be transformed
@return Return with the size N of the unitary NxN
*/
int get_Umtx_size();

/**
@brief Call to get the optimized parameters.
@return Return with the pointer pointing to the array storing the optimized parameters
*/
double* get_optimized_parameters();

/**
@brief Call to get the optimized parameters.
@param ret Preallocated array to store the optimized parameters.
*/
void get_optimized_parameters( double* ret );

/**
@brief Calculate the transformed matrix resulting by an array of operations on the matrix Umtx
@param parameters An array containing the parameters of the U3 operations.
@param operations_it An iterator pointing to the first operation to be applied on the initial matrix.
@param num_of_operations The number of operations to be applied on the initial matrix
@return Returns with the transformed matrix.
*/
Matrix get_transformed_matrix( const double* parameters, std::vector<Operation*>::iterator operations_it, int num_of_operations );



/**
@brief Calculate the transformed matrix resulting by an array of operations on a given initial matrix.
@param parameters An array containing the parameters of the U3 operations.
@param operations_it An iterator pointing to the first operation to be applied on the initial matrix.
@param num_of_operations The number of operations to be applied on the initial matrix
@param initial_matrix The initial matrix wich is transformed by the given operations.
@return Returns with the transformed matrix.
*/
Matrix get_transformed_matrix( const double* parameters, std::vector<Operation*>::iterator operations_it, int num_of_operations, Matrix& initial_matrix );


/**
@brief Calculate the decomposed matrix resulted by the effect of the optimized operations on the unitary Umtx
@return Returns with the decomposed matrix.
*/
Matrix get_decomposed_matrix();


/**
@brief Apply an operations on the input matrix
@param operation_mtx The matrix of the operation.
@param input_matrix The input matrix to be transformed.
@return Returns with the transformed matrix
*/
Matrix apply_operation( Matrix& operation_mtx, Matrix& input_matrix );

/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void reorder_qubits( std::vector<int> qbit_list);

/**
@brief Set the maximal number of layers used in the subdecomposition of the n-th qubit.
@param n The number of qubits for which the maximal number of layers should be used in the subdecomposition.
@param max_layer_num_in The maximal number of the operation layers used in the subdecomposition.
@return Returns with 0 if succeded.
*/
int set_max_layer_num( int n, int max_layer_num_in );

/**
@brief Set the maximal number of layers used in the subdecomposition of the n-th qubit.
@param max_layer_num_in An <int,int> map containing the maximal number of the operation layers used in the subdecomposition.
@return Returns with 0 if succeded.
*/
int set_max_layer_num( std::map<int, int> max_layer_num_in );


/**
@brief Set the number of iteration loops during the subdecomposition of the n-th qubit.
@param n The number of qubits for which number of iteration loops should be used in the subdecomposition.,
@param iteration_loops_in The number of iteration loops in each sted of the subdecomposition.
@return Returns with 0 if succeded.
*/
int set_iteration_loops( int n, int iteration_loops_in );

/**
@brief Set the number of iteration loops during the subdecomposition of the qbit-th qubit.
@param iteration_loops_in An <int,int> map containing the number of iteration loops for the individual subdecomposition processes
@return Returns with 0 if succeded.
*/
int set_iteration_loops( std::map<int, int> iteration_loops_in );


/**
@brief Initializes default layer numbers
*/
static void Init_max_layer_num();


/**
@brief Call to prepare the optimized operations to export. The operations are stored in the attribute operations
*/
void prepare_operations_to_export();

/**
@brief Call to prepare the optimized operations to export
@param ops A list of operations
@param parameters The parameters of the operations
@return Returns with a list of CNOT and U3 operations.
*/
std::vector<Operation*> prepare_operations_to_export( std::vector<Operation*> ops, const double* parameters );



/**
@brief Call to prepare the operations of an operation block to export
@param block_op A pointer to a block of operations
@param parameters The parameters of the operations
@return Returns with a list of CNOT and U3 operations.
*/
std::vector<Operation*> prepare_operations_to_export( Operation_block* block_op, const double* parameters );

/**
@brief Call to prepare the optimized operations to export
@param n Integer labeling the n-th oepration  (n>=0).
@param type The type of the operation from enumeration operation_type is returned via this parameter.
@param target_qbit The ID of the target qubit is returned via this input parameter.
@param control_qbit The ID of the control qubit is returned via this input parameter.
@param parameters The parameters of the operations
@return Returns with 0 if the export of the n-th operation was successful. If the n-th operation does not exists, -1 is returned. If the operation is not allowed to be exported, i.e. it is not a CNOT or U3 operation, then -2 is returned.
*/
int get_operation( unsigned int n, operation_type &type, int &target_qbit, int &control_qbit, double* parameters );


/**
@brief Call to set the verbose attribute to true or false.
@param verbose_in Logical variable. Set true for verbose mode, or to false to suppress output messages.
*/
void set_verbose( bool verbose_in );


/**
@brief Call to get the error of the decomposition
@return Returns with the error of the decomposition
*/
double get_decomposition_error( );



};

#endif //DECOMPOSITION_BASE
