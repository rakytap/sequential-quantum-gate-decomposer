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
/*! \file Decomposition_Base.h
    \brief Header file for a class containing basic methods for the decomposition process.
*/

#ifndef DECOMPOSITION_BASE_H
#define DECOMPOSITION_BASE_H


#include "Gates_block.h"
#include "CZ.h"
#include "CH.h"
#include "CNOT.h"
#include "U3.h"
#include "RX.h"
#include "H.h"
#include "X.h"
#include "Y.h"
#include "Z.h"
#include "SX.h"
#include "RY.h"
#include "CRY.h"
#include "RZ.h"
#include "RZ_P.h"
#include "SYC.h"
#include "UN.h"
#include "ON.h"
#include "Adaptive.h"
#include "CZ_NU.h"
#include "Composite.h"
#include <map>
#include <cstdlib>
#include <time.h>
#include <ctime>
#include <cfloat>
#include <limits>
#include <tbb/cache_aligned_allocator.h>

#include "config_element.h"

#include <random>

/// @brief Type definition of the types of the initial guess
typedef enum guess_type {ZEROS, RANDOM, CLOSE_TO_ZERO} guess_type;

#ifndef OPENBLAS_CONST
# define OPENBLAS_CONST const
#endif
#if defined(OS_WINDOWS) && defined(__64BIT__)
typedef long long BLASLONG;
typedef unsigned long long BLASULONG;
#else
typedef long BLASLONG;
typedef unsigned long BLASULONG;
#endif
typedef BLASLONG blasint;
#if defined(OS_WINDOWS) && defined(__64BIT__)
#define blasabs(x) llabs(x)
#else
#define blasabs(x) labs(x)
#endif

extern "C" double cblas_dznrm2(OPENBLAS_CONST blasint N, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX);

/**
@brief A class containing basic methods for the decomposition process.
*/
class Decomposition_Base : public Gates_block {


public:
  
    /// number of gate blocks used in one shot of the optimization process
    int optimization_block;

    /// A map of <int n: int num> indicating that how many layers should be used in the subdecomposition process by default for the subdecomposing of the nth qubits.
    static std::map<int,int> max_layer_num_def;

    /// The maximal allowed error of the optimization problem (The error of the decomposition would scale with the square root of this value)
    double optimization_tolerance;
    
    ///The global phase
    QGD_Complex16 global_phase_factor;
    
    ///the name of the project
    std::string project_name;

    /// config metadata utilized during the optimization
    std::map<std::string, Config_Element> config;


protected:

    ///  A map of <int n: int num> indicating that how many layers should be used in the subdecomposition process for the subdecomposing of the nth qubits.
    std::map<int,int> max_layer_num;

    /// A map of <int n: int num> indicating the number of iteration in each step of the decomposition.
    std::map<int,int> iteration_loops;

    /// The unitary to be decomposed
    Matrix Umtx;

    /// The optimized parameters for the gates
    Matrix_real optimized_parameters_mtx;

    /// The optimized parameters for the gates
    //double* optimized_parameters;

    /// logical value describing whether the decomposition was finalized or not (i.e. whether the decomposed qubits were rotated into the state |0> or not)
    bool decomposition_finalized;

    /// error of the final decomposition
    double decomposition_error;

    /// number of finalizing (deterministic) opertaions rotating the disentangled qubits into state |0>.
    int finalizing_gates_num;

    /// the number of the finalizing (deterministic) parameters of gates rotating the disentangled qubits into state |0>.
    int finalizing_parameter_num;

    /// The current minimum of the optimization problem
    double current_minimum;

    /// The global target minimum of the optimization problem
    double global_target_minimum;

    /// logical value describing whether the optimization problem was solved or not
    bool optimization_problem_solved;

    /// Maximal number of iterations allowed in the optimization process
    int max_outer_iterations;

    /// type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
    guess_type initial_guess;

    /// Store the number of OpenMP threads. (During the calculations OpenMP multithreading is turned off.)
    int num_threads;

    /// The convergence threshold in the optimization process
    double convergence_threshold;
    

    /// Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen; 



public:

/** Nullary constructor of the class
@return An instance of the class
*/
Decomposition_Base();

/** Contructor of the class
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary to be decomposed.
@param initial_guess_in Type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
@return An instance of the class
*/
Decomposition_Base( Matrix Umtx_in, int qbit_num_in, std::map<std::string, Config_Element>& config_in, guess_type initial_guess_in);

/**
@brief Destructor of the class
*/
virtual ~Decomposition_Base();


/**
@brief Call to set the number of gate blocks to be optimized in one shot
@param optimization_block_in The number of gate blocks to be optimized in one shot
*/
void set_optimization_blocks( int optimization_block_in );

/**
@brief Call to set the maximal number of the iterations in the optimization process
@param max_outer_iterations_in maximal number of iteartions in the optimization process
*/
void set_max_iteration( int max_outer_iterations_in);


/**
@brief Call to print the gates decomposing the initial unitary. These gates brings the intial matrix into unity.
@param start_index The index of the first gate
*/
void list_gates( int start_index );


/**
@brief This method can be used to solve the main optimization problem which is devidid into sub-layer optimization processes. (The aim of the optimization problem is to disentangle one or more qubits) The optimalized parameters are stored in attribute optimized_parameters.
@param solution_guess An array of the guessed parameters
@param solution_guess_num The number of guessed parameters. (not necessarily equal to the number of free parameters)
*/
void solve_optimization_problem( double* solution_guess, int solution_guess_num );


/**
@brief Abstarct function to be used to solve a single sub-layer optimization problem. The optimalized parameters are stored in attribute optimized_parameters.
@param 'num_of_parameters' The number of free parameters to be optimized
@param solution_guess_gsl Array containing the free parameters to be optimized.
*/
virtual void solve_layer_optimization_problem( int num_of_parameters, Matrix_real solution_guess_gsl);



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
Matrix_real get_optimized_parameters();

/**
@brief Call to get the optimized parameters.
@param ret Preallocated array to store the optimized parameters.
*/
void get_optimized_parameters( double* ret );


/**
@brief Call to set the optimized parameters for initial optimization.
@param ret Preallocated array to store the optimized parameters.
*/
void set_optimized_parameters( double* parameters, int num_of_parameters );


/**
@brief Calculate the decomposed matrix resulted by the effect of the optimized gates on the unitary Umtx
@return Returns with the decomposed matrix.
*/
Matrix get_decomposed_matrix();


/**
@brief Apply an gates on the input matrix
@param gate_mtx The matrix of the gate.
@param input_matrix The input matrix to be transformed.
@return Returns with the transformed matrix
*/
Matrix apply_gate( Matrix& gate_mtx, Matrix& input_matrix );

/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void reorder_qubits( std::vector<int> qbit_list);

/**
@brief Set the maximal number of layers used in the subdecomposition of the n-th qubit.
@param n The number of qubits for which the maximal number of layers should be used in the subdecomposition.
@param max_layer_num_in The maximal number of the gate layers used in the subdecomposition.
@return Returns with 0 if succeded.
*/
int set_max_layer_num( int n, int max_layer_num_in );

/**
@brief Set the maximal number of layers used in the subdecomposition of the n-th qubit.
@param max_layer_num_in An <int,int> map containing the maximal number of the gate layers used in the subdecomposition.
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
@brief Call to set the tolerance of the optimization processes.
@param tolerance_in The value of the tolerance. 
*/
void set_optimization_tolerance( double tolerance_in );


/**
@brief Call to set the threshold of convergence in the optimization processes.
@param convergence_threshold_in The value of the threshold. 
*/
void set_convergence_threshold( double convergence_threshold_in );


/**
@brief Call to get the error of the decomposition
@return Returns with the error of the decomposition
*/
double get_decomposition_error( );


/**
@brief Call to get the obtained minimum of the cost function
@return Returns with the minimum of the cost function
*/
double get_current_minimum( );

/**
@brief Call to get the current name of the project
@return Returns the name of the project
*/
std::string get_project_name();
/**
@brief Call to set the name of the project
@param project_name_new pointer to the new project name
*/
void set_project_name(std::string& project_name_new);

/**
@brief  Calculate the new global phase of the Unitary matrix after removing a trivial U3 matrix
@param global_phase_factor_new: global phase calculated from the product of two U3 matrices
*/
void calculate_new_global_phase_factor( QGD_Complex16 global_phase_factor_new );

/**
@brief Get the global phase of the Unitary matrix 
@return The current global phase
*/
QGD_Complex16 get_global_phase_factor( );

/**
@brief Call to set global phase 
@param global_phase_factor_new The value of the new phase
*/
void set_global_phase(double new_global_phase);


/**
@brief Call to apply the global phase to a matrix
@return Returns with the minimum of the cost function
*/
void apply_global_phase_factor(QGD_Complex16 global_phase_factor, Matrix& u3_gate);

/**
@brief Call to apply the current global phase to the unitary matrix
@param global_phase_factor The value of the phase
*/
void apply_global_phase_factor();

/**
@brief   exports unitary matrix to binary file
@param  filename file to be exported to
*/
void export_unitary(std::string& filename);
/**
@brief Import a Unitary matrix from a file
@param filename  .binary file to read
*/
Matrix import_unitary_from_binary(std::string& filename);


/**
@brief Get the parallel configuration from the config
@return Returns with the parallel configuration (0 for sequential execution, 1 for parallel execution with OpenMP (NOT IMPLEMENTED YET) and 2 for parallel with TBB (optional))
*/
int get_parallel_configuration();


/**
@brief Set the number of qubits spanning the matrix of the gates stored in the block of gates.
@param qbit_num_in The number of qubits spanning the matrices.
*/
void set_qbit_num( int qbit_num_in );



};
#endif //DECOMPOSITION_BASE
