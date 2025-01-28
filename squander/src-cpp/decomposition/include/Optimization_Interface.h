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
/*! \file N_Qubit_Decomposition.h
    \brief Header file for a class implementing optimization engines
*/

#ifndef Optimization_Interface_H
#define Optimization_Interface_H

#include "Decomposition_Base.h"
#include "BFGS_Powell.h"
#include "Bayes_Opt.h"
#include "Powells_method.h"

#ifdef __DFE__
#include "common_DFE.h"
#endif

/// @brief Type definition of the fifferent types of the cost function
typedef enum cost_function_type {FROBENIUS_NORM, FROBENIUS_NORM_CORRECTION1, FROBENIUS_NORM_CORRECTION2,
    HILBERT_SCHMIDT_TEST, HILBERT_SCHMIDT_TEST_CORRECTION1, HILBERT_SCHMIDT_TEST_CORRECTION2,
    SUM_OF_SQUARES, VQE} cost_function_type;



/// implemented optimization strategies
enum optimization_aglorithms{ ADAM, BFGS, BFGS2, ADAM_BATCHED, AGENTS, COSINE, AGENTS_COMBINED, GRAD_DESCEND, BAYES_OPT, BAYES_AGENTS, GRAD_DESCEND_PARAMETER_SHIFT_RULE};


#ifdef __cplusplus
extern "C" 
{
#endif

/// Definition of the zggev function from Lapacke to calculate the eigenvalues of a complex matrix
int LAPACKE_zggev 	( 	int  	matrix_layout,
		char  	jobvl,
		char  	jobvr,
		int  	n,
		QGD_Complex16 *  	a,
		int  	lda,
		QGD_Complex16 *  	b,
		int  	ldb,
		QGD_Complex16 *  	alpha,
		QGD_Complex16 *  	beta,
		QGD_Complex16 *  	vl,
		int  	ldvl,
		QGD_Complex16 *  	vr,
		int  	ldvr 
	); 	

#ifdef __cplusplus
}
#endif





/**
@brief A base class to determine the decomposition of an N-qubit unitary into a sequence of CNOT and U3 gates.
This class contains the non-template implementation of the decomposition class.
*/
class Optimization_Interface : public Decomposition_Base {


public:

    /// the maximal number of iterations for which an optimization engine tries to solve the optimization problem
    int max_inner_iterations;
    /// the maximal number of parameter randomization tries to escape a local minimum.
    int random_shift_count_max;
    /// unique id indentifying the instance of the class
    int id;
protected:


    /// logical value. Set true to optimize the minimum number of gate layers required in the decomposition, or false when the predefined maximal number of layer gates is used (ideal for general unitaries).
    bool optimize_layer_num;

    /// A map of <int n: int num> indicating that how many identical successive blocks should be used in the disentanglement of the nth qubit from the others
    std::map<int,int> identical_blocks;
    /// The optimization algorithm to be used in the optimization
    optimization_aglorithms alg;
    /// The chosen variant of the cost function
    cost_function_type cost_fnc;
    /// the previous value of the cost funtion to be used to evaluate bitflip errors in the cost funtion (see Eq. (21) in arXiv:2210.09191)
    double prev_cost_fnv_val;
    /// prefactor of the single-bitflip errors in the cost function. (see Eq. (21) in arXiv:2210.09191)
    double correction1_scale;
    /// prefactor of the double-bitflip errors in the cost function. (see Eq. (21) in arXiv:2210.09191)
    double correction2_scale;    
    

    /// number of iterations
    int number_of_iters;

    /// logical variable indicating whether adaptive learning reate is used in the ADAM algorithm
    bool adaptive_eta;
    /// parameter to contron the radius of parameter randomization around the curren tminimum
    double radius;
    /// randomization rate
    double randomization_rate;
    /// number of utilized accelerators
    int accelerator_num;

    /// The offset in the first columns from which the "trace" is calculated. In this case Tr(A) = sum_(i-offset=j) A_{ij}
    int trace_offset;


    /// Time spent on circuit simulation/cost function evaluation
    double circuit_simulation_time;
    /// time spent on optimization
    double CPU_time;    




public:

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
Optimization_Interface();



/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param config_in A map that can be used to set hyperparameters during the process
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
Optimization_Interface( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, std::map<std::string, Config_Element>& config, guess_type initial_guess_in, int accelerator_num_in=0 );



/**
@brief Destructor of the class
*/
virtual ~Optimization_Interface();

/**
@brief Calculate the error of the decomposition according to the spectral norm of \f$ U-U_{approx} \f$, where \f$ U_{approx} \f$ is the unitary produced by the decomposing quantum cirquit. The calculated error is stored in the attribute decomposition_error.
@param decomposed_matrix The decomposed matrix, i.e. the result of the decomposing gate structure applied on the initial unitary.
@return Returns with the calculated spectral norm.
*/
void calc_decomposition_error(Matrix& decomposed_matrix );


/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
virtual void add_finalyzing_layer();


/**
@brief final optimization procedure improving the accuracy of the decompositin when all the qubits were already disentangled.
*/
void final_optimization();



/**
@brief Call to solve layer by layer the optimization problem via calling one of the implemented algorithms. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess Array containing the solution guess.
*/
void solve_layer_optimization_problem( int num_of_parameters, Matrix_real solution_guess);


/**
@brief Call to solve layer by layer the optimization problem via the COSINE algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess A matrix containing the solution guess.
*/
void solve_layer_optimization_problem_COSINE( int num_of_parameters, Matrix_real& solution_guess);


/**
@brief Call to solve layer by layer the optimization problem via the GRAD_DESCEND_PARAMETER_SHIFT_RULE algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess A matrix containing the solution guess.
*/
void solve_layer_optimization_problem_GRAD_DESCEND_PARAMETER_SHIFT_RULE( int num_of_parameters, Matrix_real& solution_guess);


/**
@brief Call to solve layer by layer the optimization problem via the GRAD_DESCEND (line search in the direction determined by the gradient) algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess A matrix containing the solution guess.
*/
void solve_layer_optimization_problem_GRAD_DESCEND( int num_of_parameters, Matrix_real& solution_guess);


/**
@brief Call to solve layer by layer the optimization problem via the AGENT algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess A matrix containing the solution guess.
*/
void solve_layer_optimization_problem_AGENTS( int num_of_parameters, Matrix_real& solution_guess);



/**
@brief Call to print out into a file the current cost function and the second Rényi entropy on the subsystem made of qubits 0 and 1. Parameters stored in the class are used to calculate the Renyi entropy
@param current_minimum The current minimum (to avoid calculating it again
*/
void export_current_cost_fnc(double current_minimum );


/**
@brief Call to print out into a file the current cost function and the second Rényi entropy on the subsystem made of qubits 0 and 1.
@param current_minimum The current minimum (to avoid calculating it again
@param parameters Parameters to be used in the calculations (For Rényi entropy)
*/
void export_current_cost_fnc(double current_minimum, Matrix_real& parameters );


/**
@brief Call to print out into a file the current cost function and the second Rényi entropy on the subsystem made of qubits 0 and 1.
@param current_minimum The current minimum (to avoid calculating it again
@param parameters Parameters to be used in the calculations (For Rényi entropy)
@param instance A pointer pointing ti the current class instance.
*/
static void export_current_cost_fnc(double current_minimum, Matrix_real& parameters, void* void_instance);


/**
@brief Call to solve layer by layer the optimization problem via the AGENT COMBINED algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess A matrix containing the solution guess.
*/
void solve_layer_optimization_problem_AGENTS_COMBINED( int num_of_parameters, Matrix_real& solution_guess);


/**
@brief Call to solve layer by layer the optimization problem via BBFG algorithm. (optimal for smaller problems) The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess A matrix containing the solution guess.
*/
void solve_layer_optimization_problem_BFGS( int num_of_parameters, Matrix_real& solution_guess);


/**
@brief Call to solve layer by layer the optimization problem via BBFG algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess A matrix containing the solution guess.
*/
void solve_layer_optimization_problem_BFGS2( int num_of_parameters, Matrix_real solution_guess);

/**
@brief Call to solve layer by layer the optimization problem via Bayes algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess A matrix containing the solution guess.
*/
void solve_layer_optimization_problem_BAYES_OPT( int num_of_parameters, Matrix_real& solution_guess);


/**
@brief Call to solve layer by layer the optimization problem via Bayes & Agents algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess A matrix containing the solution guess.
*/
void solve_layer_optimization_problem_BAYES_AGENTS( int num_of_parameters, Matrix_real& solution_guess);



/**
@brief Call to solve layer by layer the optimization problem via batched ADAM algorithm. (optimal for larger problems) The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess A matrix containing the solution guess.
*/
void solve_layer_optimization_problem_ADAM_BATCHED( int num_of_parameters, Matrix_real& solution_guess_);

/**
@brief Call to solve layer by layer the optimization problem via ADAM algorithm. (optimal for larger problems) The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess A matrix containing the solution guess.
*/
void solve_layer_optimization_problem_ADAM( int num_of_parameters, Matrix_real& solution_guess);

/**
@brief Call to randomize the parameter.
@param input The parameters are randomized around the values stores in this array
@param output The randomized parameters are stored within this array
@param f0 weight in the randomiztaion (output = input + rand()*sqrt(f0) ).
*/
void randomize_parameters( Matrix_real& input, Matrix_real& output, const double& f0 );

/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double optimization_problem( double* parameters);


/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
virtual double optimization_problem( Matrix_real& parameters);


/**
@brief The optimization problem of the final optimization with batched input (implemented only for the Frobenius norm cost function)
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
Matrix_real optimization_problem_batched( std::vector<Matrix_real>& parameters_vec);



/**
// @brief The optimization problem of the final optimization
@param parameters Array containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param ret_temp A matrix to store trace in for gradient for HS test 
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double optimization_problem( Matrix_real parameters, void* void_instance, Matrix ret_temp);


/**
@brief The optimization problem of the final optimization
@param parameters Array containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
static double optimization_problem( Matrix_real parameters, void* void_instance);


/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
virtual double optimization_problem_non_static( Matrix_real parameters, void* void_instance);


/**
@brief Calculate the derivative of the cost function with respect to the free parameters.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad Array containing the calculated gradient components.
*/
static void optimization_problem_grad( Matrix_real parameters, void* void_instance, Matrix_real& grad );




/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
static void optimization_problem_combined( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad );


/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
virtual void optimization_problem_combined_non_static( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad );


/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters The parameters for which the cost fuction shoule be calculated
@param f0 The value of the cost function at x0.
@param grad An array storing the calculated gradient components
@param onlyCPU Set true to use only CPU in the calculations (has effect if compiled to use accelerator devices)
*/
virtual void optimization_problem_combined( Matrix_real parameters, double* f0, Matrix_real grad );


/**
@brief Call to calculate both the effect of the circuit on th eunitary and it's gradient componets.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param Umtx The unitary on which the circuit is applied in place.
@param Umtx_deriv Array containing the calculated gradient components.
*/
static void optimization_problem_combined_unitary( Matrix_real parameters, void* void_instance, Matrix& Umtx, std::vector<Matrix>& Umtx_deriv );

/**
@brief Call to calculate both the effect of the circuit on th eunitary and it's gradient componets.
@param parameters Array containing the free parameters to be optimized.
@param Umtx The unitary on which the circuit is applied in place.
@param Umtx_deriv Array containing the calculated gradient components.
*/
void optimization_problem_combined_unitary( Matrix_real parameters, Matrix& Umtx, std::vector<Matrix>& Umtx_deriv );


/**
// @brief The optimization problem of the final optimization
@param parameters A GNU Scientific Library containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double optimization_problem_panelty( double* parameters, Gates_block* gates_block );


/**
@brief Call to get the variant of the cost function used in the calculations
*/
cost_function_type get_cost_function_variant();


/**
@brief Call to retrieve the previous value of the cost funtion to be used to evaluate bitflip errors in the cost funtion (see Eq. (21) in arXiv:2210.09191)
*/
double get_previous_cost_function_value();


/**
@brief Call to set the variant of the cost function used in the calculations
@param variant The variant of the cost function from the enumaration cost_function_type
*/
void set_cost_function_variant( cost_function_type variant  );


/**
@brief Call to get the prefactor of the single-bitflip errors in the cost function. (see Eq. (21) in arXiv:2210.09191)
@return Returns with the prefactor of the single-bitflip errors in the cost function. 
*/
double get_correction1_scale();





/**
@brief Call to get the prefactor of the two-bitflip errors in the cost function. (see Eq. (21) in arXiv:2210.09191)
@return Returns with the prefactor of the two-bitflip errors in the cost function. 
*/
double get_correction2_scale();






/**
@brief Call to set the maximal number of iterations for which an optimization engine tries to solve the optimization problem
@param max_inner_iterations_in The number of iterations for which an optimization engine tries to solve the optimization problem 
*/
void set_max_inner_iterations( int max_inner_iterations_in  );


/**
@brief Call to set the maximal number of parameter randomization tries to escape a local minimum.
@param random_shift_count_max_in The number of maximal number of parameter randomization tries to escape a local minimum.
*/
void set_random_shift_count_max( int random_shift_count_max_in  );


/**
@brief Call to set the optimizer engine to be used in solving the optimization problem.
@param alg_in The chosen algorithm
*/
void set_optimizer( optimization_aglorithms alg_in );


/**
@brief Get the trace ffset used in the evaluation of the cost function
*/
int get_trace_offset();

/**
@brief Set the trace offset used in the evaluation of the cost function
*/
void set_trace_offset(int trace_offset_in);


/**
@brief Get the number of processed iterations during the optimization process
*/
int get_num_iters();

/**
@brief Call to set custom layers to the gate structure that are intended to be used in the subdecomposition.
@param gate_structure An <int, Gates_block*> map containing the gate structure used in the individual subdecomposition (default is used, if a gate structure for specific subdecomposition is missing).
*/
void set_custom_gate_structure( Gates_block* gate_structure_in );


#ifdef __DFE__

/**
@brief Call to upload the unitary up to the DFE
*/
void upload_Umtx_to_DFE();


/**
@brief Get the number of accelerators to be reserved on DFEs on users demand.
*/
int get_accelerator_num();

#ifdef __DFE__
virtual void apply_to( Matrix_real& parameters_mtx, Matrix& input, int parallel=0 )
{
    if (ctz(input.rows) == 17) {
        std::vector<int> target_qbit;
        std::vector<int> control_qbit;
        std::vector<Matrix> u3_qbit;
        get_matrices_target_control(u3_qbit, target_qbit, control_qbit, parameters_mtx);
        const int device_num = 0;
        apply_to_groq_sv(device_num, u3_qbit, input, target_qbit, control_qbit);
        return;
    }
    Decomposition_Base::apply_to(parameters_mtx, input, parallel);
}
#endif

#endif

};


#endif
