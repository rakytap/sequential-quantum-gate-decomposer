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
/*! \file Generative_Quantum_Machine_Learning_Base.h
    \brief Class to solve GQML problems
*/


#ifndef GENERATIVE_QUANTUM_MACHINE_LEARNING_BASE_H
#define GENERATIVE_QUANTUM_MACHINE_LEARNING_BASE_H

#include "Optimization_Interface.h"
#include "matrix_real.h"

/// @brief Type definition of the fifferent types of ansatz
typedef enum ansatz_type {HEA, HEA_ZYZ, QCMRF} ansatz_type;
    
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
@brief A base class to solve GQML problems
This class can be used to approximate a given distribution via a quantum circuit
*/
class Generative_Quantum_Machine_Learning_Base : public Optimization_Interface {
public:


private:

    /// The state vector's corresponding indices of the training data
    std::vector<int> sample_indices;

    /// Same as the x_vectors but in binary 
    std::vector<std::vector<int>> sample_bitstrings;

    /// The distribution we are trying to approximate
    Matrix_real P_star;
    
    /// Quantum state used as an initial state in the VQE iterations
    Matrix initial_state;
    
    /// Ansatz type (HEA stands for hardware efficient ansatz)
    ansatz_type ansatz;

    /// The expectation value of the the square of the given ditribution (only needed to calculate once)
    double ev_P_star_P_star;

    /// Parameter of the Gaussian kernel
    Matrix_real sigma;
    
    // Lookup table for the gauss kernels
    std::vector<std::vector<double>> gaussian_lookup_table;

    // Decide to use lookup table
    bool use_lookup;

    std::vector<std::vector<int>> all_bitstrings;

    // Number samples
    int sample_size;

    // The cliques in the graph
    std::vector<std::vector<int>> cliques;

    double (Generative_Quantum_Machine_Learning_Base::*MMD_of_the_distributions)(Matrix&);

    bool use_exact;

public:


/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
Generative_Quantum_Machine_Learning_Base();


/**
@brief Constructor of the class.
@param sample_indices_in The input data indices
@param sample_bitstrings_in The input data bitstrings
@param P_star_in The distribution to approximate
@param sigma_in Parameter of the gaussian kernels
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param use_lookup_table_in Use lookup table for the gaussian kernel
@param cliques_in A list of the cliques int the graph
@param use_exact Use exact calculation of the MMD or approximation with samples
@param config_in A map that can be used to set hyperparameters during the process
@return An instance of the class
*/
Generative_Quantum_Machine_Learning_Base( std::vector<int> sample_indices_in, std::vector<std::vector<int>> sample_bitstrings_in, Matrix_real P_star_in, Matrix_real sigma_in, int qbit_num_in, bool use_lookup_table_in, std::vector<std::vector<int>> cliques_in, bool use_exact, std::map<std::string, Config_Element>& config_in);

/**
@brief Destructor of the class
*/
virtual ~Generative_Quantum_Machine_Learning_Base();


/**
@brief Call to evaluate the value of one gaussian kernel function
@param x The index of the first input data
@param y The index of the second input data
@param sigma The parameters of the kernel
@return The calculated value of the kernel function
*/
double Gaussian_kernel(int x, int y, Matrix_real& sigma);

/**
@brief Call to evaluate the approximated expectation value of the square of the distribution
@return The approximated value of the expectation value of the square of the distribution
*/
double expectation_value_P_star_P_star_approx();

/**
@brief Call to evaluate the expectation value of the square of the distribution
@return The calculated value of the expectation value of the square of the distribution
*/
double expectation_value_P_star_P_star_exact();

/**
@brief Call to calculate and save the values of the gaussian kernel needed for traing
*/
void fill_lookup_table();

/**
@brief Call to evaluate the total variational distance of the given distribution and the one created by our circuit
@param State_right The state on the right for which the expectation value is evaluated. It is a column vector.
@return The calculated total variational distance of the distributions
*/
double TV_of_the_distributions(Matrix& State_right);

/**
@brief Call to evaluate the approximated maximum mean discrepancy of the given distribution and the one created by our circuit
@param State_right The state on the right for which the expectation value is evaluated. It is a column vector.
@return The calculated mmd
*/
double MMD_of_the_distributions_approx(Matrix& State_right);

/**
@brief Call to evaluate the maximum mean discrepancy of the given distribution and the one created by our circuit
@param State_right The state on the right for which the expectation value is evaluated. It is a column vector.
@return The calculated mmd
*/
double MMD_of_the_distributions_exact(Matrix& State_right);


/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
virtual double optimization_problem(Matrix_real& parameters) override;


/**
@brief The optimization problem of the final optimization
@param parameters Array containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
virtual double optimization_problem_non_static( Matrix_real parameters, void* void_instance) override;


/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double optimization_problem( double* parameters);


/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
virtual void optimization_problem_combined_non_static( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad ) override;


/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
virtual void optimization_problem_combined( Matrix_real parameters, double* f0, Matrix_real grad ) override;


/**
@brief Calculate the derivative of the cost function with respect to the free parameters.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad Array containing the calculated gradient components.
*/
static void optimization_problem_grad_vqe( Matrix_real parameters, void* void_instance, Matrix_real& grad );

/**
@brief Call to print out into a file the current cost function and the second Rényi entropy on the subsystem made of qubits 0 and 1, and the TV distance.
@param current_minimum The current minimum (to avoid calculating it again
@param parameters Parameters to be used in the calculations (For Rényi entropy)
*/
void export_current_cost_fnc(double current_minimum, Matrix_real& parameters);

/**
@brief Initialize the state used in the quantun circuit. All qubits are initialized to state 0
*/
void initialize_zero_state();


/**
@brief Call to start solving the GQML problem
*/ 
void start_optimization();


/**
@brief Call to set the ansatz type. Currently imp
@param ansatz_in The ansatz type . Possible values: "HEA" (hardware efficient ansatz with U3 and CNOT gates).
*/ 
void set_ansatz(ansatz_type ansatz_in);


/**
@brief Call to generate the circuit ansatz
@param layers The number of layers. The depth of the generated circuit is 2*layers+1 (U3-CNOT-U3-CNOT...CNOT)
@param inner_blocks The number of U3-CNOT repetition within a single layer
*/
void generate_circuit( int layers, int inner_blocks );


/**
@brief Call to generate the circuit ansatz for the given clique
@param qbits The qbits in the clique.
@param res The qbits for previously generated gates to avoid duplication
@param subset Temporary variable for storing subsets.
*/
void generate_clique_circuit(int i, std::vector<int>& qbits, std::vector<std::vector<int>>& res, std::vector<int>& subset);


/**
@brief Call to generate a MultiRZ gate
@param qbits The qbits the gate operates on. The depth of the generated circuit is 2*number of qbits
*/
void MultyRZ(std::vector<int>& qbits);

/**
@brief Call to set custom layers to the gate structure that are intended to be used in the GQML process.
@param filename The path to the binary file
*/
void set_gate_structure( std::string filename );


/**
@brief Call to set the initial quantum state in the VQE iterations
@param initial_state_in A vector containing the amplitudes of the initial state.
*/
void set_initial_state( Matrix initial_state_in );



};

#endif
