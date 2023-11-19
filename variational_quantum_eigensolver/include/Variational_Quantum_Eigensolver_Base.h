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
/*! \file Variational_Quantum_Eigensolver_Base.h
    \brief Class to solve VQE problems
*/


#ifndef VARIATIONAL_QUANTUM_EIGENSOLVER_BASE_H
#define  VARIATIONAL_QUANTUM_EIGENSOLVER_BASE_H

#include "N_Qubit_Decomposition_Base.h"

/// @brief Type definition of the fifferent types of ansatz
typedef enum ansatz_type {HEA} ansatz_type;
    
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
@brief A base class to solve VQE problems
This class can be used to approximate the ground state of the input Hamiltonian (sparse format) via a quantum circuit
*/
class Variational_Quantum_Eigensolver_Base : public N_Qubit_Decomposition_Base {
public:


private:

    /// The Hamiltonian of the system
    Matrix_sparse Hamiltonian;
    
    /// Quantum state where all of the quabits are in state zero
    Matrix Zero_state;
    
    /// Ansatz type (HEA stands for hardware efficient ansatz)
    ansatz_type ansatz;

public:


/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
Variational_Quantum_Eigensolver_Base();


/**
@brief Constructor of the class.
@param Hamiltonian_in The Hamiltonian describing the physical system
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param config_in A map that can be used to set hyperparameters during the process
@return An instance of the class
*/
Variational_Quantum_Eigensolver_Base( Matrix_sparse Hamiltonian_in, int qbit_num_in, std::map<std::string, Config_Element>& config_in);

/**
@brief Destructor of the class
*/
virtual ~Variational_Quantum_Eigensolver_Base();

/**
@brief Call to evaluate the expectation value of the energy.
@param State The state for which the expectation value is evaluated
@return The calculated expectation value
*/
double Expectation_value_of_energy(Matrix& State);


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
@brief Initialize the state used in the quantun circuit. All qubits are initialized to state 0
*/
void initialize_zero_state();


/**
@brief Call to start solving the VQE problem to get the approximation for the ground state  
*/ 
void Get_ground_state();


/**
@brief Call to set the ansatz type. Currently imp
@param ansatz_in The ansatz type . Possible values: "HEA" (hardware efficient ansatz with U3 and CNOT gates).
*/ 
void set_ansatz(ansatz_type ansatz_in);


/**
@brief Call to generate the circuit ansatz
@param layers The number of layers. The depth of the generated circuit is 2*layers+1 (U3-CNOT-U3-CNOT...CNOT-U3)
*/
void generate_circuit( int layers );


/**
@brief Call to set custom layers to the gate structure that are intended to be used in the VQE process.
@param filename The path to the binary file
*/
void set_adaptive_gate_structure( std::string filename );


};

#endif
