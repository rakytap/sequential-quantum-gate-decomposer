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
/*! \file Gates_block.cpp
    \brief Class responsible for grouping two-qubit (CNOT,CZ,CH) and one-qubit gates into layers
*/


#ifndef VARIATIONAL_QUANTUM_EIGENSOLVER_BASE_H
#define  VARIATIONAL_QUANTUM_EIGENSOLVER_BASE_H

#include "N_Qubit_Decomposition_Base.h"
#include "BFGS_Powell.h"

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

class Variational_Quantum_Eigensolver_Base : public N_Qubit_Decomposition_Base {
public:

protected: 
    std::map<int, Gates_block*> gate_structure;    


private:

    /// The Unitary of the system
    Matrix_sparse Hamiltonian;
    
    Matrix Zero_state;
    
    ansatz_type ansatz;

public:

Variational_Quantum_Eigensolver_Base();

Variational_Quantum_Eigensolver_Base( Matrix_sparse Hamiltonian_in, int qbit_num_in, std::map<std::string, Config_Element>& config_in);

virtual ~Variational_Quantum_Eigensolver_Base();

double Expected_energy(Matrix State);

static double optimization_problem_vqe(Matrix_real parameters, void* void_instance);

virtual double optimization_problem(Matrix_real& parameters) override;

double optimization_problem( double* parameters);

virtual BFGS_Powell create_bfgs_problem() override;

static void optimization_problem_combined_vqe( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad );

virtual void optimization_problem_combined( Matrix_real parameters, double* f0, Matrix_real grad ) override;

static void optimization_problem_grad_vqe( Matrix_real parameters, void* void_instance, Matrix_real& grad );

void initialize_zero_state();

void Get_ground_state();

void set_ansatz(ansatz_type ansatz_in);

void generate_initial_circuit( int layers, int blocks, int rot_layers );

void set_adaptive_gate_structure( std::string filename );

void set_custom_gate_structure( std::map<int, Gates_block*> gate_structure_in );

};

#endif
