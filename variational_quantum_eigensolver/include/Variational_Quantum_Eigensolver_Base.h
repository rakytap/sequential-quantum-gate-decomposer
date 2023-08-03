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
/*! \file Decomposition_Base.h
    \brief Header file for a class containing basic methods for the decomposition process.
*/


#ifndef VARIATIONAL_QUANTUM_EIGENSOLVER_BASE_H
#define  VARIATIONAL_QUANTUM_EIGENSOLVER_BASE_H

#include "N_Qubit_Decomposition_Base.h"


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
    /// The maximal allowed error of the optimization problem (The error of the decomposition would scale with the square root of this value)
    double optimization_tolerance;
    
    ///the name of the project
    std::string project_name;

    /// config metadata utilized during the optimization
    std::map<std::string, Config_Element> config;
    

    
protected: 
    std::map<int, Gates_block*> gate_structure;    

private:
    ///  A map of <int n: int num> indicating that how many layers should be used in the subdecomposition process for the subdecomposing of the nth qubits.
    std::map<int,int> max_layer_num;

    /// A map of <int n: int num> indicating the number of iteration in each step of the decomposition.
    std::map<int,int> iteration_loops;

    /// The Unitary of the system
    Matrix_sparse Hamiltonian;
    
    Matrix Zero_state;
    
    int random_shift_count_max;
    
    optimization_aglorithms alg;
    
    /// The optimized parameters for the gates
    Matrix_real optimized_parameters_mtx;

	int qbit_num;
    /// The optimized parameters for the gates
    //double* optimized_parameters;

    /// logical value describing whether the decomposition was finalized or not (i.e. whether the decomposed qubits were rotated into the state |0> or not)
    bool decomposition_finalized;

    /// error of the final decomposition
    double decomposition_error;

    unsigned long long iteration_threshold_of_randomization;

    /// The current minimum of the optimization problem
    double current_minimum;


    /// logical value describing whether the optimization problem was solved or not
    bool optimization_problem_solved;

    /// Store the number of OpenMP threads. (During the calculations OpenMP multithreading is turned off.)
    int num_threads;

    /// The convergence threshold in the optimization process
    double convergence_threshold;
    
    double learning_rate;
    
    int max_iters;
    
        /// Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen; 

    /// logical variable indicating whether adaptive learning reate is used in the ADAM algorithm
    bool adaptive_eta;
    
    double prev_cost_fnv_val;
    
    
public:

Variational_Quantum_Eigensolver_Base();

Variational_Quantum_Eigensolver_Base( Matrix_sparse Hamiltonian_in, int qbit_num_in, std::map<std::string, Config_Element>& config_in);

virtual ~Variational_Quantum_Eigensolver_Base();

double Expected_energy(Matrix State);

static double optimization_problem_vqe(Matrix_real parameters, void* void_instance);

virtual double optimization_problem(Matrix_real& parameters) override;

double optimization_problem( double* parameters);

static void optimization_problem_combined_vqe( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad );

virtual void optimization_problem_combined( Matrix_real parameters, double* f0, Matrix_real grad ) override;

static void optimization_problem_grad_vqe( Matrix_real parameters, void* void_instance, Matrix_real& grad );

void initialize_zero_state();

virtual Problem create_BFGS_problem(int num_of_parameters) override;
void Get_initial_circuit();

void set_adaptive_gate_structure( std::string filename );

void set_custom_gate_structure( std::map<int, Gates_block*> gate_structure_in );

};

#endif
