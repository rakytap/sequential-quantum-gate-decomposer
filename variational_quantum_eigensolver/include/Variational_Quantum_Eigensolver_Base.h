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
#include "BFGS_Powell.h"

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

void set_adaptive_gate_structure( std::string filename );

void set_custom_gate_structure( std::map<int, Gates_block*> gate_structure_in );

};

#endif
