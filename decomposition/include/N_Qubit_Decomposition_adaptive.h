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
/*! \file N_Qubit_Decomposition.h
    \brief Header file for a class to determine the decomposition of an N-qubit unitary into a sequence of CNOT and U3 gates.
*/

#ifndef N_Qubit_Decomposition_adaptive_H
#define N_Qubit_Decomposition_adaptive_H

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


/**
@brief A base class to determine the decomposition of an N-qubit unitary into a sequence of CNOT and U3 gates.
This class contains the non-template implementation of the decomposition class.
*/
class N_Qubit_Decomposition_adaptive : public N_Qubit_Decomposition_Base {


public:

protected:


    /// A gate structure describing custom gate structure to be used in the decomposition. 
    Gates_block* gate_structure;
    ///
    int level_limit;
    ///
    int level_limit_min;
    ///
    std::vector<matrix_base<int>> topology;
    
    

public:

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive();


/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in );


/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::vector<matrix_base<int>> topology_in );



/**
@brief Destructor of the class
*/
virtual ~N_Qubit_Decomposition_adaptive();


/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
@param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
*/
virtual void start_decomposition(bool prepare_export=true);

/**
@brief ??????????????
*/
Gates_block* optimize_imported_gate_structure(Matrix_real& optimized_parameters_mtx_loc);


/**
@brief ??????????????
*/
Gates_block* determine_initial_gate_structure(Matrix_real& optimized_parameters_mtx);



/**
@brief ???????????????
*/
Gates_block* compress_gate_structure( Gates_block* gate_structure );

/**
@brief ???????????????
*/
Gates_block* compress_gate_structure( Gates_block* gate_structure, int layer_idx, Matrix_real& optimized_parameters, double& currnt_minimum_loc );

/**
@brief ???????????????
*/
Gates_block* replace_trivial_CRY_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters );

/**
@brief ???????????????
*/
virtual unsigned int get_panelty( Gates_block* gate_structure, Matrix_real& optimized_parameters );


/**
@brief ???????????????
*/
virtual Gates_block* remove_trivial_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters, double& currnt_minimum_loc );

/**
@brief ???????????????
*/
Matrix_real create_reduced_parameters( Gates_block* gate_structure, Matrix_real& optimized_parameters, int layer_idx );

/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
Gates_block* construct_gate_layer( const int& _target_qbit, const int& _control_qbit);

/**
@brief ??????????????????
*/
void add_finalyzing_layer( Gates_block* gate_structure );



/**
@brief Call to set custom layers to the gate structure that are intended to be used in the decomposition.
@param gate_structure_in
*/
void set_adaptive_gate_structure( Gates_block* gate_structure_in );


/**
@brief Call to set custom layers to the gate structure that are intended to be used in the decomposition.
@param filename
*/
void set_adaptive_gate_structure( std::string filename );
 
 
 /**
 @brief Set unitary matrix 
 @param filename file to read unitary from
 */
 void set_unitary( std::string filename );

/**
@brief Call to append custom layers to the gate structure that are intended to be used in the decomposition.
@param filename
*/
void add_adaptive_gate_structure( std::string filename );

/**
@brief Call to apply the imported gate structure on the unitary. The transformed unitary is to be decomposed in the calculations, and the imported gfate structure is released.
*/
void apply_imported_gate_structure();


/**
@brief Call to add an adaptive layer to the gate structure previously imported gate structure
@param filename
*/
void add_layer_to_imported_gate_structure();


};






#endif
