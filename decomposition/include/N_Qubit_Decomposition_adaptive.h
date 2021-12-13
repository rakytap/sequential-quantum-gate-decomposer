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

/*
    /// logical value. Set true to optimize the minimum number of gate layers required in the decomposition, or false when the predefined maximal number of layer gates is used (ideal for general unitaries).
    bool optimize_layer_num;

    /// A map of <int n: int num> indicating that how many identical successive blocks should be used in the disentanglement of the nth qubit from the others
    std::map<int,int> identical_blocks;

    /// A map of <int n: Gates_block* block> describing custom gate structure to be used in the decomposition. Gate block corresponding to n is used in the subdecomposition of the n-th qubit. The Gate block is repeated periodically.
    Gates_block* gate_structure;


    std::vector<Decomposition_Tree_Node*> root_nodes;
*/

    /// A map of <int n: Gates_block* block> describing custom gate structure to be used in the decomposition. Gate block corresponding to n is used in the subdecomposition of the n-th qubit. The Gate block is repeated periodically.
    Gates_block* gate_structure;
    ///
    std::vector<Decomposition_Tree_Node*> root_nodes;
    ///
    int level_limit;
    ///
    int decomposition_iterations;

    

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
N_Qubit_Decomposition_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, guess_type initial_guess_in );



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
@brief ???????????????
*/
Gates_block* compress_gate_structure( Gates_block* gate_structure );

/**
@brief ???????????????
*/
Gates_block* compress_gate_structure( Gates_block* gate_structure, Matrix_real& optimized_parameters, int layer_idx, double& current_minimum );

/**
@brief ???????????????
*/
Gates_block* remove_trivial_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters, double& current_minimum );

/**
@brief ???????????????
*/
Matrix_real create_reduced_parameters( Gates_block* gate_structure, Matrix_real& optimized_parameters, int layer_idx );


/**
@brief ???????????????
*/
void decompose_UN_gates();

/**
@brief ????????????????
*/
void create_layers_from_decomposition_tree( const Decomposition_Tree_Node* minimal_root_node, int max_level, Gates_block* gate_structure );



/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
Gates_block* construct_gate_layer( const int& _target_qbit, const int& _control_qbit);


/**
@brief Call to add static gate components in front of the adaptively optimized gate structure.
*/
void add_static_gate_layers_1( Gates_block* gate_structure );


/**
@brief Call to add static gate components following the the adaptively optimized gate structure.
*/
void add_static_gate_layers_2( Gates_block* gate_structure );


/**
@brief ??????????????????
*/
void add_finalyzing_layer( Gates_block* gate_structure );


/**
@brief ??????????????????
*/
Decomposition_Tree_Node* find_minimal_child_node( std::vector<Decomposition_Tree_Node*> &children );


/**
@brief ??????????????????
*/
void delete_root_node( Decomposition_Tree_Node* root_node );


};



#endif
