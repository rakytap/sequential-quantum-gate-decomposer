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

#ifndef N_Qubit_Decomposition_adaptive_general_H
#define N_Qubit_Decomposition_adaptive_general_H

#include "N_Qubit_Decomposition_adaptive.h"


/**
@brief A base class to determine the decomposition of an N-qubit unitary into a sequence of CNOT and U3 gates.
This class contains the non-template implementation of the decomposition class.
*/
class N_Qubit_Decomposition_adaptive_general : public N_Qubit_Decomposition_adaptive {


    

public:

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive_general();



/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive_general( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in );



/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive_general( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::vector<matrix_base<int>> topology_in );

/**
@brief Destructor of the class
*/
virtual ~N_Qubit_Decomposition_adaptive_general();


/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
@param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
*/
virtual void start_decomposition(bool prepare_export=true);



/**
@brief ???????????????
*/
int get_panelty( Gates_block* gate_structure, Matrix_real& optimized_parameters );


/**
@brief ???????????????
*/
Gates_block* remove_trivial_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters, double& currnt_minimum_loc );

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



};



#endif
