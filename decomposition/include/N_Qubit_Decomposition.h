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

//
// @brief A base class responsible for constructing matrices of C-NOT gates
// gates acting on the N-qubit space

#pragma once
#include "Decomposition_Base.h"
#include "Sub_Matrix_Decomposition.h"
#include "Two_Qubit_Decomposition.h"


////
// @brief A class containing basic methods for the decomposition process.

class N_Qubit_Decomposition : public Decomposition_Base {


public:

protected:

    // logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
    bool optimize_layer_num;

    // The number of successive identical blocks in one leyer
    std::map<int,int> identical_blocks;


public:

//// 
// @brief Constructor of the class.
// @param Umtx The unitary matrix to be decomposed
// @param max_layer_num_in A map of <int n: int num> indicating that how many layers should be used in the subdecomposition process at the subdecomposing of n-th qubits.
// @param identical_blocks_in A map of <int n: int num> indicating that how many identical succesive blocks should be used in the disentanglement of the nth qubit from the others
// @param optimize_layer_num Optional logical value. If true, then the optimalization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimalization is performed for the maximal number of layers.
// @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: 'zeros' (deafult),'random', 'close_to_zero'
// @return An instance of the class
N_Qubit_Decomposition( MKL_Complex16*, int, std::map<int,int>, std::map<int,int>, bool, string );



/// 
// @brief Destructor of the class
~N_Qubit_Decomposition();


////
// @brief Start the disentanglig process of the least significant two qubit unitary
// @param finalize_decomposition Optional logical parameter. If true (default), the decoupled qubits are rotated into
// state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
void start_decomposition(bool finalize_decomposition);

////
// @brief stores the calculated parameters and operations of the sub-decomposition processes
// @param cSub_decomposition An instance of class Sub_Two_Qubit_Decomposition used to disentangle qubit pairs from the others.
// @param qbits_reordered A permutation of qubits that was applied on the initial unitary in prior of the sub decomposition.
// (This is needed to restore the correct qubit indices.)
void  extract_subdecomposition_results( Sub_Matrix_Decomposition* cSub_decomposition );

////
// @brief Start the decompostion process to disentangle the submatrices
void  decompose_submatrix();

};
