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
/*! \file CCX.cpp
    \brief Class representing a CCX (Toffoli) gate.
*/

#include "CCX.h"
#include "apply_dedicated_gate_kernel_to_input.h"


using namespace std;


/**
@brief Nullary constructor of the class.
*/
CCX::CCX() : Gate(){

    // A string labeling the gate operation
    name = "CCX";

    // A string describing the type of the gate
    type = CCX_OPERATION;

    // Initialize control qubits vector (empty for nullary constructor)
    control_qbits.clear();
}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbits_in Vector of control qubit indices (should contain exactly 2 elements for CCX)
*/
CCX::CCX(int qbit_num_in, int target_qbit_in, const std::vector<int>& control_qbits_in)
    : Gate(qbit_num_in, std::vector<int>{target_qbit_in}, control_qbits_in) {

    // A string labeling the gate operation
    name = "CCX";

    // A string describing the type of the gate
    type = CCX_OPERATION;

    // Validate that we have exactly 2 control qubits
    if (control_qbits_in.size() != 2) {
        std::stringstream sstream;
        sstream << "CCX gate requires exactly 2 control qubits, got " << control_qbits_in.size() << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    // Check that control qubits are unique
    if (control_qbits_in[0] == control_qbits_in[1]) {
        std::stringstream sstream;
        sstream << "The two control qubits cannot be the same" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }
}

/**
@brief Destructor of the class
*/
CCX::~CCX() {
}

/**
@brief Call to retrieve the gate matrix
@return Returns with a matrix of the gate
*/
Matrix
CCX::get_matrix() {
    return get_matrix(0);
}

/**
@brief Call to retrieve the gate matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB (optional)
@return Returns with a matrix of the gate
*/
Matrix
CCX::get_matrix(int parallel) {
    Matrix CCX_matrix = create_identity(matrix_size);
    apply_to(CCX_matrix, parallel);

#ifdef DEBUG
    if (CCX_matrix.isnan()) {
        std::stringstream sstream;
        sstream << "CCX::get_matrix: CCX_matrix contains NaN." << std::endl;
        print(sstream, 1);
    }
#endif

    return CCX_matrix;
}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CCX* CCX::clone() {

    CCX* ret = new CCX( qbit_num, target_qbit, control_qbits );

    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}

/**
@brief Call to apply the gate operation on the input matrix (without parameters)
@param input The input matrix on which the transformation is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
void CCX::apply_to(Matrix& input, int parallel) {

    int matrix_size = input.rows;

    // Apply the dedicated X kernel with control qubits vector
    switch (parallel){
        case 0:
            apply_X_kernel_to_input(input, target_qbits, control_qbits, matrix_size); break;
        case 1:
            apply_X_kernel_to_input_omp(input, target_qbits, control_qbits, matrix_size); break;
        case 2:
            apply_X_kernel_to_input_tbb(input, target_qbits, control_qbits, matrix_size); break;
    }

}

/**
@brief Call to apply the gate operation on the input matrix
@param input The input matrix on which the transformation is applied
@param parameters An array of parameters to calculate the matrix elements
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
void CCX::apply_to(Matrix& input, const Matrix_real& parameters, int parallel) {

    int matrix_size = input.rows;

    // Apply the dedicated X kernel with control qubits vector
    switch (parallel){
        case 0:
            apply_X_kernel_to_input(input, target_qbits, control_qbits, matrix_size); break;
        case 1:
            apply_X_kernel_to_input_omp(input, target_qbits, control_qbits, matrix_size); break;
        case 2:
            apply_X_kernel_to_input_tbb(input, target_qbits, control_qbits, matrix_size); break;
    }
}

/**
@brief Call to get the qubits involved in the gate operation.
@return Return with a list of the involved qubits
*/
std::vector<int> CCX::get_involved_qubits(bool only_target) {
    // Use Gate's implementation which now handles vectors
    return Gate::get_involved_qubits(only_target);
}