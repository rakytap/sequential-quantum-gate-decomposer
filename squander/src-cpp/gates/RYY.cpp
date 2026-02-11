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
/*! \file RYY.cpp
    \brief Class representing a RYY gate.
*/

#include "RYY.h"
#include "apply_large_kernel_to_input.h"
#ifdef USE_AVX
    #include "apply_large_kernel_to_input_AVX.h"
#endif
using namespace std;

/**
@brief Nullary constructor of the class.
*/
RYY::RYY() : Gate() {

    // A string labeling the gate operation
    name = "RYY";

    // A string describing the type of the gate
    type = RYY_OPERATION;

    // Initialize target qubits vector (empty for nullary constructor)
    target_qbits.clear();

    parameter_num = 1;
}

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbits_in Vector of target qubit indices (should contain exactly 2 elements for RYY)
*/
RYY::RYY(int qbit_num_in, const std::vector<int>& target_qbits_in)
    : Gate(qbit_num_in, target_qbits_in) {

    // A string labeling the gate operation
    name = "RYY";

    // A string describing the type of the gate
    type = RYY_OPERATION;

    // Validate that we have exactly 2 target qubits
    if (target_qbits_in.size() != 2) {
        std::stringstream sstream;
        sstream << "RYY gate requires exactly 2 target qubits, got " << target_qbits_in.size() << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    // Check that target qubits are unique
    if (target_qbits_in[0] == target_qbits_in[1]) {
        std::stringstream sstream;
        sstream << "The two target qubits cannot be the same" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    parameter_num = 1;
}

/**
@brief Destructor of the class
*/
RYY::~RYY() {

}

/**
@brief Call to retrieve the gate matrix
@return Returns with a matrix of the gate
*/
Matrix
RYY::get_matrix(Matrix_real& parameters) {
    return get_matrix(parameters,false);
}

/**
@brief Call to retrieve the gate matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the gate
*/
Matrix
RYY::get_matrix(Matrix_real& parameters, int parallel) {
    Matrix RYY_matrix = create_identity(matrix_size);
    apply_to(parameters, RYY_matrix, parallel);

#ifdef DEBUG
    if (RYY_matrix.isnan()) {
        std::stringstream sstream;
        sstream << "RYY::get_matrix: RYY_matrix contains NaN." << std::endl;
        print(sstream, 1);
    }
#endif

    return RYY_matrix;
}


/**
@brief Call to apply the gate operation on the input matrix
@param input The input matrix on which the transformation is applied
@param parameters An array of parameters to calculate the matrix elements
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
void RYY::apply_to(Matrix_real& parameters, Matrix& input, int parallel) {

    double ThetaOver2;
    ThetaOver2 = parameters[0];
    Matrix U_2qbit(4,4);
    memset(U_2qbit.get_data(),0,(U_2qbit.size()*2)*sizeof(double));
    U_2qbit[0].real = std::cos(ThetaOver2);
    U_2qbit[3].imag = 1.*std::sin(ThetaOver2);
    U_2qbit[1*4+1].real = std::cos(ThetaOver2);
    U_2qbit[1*4+2].imag = -1.*std::sin(ThetaOver2);
    U_2qbit[2*4+2].real = std::cos(ThetaOver2);
    U_2qbit[2*4+1].imag = -1.*std::sin(ThetaOver2);
    U_2qbit[3*4+3].real = std::cos(ThetaOver2);
    U_2qbit[3*4].imag = 1.*std::sin(ThetaOver2);
    int inner_qbit = target_qbits[0]<target_qbits[1] ? target_qbits[0]:target_qbits[1];
    int outer_qbit = target_qbits[0]>target_qbits[1] ? target_qbits[0]:target_qbits[1];
    switch (parallel){
        case 0:{
            apply_large_kernel_to_input(U_2qbit,input,target_qbits,input.rows);
            break;
        }
        case 1:{
            #ifdef USE_AVX
                apply_large_kernel_to_input_AVX_OpenMP(U_2qbit,input,target_qbits,input.cols);
            #else
                apply_large_kernel_to_input(U_2qbit,input,target_qbits,input.rows);
            #endif
            break;
        }
        case 2:{
            #ifdef USE_AVX
                apply_large_kernel_to_input_AVX_TBB(U_2qbit,input,target_qbits,input.cols);
            #else
                apply_large_kernel_to_input(U_2qbit,input,target_qbits,input.rows);
            #endif
            break;        }
    }


}
/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/

std::vector<Matrix> RYY::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ){
    if (input.rows != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in RYY apply_derivate_to" << std::endl;
        print(sstream, 0);
        exit(-1);
    }


    std::vector<Matrix> ret;

    Matrix_real parameters_tmp(1,1);

    parameters_tmp[0] = parameters_mtx[0] + M_PI/2;
    Matrix res_mtx = input.copy();
    apply_to(parameters_tmp, res_mtx,  parallel );
    ret.push_back(res_mtx);



    return ret;

}
/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
RYY* RYY::clone() {

    RYY* ret = new RYY(qbit_num, target_qbits);

    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);

    return ret;
}

/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void RYY::reorder_qubits(std::vector<int> qbit_list) {
    // Use Gate's implementation which now handles vectors
    Gate::reorder_qubits(qbit_list);
}

/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
void RYY::set_qbit_num(int qbit_num_in) {
    // setting the number of qubits
    Gate::set_qbit_num(qbit_num_in);
}

/**
@brief Get list of involved qubits
@param only_target If true, return only target qubits, otherwise include control qubits too
@return Vector of qubit indices
*/
std::vector<int> RYY::get_involved_qubits(bool only_target) {
    // Use Gate's implementation which now handles vectors
    return Gate::get_involved_qubits(only_target);
}
