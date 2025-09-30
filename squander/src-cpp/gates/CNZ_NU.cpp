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
/*! \file CNZ_NU.cpp
    \brief Class for the representation of N-qubit non-uniform controlled-Z gate operation.
*/


#include "CNZ_NU.h"
#include "common.h"
#include "dot.h"
#include "Random_Unitary.h"
#include "apply_large_kernel_to_input.h"

static double M_PIOver2 = M_PI/2;
/**
@brief Deafult constructor of the class.
@return An instance of the class
*/
CNZ_NU::CNZ_NU() {

    // A string labeling the gate operation
    name = "CNZ_NU";

    // number of qubits spanning the matrix of the operation
    qbit_num = -1;
    // The size N of the NxN matrix associated with the operations.
    matrix_size = -1;
    // The type of the operation (see enumeration gate_type)
    type = CNZ_NU_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // the number of free parameters of the operation
    parameter_num = -1;  // Will be set when qbit_num is known
}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the unitaries
@return An instance of the class
*/
CNZ_NU::CNZ_NU(int qbit_num_in) {

    // A string labeling the gate operation
    name = "CNZ_NU";

    // number of qubits spanning the matrix of the operation
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the operation
    type = CNZ_NU_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // The number of parameters - single parameter selects which diagonal element is -1
    parameter_num = 1;

    // Initialize centers for computing distance-based logits
    centers = std::vector<double>();
    for (int idx = 0; idx < matrix_size; idx++){
        centers.push_back(static_cast<double>(idx));  // Use integer positions
    }

    // Initialize temperature (0.01 gives sharp softmax close to discrete selection)
    temperature = 0.05;
}


/**
@brief Destructor of the class
*/
CNZ_NU::~CNZ_NU() {
}

/**
@brief Set the number of qubits spanning the matrix of the operation
@param qbit_num_in The number of qubits spanning the matrix
*/
void CNZ_NU::set_qbit_num( int qbit_num_in ) {
    // setting the number of qubits
    qbit_num = qbit_num_in;

    // update the size of the matrix
    matrix_size = Power_of_2(qbit_num);

}


/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
CNZ_NU::apply_to_list(Matrix_real& parameters_mtx,  std::vector<Matrix>& input ) {


    for ( std::vector<Matrix>::iterator it=input.begin(); it != input.end(); it++ ) {
        apply_to( parameters_mtx,  *it, 0);
    }

}

/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
CNZ_NU::apply_to_list(Matrix_real& parameters_mtx, std::vector<Matrix>& inputs, int parallel ) {

    int work_batch = 1;
    if ( parallel == 0 ) {
        work_batch = inputs.size();
    }
    else {
        work_batch = 1;
    }


    tbb::parallel_for( tbb::blocked_range<int>(0,inputs.size(),work_batch), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 

            Matrix* input = &inputs[idx];

            apply_to(parameters_mtx,*input, parallel );

        }

    });

}





/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void
CNZ_NU::apply_to(Matrix_real& parameters_mtx, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
        std::string err("CNZ_NU::apply_to: Wrong matrix size in CNZ_NU gate apply.");
        throw err;
    }

    double x = parameters_mtx[0];

    // Build diagonal matrix using softmax-weighted phases
    // Each diagonal element gets phase π * p_k(x) where p_k is softmax probability
    // This smoothly interpolates: when p_k ≈ 1, that position gets -1, others get +1
    Matrix com_matrix(matrix_size, 1);
    memset(com_matrix.get_data(), 0.0, (com_matrix.size()*2)*sizeof(double));

    for (int idx = 0; idx < matrix_size; idx++){
        // Phase is π * softmax_k(x): gives π (→ -1) at preferred position, 0 (→ +1) elsewhere
        double phase = M_PI * softmax_k(x, idx);
        com_matrix[idx].real = std::cos(phase);
        com_matrix[idx].imag = std::sin(phase);
    }

    apply_diagonal_gate_to_matrix_input(com_matrix, input, input.rows);
}

std::vector<Matrix>
CNZ_NU::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
        std::string err("CNZ_NU::apply_derivate_to: Wrong matrix size");
        throw err;
    }

    std::vector<Matrix> ret;
    double x = parameters_mtx[0];

    // Single parameter derivative
    Matrix res_mtx = input.copy();
    Matrix com_matrix(matrix_size, 1);
    memset(com_matrix.get_data(), 0.0, 2*com_matrix.size()*sizeof(double));

    // Derivative of e^(i*π*softmax_k(x)) w.r.t. x
    // d/dx[e^(i*π*p_k(x))] = i*π*p_k'(x)*e^(i*π*p_k(x))
    for (int idx = 0; idx < matrix_size; idx++){
        double prob = softmax_k(x, idx);
        double phase = M_PI * prob;
        double phase_derivative = M_PI * softmax_k_derivative(x, idx);

        // i * phase_derivative * e^(i*phase)
        // = i * phase_derivative * (cos(phase) + i*sin(phase))
        // = -phase_derivative*sin(phase) + i*phase_derivative*cos(phase)
        com_matrix[idx].real = -phase_derivative * std::sin(phase);
        com_matrix[idx].imag = phase_derivative * std::cos(phase);
    }

    apply_diagonal_gate_to_matrix_input(com_matrix, res_mtx, res_mtx.rows);
    ret.push_back(res_mtx);

    return ret;

}

/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
void
CNZ_NU::apply_from_right(Matrix_real& parameters_mtx, Matrix& input ) {

    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "CNZ_NU::apply_from_right: Wrong matrix size" << std::endl;
        print(sstream, 0);
        exit(-1);
    }

    double x = parameters_mtx[0];

    // Build diagonal matrix using softmax-weighted phases
    Matrix com_matrix(matrix_size, 1);
    memset(com_matrix.get_data(), 0.0, (com_matrix.size()*2)*sizeof(double));

    for (int idx = 0; idx < matrix_size; idx++){
        double phase = M_PI * softmax_k(x, idx);
        com_matrix[idx].real = std::cos(phase);
        com_matrix[idx].imag = std::sin(phase);
    }

    apply_diagonal_gate_to_matrix_input(com_matrix, input, input.rows);

}

// Compute softmax probability for position k
// Uses distance from x (wrapped to [0, matrix_size)) to each center
double CNZ_NU::softmax_k(double x, int k) {
    // Wrap x to [0, matrix_size) range
    x = x - std::floor(x / matrix_size) * matrix_size;
    if (x < 0) x += matrix_size;

    // Compute logits based on negative squared distance from x to each center
    // Closer centers get higher logits
    double logit_k = -((x - centers[k]) * (x - centers[k])) / temperature;

    // Compute softmax denominator
    double sum_exp = 0.0;
    for (int j = 0; j < matrix_size; j++) {
        double logit_j = -((x - centers[j]) * (x - centers[j])) / temperature;
        sum_exp += std::exp(logit_j);
    }

    // Safeguard against numerical issues
    if (sum_exp < 1e-100) {
        return 0.0;
    }

    return std::exp(logit_k) / sum_exp;
}

// Derivative of softmax probability w.r.t. x
double CNZ_NU::softmax_k_derivative(double x, int k) {
    // Wrap x to [0, matrix_size) range
    x = x - std::floor(x / matrix_size) * matrix_size;
    if (x < 0) x += matrix_size;

    // Compute softmax probabilities
    std::vector<double> probs(matrix_size);
    double sum_exp = 0.0;
    for (int j = 0; j < matrix_size; j++) {
        double logit_j = -((x - centers[j]) * (x - centers[j])) / temperature;
        probs[j] = std::exp(logit_j);
        sum_exp += probs[j];
    }

    // Normalize probabilities
    for (int j = 0; j < matrix_size; j++) {
        probs[j] /= sum_exp;
    }

    // Derivative of logit_k w.r.t. x: d/dx[-(x-c_k)^2/τ] = -2(x-c_k)/τ
    double dlogit_k_dx = -2.0 * (x - centers[k]) / temperature;

    // Derivative of softmax: p_k * (dlogit_k/dx - Σ_j p_j * dlogit_j/dx)
    double sum_weighted_grad = 0.0;
    for (int j = 0; j < matrix_size; j++) {
        double dlogit_j_dx = -2.0 * (x - centers[j]) / temperature;
        sum_weighted_grad += probs[j] * dlogit_j_dx;
    }

    return probs[k] * (dlogit_k_dx - sum_weighted_grad);
}

// Set temperature parameter
void CNZ_NU::set_temperature(double temp) {
    temperature = temp;
}

/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
gate_type 
CNZ_NU::get_type() {
    return type;
}


/**
@brief Call to get the number of qubits composing the unitary
@return Return with the number of qubits composing the unitary
*/
int 
CNZ_NU::get_qbit_num() {
    return qbit_num;
}


/**
@brief Call to retrieve the operation matrix
@return Returns with a matrix of the operation
*/
Matrix
CNZ_NU::get_matrix() {
    return get_matrix( false );
}

/**
@brief Call to retrieve the operation matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the operation
*/
Matrix
CNZ_NU::get_matrix( int parallel ) {
    Matrix_real parameters(1,1);
    parameters[0] = 0.5;

    Matrix CNZ_matrix = create_identity(matrix_size);
    apply_to( parameters, CNZ_matrix, parallel );

#ifdef DEBUG
    if (CNZ_matrix.isnan()) {
        std::stringstream sstream;
        sstream << "CNZ_NU::get_matrix: CNZ_matrix contains NaN." << std::endl;
        print(sstream, 1);
    }
#endif

    return CNZ_matrix;
}

/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix
@return Returns with a matrix of the operation
*/
Matrix
CNZ_NU::get_matrix( Matrix_real& parameters ) {
    return get_matrix( parameters, false );
}

/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the operation
*/
Matrix
CNZ_NU::get_matrix( Matrix_real& parameters, int parallel ) {
    Matrix CNZ_matrix = create_identity(matrix_size);
    apply_to( parameters, CNZ_matrix, parallel );

#ifdef DEBUG
    if (CNZ_matrix.isnan()) {
        std::stringstream sstream;
        sstream << "CNZ_NU::get_matrix: CNZ_matrix contains NaN." << std::endl;
        print(sstream, 1);
    }
#endif

    return CNZ_matrix;
}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CNZ_NU* CNZ_NU::clone() {

    CNZ_NU* ret = new CNZ_NU( qbit_num );

    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}



