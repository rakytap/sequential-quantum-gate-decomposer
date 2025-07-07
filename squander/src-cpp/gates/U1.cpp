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
/*! \file U1.cpp
    \brief Class representing a U1 gate.
*/

#include "U1.h"
#include "tbb/tbb.h"

// pi/2
static double M_PIOver2 = M_PI/2;

/**
@brief Nullary constructor of the class.
*/
U1::U1() {
    name = "U1";
    qbit_num = -1;
    matrix_size = -1;
    type = U1_OPERATION;
    target_qbit = -1;
    control_qbit = -1;
    parameter_num = 1;
}

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
*/
U1::U1(int qbit_num_in, int target_qbit_in) {
    name = "U1";
    qbit_num = qbit_num_in;
    matrix_size = Power_of_2(qbit_num);
    type = U1_OPERATION;

    if (target_qbit_in >= qbit_num) {
        std::stringstream sstream;
        sstream << "The index of the target qubit is larger than the number of qubits" << std::endl;
        print(sstream, 0);             
        throw "The index of the target qubit is larger than the number of qubits";
    }
    
    target_qbit = target_qbit_in;
    control_qbit = -1;
    parameter_num = 1;
    parameters = Matrix_real(1, parameter_num);
}

/**
@brief Destructor of the class
*/
U1::~U1() {
}

/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U1 gate.
@return Returns with a matrix of the gate
*/
Matrix U1::get_matrix( Matrix_real& parameters ) {
    return get_matrix( parameters, false );
}

/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U1 gate.
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the gate
*/
Matrix U1::get_matrix( Matrix_real& parameters, int parallel ) {
    Matrix U1_matrix = create_identity(matrix_size);
    apply_to(parameters, U1_matrix, parallel);

#ifdef DEBUG
    if (U1_matrix.isnan()) {
        std::stringstream sstream;
        sstream << "U1::get_matrix: U1_matrix contains NaN." << std::endl;
        print(sstream, 1);	  
    }
#endif

    return U1_matrix;
}

/**
@brief Call to apply the gate on a list of inputs
@param parameters An array of parameters to calculate the matrix of the U1 gate.
@param inputs The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void U1::apply_to_list( Matrix_real& parameters, std::vector<Matrix>& inputs, int parallel ) {
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

            apply_to( parameters, *input, parallel );

        }

    });
}

/**
@brief Call to apply the gate on the input array/matrix
@param parameters An array of parameters to calculate the matrix of the U1 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void U1::apply_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {
    if (input.rows != matrix_size ) {
        std::string err("U1::apply_to: Wrong input size in U1 gate apply.");
        throw err;    
    }

    double Lambda = parameters_mtx[0];
    
    // get the U1 gate of one qubit
    Matrix u1_1qbit = calc_one_qubit_u3(Lambda);
    
    apply_kernel_to( u1_1qbit, input, false, parallel );
}

/**
@brief Call to apply the gate on the input array/matrix by input*U1
@param parameters An array of parameters to calculate the matrix of the U1 gate.
@param input The input array on which the gate is applied
*/
void U1::apply_from_right( Matrix_real& parameters_mtx, Matrix& input ) {
    if (input.cols != matrix_size ) {
        std::string err("U1::apply_from_right: Wrong matrix size in U1 apply_from_right.");
        throw err;    
    }

    double Lambda = parameters_mtx[0];
    
    // get the U1 gate of one qubit
    Matrix u1_1qbit = calc_one_qubit_u3(Lambda);
    
    apply_kernel_from_right(u1_1qbit, input);
}

/**
@brief Call to evaluate the derivate of the circuit on an input with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
std::vector<Matrix> U1::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {
    if (input.rows != matrix_size ) {
        std::string err("U1::apply_derivate_to: Wrong matrix size in U1 gate apply.");
        throw err;    
    }

    std::vector<Matrix> ret;
    double Lambda = parameters_mtx[0];
    bool deriv = true;

    Matrix u1_1qbit_lambda = calc_one_qubit_u3(Lambda + M_PIOver2);
    memset(u1_1qbit_lambda.get_data(), 0.0, sizeof(QGD_Complex16) );
    memset(u1_1qbit_lambda.get_data()+2, 0.0, sizeof(QGD_Complex16) );

    Matrix res_mtx_lambda = input.copy();
    apply_kernel_to( u1_1qbit_lambda, res_mtx_lambda, deriv, parallel );
    ret.push_back(res_mtx_lambda);

    return ret;
}

/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
void U1::set_qbit_num(int qbit_num_in) {
    Gate::set_qbit_num(qbit_num_in);
}

/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void U1::reorder_qubits( std::vector<int> qbit_list) {
    Gate::reorder_qubits(qbit_list);
}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
U1* U1::clone() {
    U1* ret = new U1(qbit_num, target_qbit);

    if ( parameters.size() > 0 ) {
        ret->set_optimized_parameters(parameters[0]);
    }
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;
}

/**
@brief Call to set the final optimized parameters of the gate.
@param Lambda Real parameter standing for the parameter lambda.
*/
void U1::set_optimized_parameters(double Lambda ) {
    parameters = Matrix_real(1, 1);
    parameters[0] = Lambda;
}

/**
@brief Call to get the final optimized parameters of the gate.
@return Returns with the parameters of the U1 gate.
*/
Matrix_real U1::get_optimized_parameters() {
    return parameters.copy();
}

/**
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is embedded.
@param parameters The parameter array corresponding to the circuit in which the gate is embedded
@return Returns with the array of the extracted parameters.
*/
Matrix_real U1::extract_parameters( Matrix_real& parameters ) {
    if ( get_parameter_start_idx() + get_parameter_num() > parameters.size()  ) {
        std::string err("U1::extract_parameters: Can't extract parameters, since the input array has not enough elements.");
        throw err;
    }

    Matrix_real ret(1, get_parameter_num());
    memcpy(ret.get_data(), parameters.get_data()+get_parameter_start_idx(), get_parameter_num()*sizeof(double));
    return ret;
}

/**
@brief Calculate the matrix of a U1 gate corresponding to the given parameters acting on a single qbit space. (Virtual method override)
@param Theta Real parameter standing for the parameter theta (ignored for U1).
@param Phi Real parameter standing for the parameter phi (ignored for U1).
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
Matrix U1::calc_one_qubit_u3(double Theta, double Phi, double Lambda) {
    // U1 only uses Lambda parameter, ignore Theta and Phi
    return calc_one_qubit_u3(Lambda);
}

/**
@brief Calculate the matrix of a U1 gate corresponding to the given parameters acting on a single qbit space. (Convenience method)
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
Matrix U1::calc_one_qubit_u3(double Lambda ) {
    Matrix u1_1qbit = Matrix(2,2); 

    double cos_lambda = 1.0, sin_lambda = 0.0;

    if (Lambda!=0.0) sincos(Lambda, &sin_lambda, &cos_lambda);

    // U1 gate matrix: [[1, 0], [0, e^(i*lambda)]]
    u1_1qbit[0].real = 1.0;
    u1_1qbit[0].imag = 0.0;
    u1_1qbit[1].real = 0.0;
    u1_1qbit[1].imag = 0.0;
    u1_1qbit[2].real = 0.0;
    u1_1qbit[2].imag = 0.0;
    u1_1qbit[3].real = cos_lambda;
    u1_1qbit[3].imag = sin_lambda;

    return u1_1qbit;
}