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


/**
@brief Nullary constructor of the class.
*/
U1::U1() {

    // A string labeling the gate operation
    name = "U1";

    // number of qubits spanning the matrix of the gate
    qbit_num = -1;
    // the size of the matrix
    matrix_size = -1;
    // A string describing the type of the gate
    type = U1_OPERATION;

    // The index of the qubit on which the gate acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
    control_qbit = -1;

    parameter_num = 0;

}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
*/
U1::U1(int qbit_num_in, int target_qbit_in) {
    
    // A string labeling the gate operation
    name = "U1";

    // number of qubits spanning the matrix of the gate
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the gate
    type = U1_OPERATION;

    if (target_qbit_in >= qbit_num) {
        std::stringstream sstream;
        sstream << "The index of the target qubit is larger than the number of qubits" << std::endl;
    print(sstream, 0);        
        throw "The index of the target qubit is larger than the number of qubits";
    }
    
    // The index of the qubit on which the gate acts (target_qbit >= 0)
    target_qbit = target_qbit_in;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
    control_qbit = -1;

    parameter_num = 1;

}


/**
@brief Destructor of the class
*/
U1::~U1() {

}


/**
@brief Call to apply the gate on the input array/matrix by U1*input
@param parameters An array of parameters to calculate the matrix of the U1 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
U1::apply_to( Matrix_real& parameters, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
        std::string err("U1::apply_to: Wrong input size in U1 gate apply.");
        throw err;    
    }


    double ThetaOver2, Phi, Lambda;

    Lambda = parameters[0];
    parameters_for_calc_one_qubit(ThetaOver2, Phi, Lambda);
  
    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );


    apply_kernel_to( u3_1qbit, input, false, parallel );


}


/**
@brief Call to apply the gate on the input array/matrix by input*U1
@param parameters An array of parameters to calculate the matrix of the U1 gate.
@param input The input array on which the gate is applied
*/
void 
U1::apply_from_right( Matrix_real& parameters, Matrix& input ) {

    if (input.cols != matrix_size ) {
        std::string err("U1::apply_from_right: Wrong matrix size in U1 apply_from_right.");
        throw err;    
    }


    double ThetaOver2, Phi, Lambda;

    Lambda = parameters[0]; 
    parameters_for_calc_one_qubit(ThetaOver2, Phi, Lambda);

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );


    apply_kernel_from_right(u3_1qbit, input);


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
    
    Matrix_real parameters_tmp(1,1);

    parameters_tmp[0] = parameters_mtx[0] + M_PI/2;
    Matrix res_mtx = input.copy();
    apply_to(parameters_tmp, res_mtx, parallel);
    ret.push_back(res_mtx);

    
    
    return ret;


}


/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on a single qbit space.
@param ThetaOver2 Real parameter standing for the parameter theta/2.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
void 
U1::parameters_for_calc_one_qubit( double& ThetaOver2, double& Phi, double& Lambda){

    ThetaOver2 = 0.0;
    Phi = 0.0;
    // Lambda is passed through unchanged

}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
U1* U1::clone() {

    U1* ret = new U1(qbit_num, target_qbit);
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;
}


/**
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is embedded.
@param parameters The parameter array corresponding to the circuit in which the gate is embedded
@return Returns with the array of the extracted parameters.
*/
Matrix_real 
U1::extract_parameters( Matrix_real& parameters ) {

    if ( get_parameter_start_idx() + get_parameter_num() > parameters.size()  ) {
        std::string err("U1::extract_parameters: Cant extract parameters, since the dinput arary has not enough elements.");
        throw err;     
    }

    Matrix_real extracted_parameters(1,1);

    extracted_parameters[0] = std::fmod( parameters[ get_parameter_start_idx() ], 2*M_PI);

    return extracted_parameters;

}