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
/*! \file CU.cpp
    \brief Class representing a CNOT gate.
*/

#include "CU.h"


// pi/2
static double M_PIOver2 = M_PI/2;



/**
@brief Nullary constructor of the class.
*/
CU::CU() : U3() {

    // A string labeling the gate operation
    name = "CU";

    // A string describing the type of the gate
    type = CU_OPERATION;

    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
    control_qbit = -1;


}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
CU::CU(int qbit_num_in,  int target_qbit_in, int control_qbit_in) : U3(qbit_num_in,  target_qbit_in) {


    // A string labeling the gate operation
    name = "CU";

    // A string describing the type of the gate
    type = CU_OPERATION;


    if (control_qbit_in >= qbit_num) {
        std::stringstream sstream;
        sstream << "The index of the control qubit is larger than the number of qubits" << std::endl;
        print(sstream, 0);	    	
        throw sstream.str();
    }
    
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
    control_qbit = control_qbit_in;

    // compared to U3 here we have another parameter controling the global phase on the target qubit
    parameter_num = 4;


}

/**
@brief Destructor of the class
*/
CU::~CU() {

}



/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
CU::apply_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {
    
    if (input.rows != matrix_size ) {
        std::string err("U3::apply_to: Wrong input size in U3 gate apply.");
        throw err;    
    }

    if (parameters_mtx.size() < parameter_num ) {
        std::string err("CU::apply_to: Input parameter array should contain at least " +  std::to_string(parameter_num) + " parameters");
        throw err;    
    }

    double ThetaOver2 = parameters_mtx[0];
    double Phi = parameters_mtx[1];
    double Lambda = parameters_mtx[2];
    double global_phase = parameters_mtx[3];

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );

    
    QGD_Complex16 global_phase_factor;
    global_phase_factor.real = 1.0;
    global_phase_factor.imag = 0.0;

	if (global_phase != 0.0) {
        sincos(global_phase, &global_phase_factor.imag, &global_phase_factor.real);
    }

    // apply the global phase on th egate kernel
    mult( global_phase_factor, u3_1qbit);


    apply_kernel_to( u3_1qbit, input, false, parallel );
}


/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
CU::apply_from_right( Matrix_real& parameters_mtx, Matrix& input ) {
    //TODO: check

    if (input.cols != matrix_size ) {
        std::string err("CU::apply_from_right: Wrong matrix size in apply_from_right.");
        throw err;    
    }

    if (parameters_mtx.size() < parameter_num ) {
        std::string err("CU::apply_from_right: Input parameter array should contain at least " +  std::to_string(parameter_num) + " parameters");
        throw err;    
    }

    double ThetaOver2 = parameters_mtx[0];
    double Phi = parameters_mtx[1];
    double Lambda = parameters_mtx[2];
    double global_phase = parameters_mtx[3];

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );

    QGD_Complex16 global_phase_factor;
    global_phase_factor.real = 1.0;
    global_phase_factor.imag = 0.0;

	if (global_phase != 0.0) {
        sincos(global_phase, &global_phase_factor.imag, &global_phase_factor.real);
    }

    // apply the global phase on th egate kernel
    mult( global_phase_factor, u3_1qbit);

    apply_kernel_from_right(u3_1qbit, input);
}



/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
std::vector<Matrix> 
CU::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {
    //TODO: check

    if (input.rows != matrix_size ) {
        std::string err("CU::apply_derivate_to: Wrong matrix size.");
        throw err;    
    }

    if (parameters_mtx.size() < parameter_num ) {
        std::string err("CU::apply_derivate_to: Input parameter array should contain at least " +  std::to_string(parameter_num) + " parameters");
        throw err;    
    }

    std::vector<Matrix> ret;

    double ThetaOver2 = parameters_mtx[0];
    double Phi = parameters_mtx[1];
    double Lambda = parameters_mtx[2];
    double global_phase = parameters_mtx[3];
    bool deriv = true;


    Matrix u3_1qbit_theta = calc_one_qubit_u3(ThetaOver2+M_PIOver2, Phi, Lambda);
    QGD_Complex16 global_phase_factor;
    global_phase_factor.real = 1.0;
    global_phase_factor.imag = 0.0;

	if (global_phase != 0.0) {
        sincos(global_phase, &global_phase_factor.imag, &global_phase_factor.real);
    }

    // apply the global phase on th egate kernel
    mult( global_phase_factor, u3_1qbit_theta);

    Matrix res_mtx_theta = input.copy();
    apply_kernel_to( u3_1qbit_theta, res_mtx_theta, deriv, parallel );
    ret.push_back(res_mtx_theta);

    /////////////////////////////////////////////////////////
    Matrix u3_1qbit_phi = calc_one_qubit_u3(ThetaOver2, Phi+M_PIOver2, Lambda );
    memset(u3_1qbit_phi.get_data(), 0.0, 2*sizeof(QGD_Complex16) );

    // apply the global phase on th egate kernel
    mult( global_phase_factor, u3_1qbit_phi);

    Matrix res_mtx_phi = input.copy();
    apply_kernel_to( u3_1qbit_phi, res_mtx_phi, deriv, parallel );
    ret.push_back(res_mtx_phi);

    //////////////////////////////////////////////////////////
    Matrix u3_1qbit_lambda = calc_one_qubit_u3(ThetaOver2, Phi, Lambda+M_PIOver2 );
    memset(u3_1qbit_lambda.get_data(), 0.0, sizeof(QGD_Complex16) );
    memset(u3_1qbit_lambda.get_data()+2, 0.0, sizeof(QGD_Complex16) );

    // apply the global phase on th egate kernel
    mult( global_phase_factor, u3_1qbit_lambda);

    Matrix res_mtx_lambda = input.copy();
    apply_kernel_to( u3_1qbit_lambda, res_mtx_lambda, deriv, parallel );
    ret.push_back(res_mtx_lambda);


    ///////////////////////////////////////////////////////////
    Matrix u3_1qbit_global_phase = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );

    // derivate of the global phase 
    double tmp = global_phase_factor.real;
    global_phase_factor.real = -global_phase_factor.imag;
    global_phase_factor.imag = tmp;

    // apply the global phase on th egate kernel
    mult( global_phase_factor, u3_1qbit_global_phase);

    Matrix res_mtx_phase_factor = input.copy();
    apply_kernel_to( u3_1qbit_global_phase, res_mtx_phase_factor, deriv, parallel );
    ret.push_back(res_mtx_phase_factor);




    return ret;
}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CU* CU::clone() {

    CU* ret = new CU( qbit_num, target_qbit, control_qbit );
    
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
CU::extract_parameters( Matrix_real& parameters ) {

    if ( get_parameter_start_idx() + get_parameter_num() > parameters.size()  ) {
        std::string err("CU::extract_parameters: Cant extract parameters, since the input arary has not enough elements.");
        throw err;     
    }

    Matrix_real extracted_parameters(1, parameter_num);

    extracted_parameters[0] = std::fmod( 2*parameters[ get_parameter_start_idx() ], 4*M_PI);
    extracted_parameters[1] = std::fmod( parameters[ get_parameter_start_idx()+1 ], 2*M_PI);
    extracted_parameters[2] = std::fmod( parameters[ get_parameter_start_idx()+2 ], 2*M_PI);
    extracted_parameters[3] = std::fmod( parameters[ get_parameter_start_idx()+3 ], 2*M_PI);

    return extracted_parameters;
}