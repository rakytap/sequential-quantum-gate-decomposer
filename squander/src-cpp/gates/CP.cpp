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
/*! \file CP.cpp
    \brief Class representing a CP gate.
*/

#include "CP.h"


// pi/2
static double M_PIOver2 = M_PI/2;



/**
@brief Nullary constructor of the class.
*/
CP::CP() : U1() {

    // A string labeling the gate operation
    name = "CP";

    // A string describing the type of the gate
    type = CP_OPERATION;

    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
    control_qbit = -1;


}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
CP::CP(int qbit_num_in,  int target_qbit_in, int control_qbit_in) : U1(qbit_num_in,  target_qbit_in) {


    // A string labeling the gate operation
    name = "CP";

    // A string describing the type of the gate
    type = CP_OPERATION;


    if (control_qbit_in >= qbit_num) {
        std::stringstream sstream;
        sstream << "The index of the control qubit is larger than the number of qubits" << std::endl;
        print(sstream, 0);	    	
        throw sstream.str();
    }
    
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
    control_qbit = control_qbit_in;

    // compared to U3 here we have another parameter controling the global phase on the target qubit
    parameter_num = 1;


}

/**
@brief Destructor of the class
*/
CP::~CP() {

}



/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
CP::apply_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {
    
    if (input.rows != matrix_size ) {
        std::string err("U3::apply_to: Wrong input size in U3 gate apply.");
        throw err;    
    }

    if (parameters_mtx.size() < parameter_num ) {
        std::string err("CU::apply_to: Input parameter array should contain at least " +  std::to_string(parameter_num) + " parameters");
        throw err;    
    }
    if (input.rows != matrix_size ) {
        std::string err("U1::apply_to: Wrong input size in U1 gate apply.");
        throw err;    
    }


    double ThetaOver2, Phi, Lambda;

    Lambda = parameters_mtx[0];
    parameters_for_calc_one_qubit(ThetaOver2, Phi, Lambda);
  
    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );


    apply_kernel_to( u3_1qbit, input, false, parallel );
}


/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
CP::apply_from_right( Matrix_real& parameters_mtx, Matrix& input ) {
    //TODO: check

    if (input.rows != matrix_size ) {
        std::string err("U3::apply_to: Wrong input size in U3 gate apply.");
        throw err;    
    }

    if (parameters_mtx.size() < parameter_num ) {
        std::string err("CU::apply_to: Input parameter array should contain at least " +  std::to_string(parameter_num) + " parameters");
        throw err;    
    }
    if (input.rows != matrix_size ) {
        std::string err("U1::apply_to: Wrong input size in U1 gate apply.");
        throw err;    
    }


    double ThetaOver2, Phi, Lambda;

    Lambda = parameters_mtx[0];
    parameters_for_calc_one_qubit(ThetaOver2, Phi, Lambda);
  
    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );

    apply_kernel_from_right(u3_1qbit, input);
}



/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
std::vector<Matrix> 
CP::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {
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
    
    Matrix_real parameters_tmp(1,1);

    double ThetaOver2, Phi, Lambda;

    Lambda = parameters_mtx[0] + M_PI/2; 

    parameters_for_calc_one_qubit(ThetaOver2, Phi, Lambda);

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );
    // set to zero for derivate
    u3_1qbit[0].real =0.0;
    u3_1qbit[0].imag =0.0;

    Matrix res_mtx = input.copy();
    apply_kernel_to(u3_1qbit, res_mtx, true ,parallel);
    ret.push_back(res_mtx);
    
    return ret;
}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CP* CP::clone() {

    CP* ret = new CP( qbit_num, target_qbit, control_qbit );
    
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
CP::extract_parameters( Matrix_real& parameters ) {

    if ( get_parameter_start_idx() + get_parameter_num() > parameters.size()  ) {
        std::string err("CU::extract_parameters: Cant extract parameters, since the input arary has not enough elements.");
        throw err;     
    }

    Matrix_real extracted_parameters(1, parameter_num);

    extracted_parameters[0] = std::fmod( parameters[ get_parameter_start_idx() ], 2*M_PI);
    return extracted_parameters;
}