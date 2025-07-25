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
/*! \file RX.cpp
    \brief Class representing a RX gate.
*/

#include "R.h"



//static tbb::spin_mutex my_mutex;
/**
@brief NullaRX constructor of the class.
*/
R::R() {

    name = "R";

    // number of qubits spanning the matrix of the gate
    qbit_num = -1;
    // the size of the matrix
    matrix_size = -1;
    // A string describing the type of the gate
    type = R_OPERATION;

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
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
R::R(int qbit_num_in, int target_qbit_in) {

    name = "R";

    // number of qubits spanning the matrix of the gate
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the gate
    type = R_OPERATION;

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

    parameter_num = 2;

}


/**
@brief Destructor of the class
*/
R::~R() {

}




/**
@brief Call to apply the gate on the input array/matrix by RX*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
R::apply_to( Matrix_real& parameters, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in RX gate apply" << std::endl;
        print(sstream, 0);	        
        exit(-1);
    }


    double ThetaOver2, Phi, Lambda;

    ThetaOver2 = parameters[0];
    Phi = parameters[1];
    
    
    Matrix u3_1qbit = Matrix(2,2);
    u3_1qbit[0].real = std::cos(ThetaOver2); 
    u3_1qbit[0].imag = 0;
    
    u3_1qbit[1].real = -1.*std::sin(ThetaOver2)*std::sin(Phi); 
    u3_1qbit[1].imag = -1.*std::sin(ThetaOver2)*std::cos(Phi);
    
    u3_1qbit[2].real = std::sin(ThetaOver2)*std::sin(Phi); 
    u3_1qbit[2].imag = -1.*std::sin(ThetaOver2)*std::cos(Phi);
    
    u3_1qbit[3].real = std::cos(ThetaOver2); 
    u3_1qbit[3].imag = 0;


    apply_kernel_to( u3_1qbit, input, false, parallel );


}



/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
R::apply_from_right( Matrix_real& parameters, Matrix& input ) {

    if (input.cols != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in U3 apply_from_right" << std::endl;
        print(sstream, 0);	   
        exit(-1);
    }


    double ThetaOver2, Phi, Lambda;

    ThetaOver2 = parameters[0]; 
    Phi = parameters[1] - M_PI/2;
    Lambda = -1.*parameters[1] + M_PI/2;
    
    
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
R::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in RX apply_derivate_to" << std::endl;
        print(sstream, 0);	      
        exit(-1);
    }


    std::vector<Matrix> ret;


    
    double ThetaOver2,Phi;
    ThetaOver2 = parameters_mtx[0];
    Phi = parameters_mtx[1];

    Matrix res_mtx = input.copy();
    Matrix u3_1qbit = Matrix(2,2);
    u3_1qbit[0].real = std::cos(ThetaOver2+ M_PI/2); 
    u3_1qbit[0].imag = 0;
    
    u3_1qbit[1].real = -1.*std::sin(ThetaOver2+ M_PI/2)*std::sin(Phi); 
    u3_1qbit[1].imag = -1.*std::sin(ThetaOver2+ M_PI/2)*std::cos(Phi);
    
    u3_1qbit[2].real = std::sin(ThetaOver2+ M_PI/2)*std::sin(Phi); 
    u3_1qbit[2].imag = -1.*std::sin(ThetaOver2+ M_PI/2)*std::cos(Phi);
    
    u3_1qbit[3].real = std::cos(ThetaOver2+ M_PI/2); 
    u3_1qbit[3].imag = 0;
    
    apply_kernel_to( u3_1qbit, res_mtx, true, parallel );
    ret.push_back(res_mtx);


    Matrix res_mtx2 = input.copy();
    u3_1qbit = Matrix(2,2);
    u3_1qbit[0].real = 0; 
    u3_1qbit[0].imag = 0;
    
    u3_1qbit[1].real = -1.*std::sin(ThetaOver2)*std::sin(Phi+ M_PI/2); 
    u3_1qbit[1].imag = -1.*std::sin(ThetaOver2)*std::cos(Phi+ M_PI/2);
    
    u3_1qbit[2].real = std::sin(ThetaOver2)*std::sin(Phi+ M_PI/2); 
    u3_1qbit[2].imag = -1.*std::sin(ThetaOver2)*std::cos(Phi+ M_PI/2);
    
    u3_1qbit[3].real = 0; 
    u3_1qbit[3].imag = 0;
    
    apply_kernel_to( u3_1qbit, res_mtx2, true, parallel );
    ret.push_back(res_mtx2);
    
    return ret;


}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
R* R::clone() {

    R* ret = new R(qbit_num, target_qbit);
    
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
R::extract_parameters( Matrix_real& parameters ) {

    if ( get_parameter_start_idx() + get_parameter_num() > parameters.size()  ) {
        std::string err("R::extract_parameters: Cant extract parameters, since the dinput arary has not enough elements.");
        throw err;     
    }

    Matrix_real extracted_parameters(1,2);

    extracted_parameters[0] = std::fmod( 2*parameters[ get_parameter_start_idx() ], 4*M_PI);
    extracted_parameters[1] = std::fmod( parameters[ get_parameter_start_idx() +1], 2*M_PI);

    return extracted_parameters;

}
