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
    \brief Class representing a YY rotation gate.
*/

#include "RZZ.h"
#include "apply_large_kernel_to_input_AVX.h"
#include "apply_large_kernel_to_input.h"


//static tbb::spin_mutex my_mutex;
/**
@brief Nullary constructor of the class.
*/
RZZ::RZZ() : RY() {

        // A string describing the type of the gate
        type = RZZ_OPERATION;

}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
RZZ::RZZ(int qbit_num_in, int target_qbit_in, int control_qbit_in) : RY(qbit_num_in, target_qbit_in) {


        // A string describing the type of the gate
        type = RZZ_OPERATION;


        if (control_qbit_in >= qbit_num) {
	    std::stringstream sstream;
	    sstream << "The index of the control qubit is larger than the number of qubits in CRY gate." << std::endl;
	    print(sstream, 0);	  
            throw sstream.str();
        }

        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = control_qbit_in;


}

/**
@brief Destructor of the class
*/
RZZ::~RZZ() {

}




/**
@brief Call to apply the gate on the input array/matrix by RYY*input
@param parameters An array of parameters to calculate the matrix of the RYY gate.
@param input The input array on which the gate is applied
@param parallel Set true to apply parallel kernels, false otherwise (optional)
*/
void 
RZZ::apply_to( Matrix_real& parameters, Matrix& input, bool parallel ) {


    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CRY gate apply" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }


    double ThetaOver2;

    ThetaOver2 = parameters[0];
/*
    ThetaOver2 = theta0;
    Phi = parameters[0];
    Lambda = lambda0;
*/
/*  
Phi = Phi + M_PI;
Phi = (1.0-std::cos(Phi/2))*M_PI;
Phi = Phi - M_PI;
*/
//Phi = 0.5*(1.0-std::cos(Phi))*M_PI;

    // get the U3 gate of one qubit
    Matrix U_2qbit(4,4);
    memset(U_2qbit.get_data(), 0.0, (U_2qbit.size()*2)*sizeof(double) );      
    U_2qbit[0].real = std::cos(ThetaOver2);
    U_2qbit[0].imag = -1.*std::sin(ThetaOver2);
    U_2qbit[1*4+1].real = std::cos(ThetaOver2);
    U_2qbit[1*4+1].imag =  1.*std::sin(ThetaOver2);
    U_2qbit[2*4+2].imag = 1.*std::sin(ThetaOver2);
    U_2qbit[2*4+2].real = std::cos(ThetaOver2);
    U_2qbit[3*4+3].imag = -1.*std::sin(ThetaOver2);
    U_2qbit[3*4+3].real = std::cos(ThetaOver2);
    int inner = (target_qbit>control_qbit) ? control_qbit:target_qbit;
    int outer = (target_qbit>control_qbit) ? target_qbit:control_qbit;
    if (parallel){
        apply_2qbit_kernel_to_state_vector_input_AVX(U_2qbit,input,inner,outer,input.size());
    }
    else{
        apply_2qbit_kernel_to_state_vector_input(U_2qbit,input,inner,outer,input.size());
    }

}



/**
@brief Call to apply the gate on the input array/matrix by input*CRY
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
RZZ::apply_from_right( Matrix_real& parameters, Matrix& input ) {




}


/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
*/
std::vector<Matrix> 
RZZ::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input ) {



    std::vector<Matrix> ret;

    double ThetaOver2;

    ThetaOver2 = parameters_mtx[0]+M_PI/2;


    // the resulting matrix
    Matrix res_mtx = input.copy();   

    Matrix U_2qbit(4,4);
    memset(U_2qbit.get_data(), 0.0, (U_2qbit.size()*2)*sizeof(double) );      
    U_2qbit[0].real = std::cos(ThetaOver2);
    U_2qbit[0].imag = -1.*std::sin(ThetaOver2);
    U_2qbit[1*4+1].real = std::cos(ThetaOver2);
    U_2qbit[1*4+1].imag =  1.*std::sin(ThetaOver2);
    U_2qbit[2*4+2].imag = 1.*std::sin(ThetaOver2);
    U_2qbit[2*4+2].real = std::cos(ThetaOver2);
    U_2qbit[3*4+3].imag = -1.*std::sin(ThetaOver2);
    U_2qbit[3*4+3].real = std::cos(ThetaOver2);

    int inner = (target_qbit>control_qbit) ? control_qbit:target_qbit;
    int outer = (target_qbit>control_qbit) ? target_qbit:control_qbit;

        apply_2qbit_kernel_to_state_vector_input(U_2qbit,res_mtx,inner,outer,input.size());

    ret.push_back(res_mtx);
    return ret;


}



/**
@brief Call to set the final optimized parameters of the gate.
@param ThetaOver2 Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
*/
void RZZ::set_optimized_parameters(double ThetaOver2 ) {

    parameters = Matrix_real(1, parameter_num);

    parameters[0] = ThetaOver2;

}


/**
@brief Call to get the final optimized parameters of the gate.
@param parameters_in Preallocated pointer to store the parameters ThetaOver2, Phi and Lambda of the U3 gate.
*/
Matrix_real RZZ::get_optimized_parameters() {

    return parameters.copy();

}



/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
RZZ* RZZ::clone() {

    RZZ* ret = new RZZ(qbit_num, target_qbit, control_qbit);

    if ( parameters.size() > 0 ) {
        ret->set_optimized_parameters(parameters[0]);
    }


    return ret;

}
