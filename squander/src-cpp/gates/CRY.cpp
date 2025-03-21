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
/*! \file CRY.cpp
    \brief Class representing a controlled Y rotattion gate.
*/

#include "CRY.h"



//static tbb::spin_mutex my_mutex;
/**
@brief Nullary constructor of the class.
*/
CRY::CRY() : RY() {

        // A string describing the type of the gate
        type = CRY_OPERATION;

}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
CRY::CRY(int qbit_num_in, int target_qbit_in, int control_qbit_in) : RY(qbit_num_in, target_qbit_in) {


        // A string describing the type of the gate
        type = CRY_OPERATION;


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
CRY::~CRY() {

}




/**
@brief Call to apply the gate on the input array/matrix by CRY3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
CRY::apply_to( Matrix_real& parameters, Matrix& input, int parallel ) {


    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CRY gate apply" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }


    double ThetaOver2, Phi, Lambda;

    ThetaOver2 = parameters[0];
    Phi = phi0;
    Lambda = lambda0;
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
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );


    // apply the computing kernel on the matrix
    apply_kernel_to(u3_1qbit, input, false, parallel);

}



/**
@brief Call to apply the gate on the input array/matrix by input*CRY
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
CRY::apply_from_right( Matrix_real& parameters, Matrix& input ) {


    if (input.cols != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CRY apply_from_right" << std::endl;
        print(sstream, 0);	
        throw "Wrong matrix size in CRY apply_from_right";
    }

    double ThetaOver2, Phi, Lambda;

    ThetaOver2 = parameters[0];
    Phi = phi0;
    Lambda = lambda0;
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
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );

    // apply the computing kernel on the matrix
    apply_kernel_from_right(u3_1qbit, input);

}


/**
@brief Call to evaluate the derivate of the circuit on an input with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
std::vector<Matrix> 
CRY::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CRY gate apply" << std::endl;
        print(sstream, 0);	   
        throw "Wrong matrix size in CRY gate apply";
    }

    std::vector<Matrix> ret;

    double ThetaOver2, Phi, Lambda;

    ThetaOver2 = parameters_mtx[0]+M_PI/2;
    Phi = phi0;
    Lambda = lambda0;

    // the resulting matrix
    Matrix res_mtx = input.copy();   


    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );


    // apply the computing kernel on the matrix
    bool deriv = true;
    apply_kernel_to(u3_1qbit, res_mtx, deriv, parallel);

    ret.push_back(res_mtx);
    return ret;


}



/**
@brief Call to set the final optimized parameters of the gate.
@param ThetaOver2 Real parameter standing for the parameter theta.
*/
void CRY::set_optimized_parameters(double ThetaOver2 ) {

    parameters = Matrix_real(1, parameter_num);

    parameters[0] = ThetaOver2;

}


/**
@brief Call to get the final optimized parameters of the gate.
@return Returns with an array containing the optimized parameter
*/
Matrix_real CRY::get_optimized_parameters() {

    return parameters.copy();

}



/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CRY* CRY::clone() {

    CRY* ret = new CRY(qbit_num, target_qbit, control_qbit);

    if ( parameters.size() > 0 ) {
        ret->set_optimized_parameters(parameters[0]);
    }

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
CRY::extract_parameters( Matrix_real& parameters ) {

    if ( get_parameter_start_idx() + get_parameter_num() > parameters.size()  ) {
        std::string err("CRY::extract_parameters: Cant extract parameters, since the dinput arary has not enough elements.");
        throw err;     
    }

    Matrix_real extracted_parameters(1,1);

    extracted_parameters[0] = std::fmod( 2*parameters[ get_parameter_start_idx() ], 4*M_PI);

    return extracted_parameters;

}
