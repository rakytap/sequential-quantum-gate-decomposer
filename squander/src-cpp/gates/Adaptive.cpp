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

#include "Adaptive.h"
#include "common.h"



//static tbb::spin_mutex my_mutex;
/**
@brief Nullary constructor of the class.
*/
Adaptive::Adaptive() : CRY() {

        // A string describing the type of the gate
        type = ADAPTIVE_OPERATION;

        limit = 1;

}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
Adaptive::Adaptive(int qbit_num_in, int target_qbit_in, int control_qbit_in) : CRY(qbit_num_in, target_qbit_in, control_qbit_in) {

        // A string describing the type of the gate
        type = ADAPTIVE_OPERATION;

        limit = 1;
}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
Adaptive::Adaptive(int qbit_num_in, int target_qbit_in, int control_qbit_in, int limit_in) : CRY(qbit_num_in, target_qbit_in, control_qbit_in) {

        // A string describing the type of the gate
        type = ADAPTIVE_OPERATION;

        limit = limit_in;
}

/**
@brief Destructor of the class
*/
Adaptive::~Adaptive() {

}




/**
@brief Call to apply the gate on the input array/matrix
@param parameters An array of parameters to calculate the matrix of the gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/

void 
Adaptive::apply_to( Matrix_real& parameters, Matrix& input, int parallel ) {


    if (input.rows != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in Adaptive gate apply" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }

    double Phi = parameters[0];

    Matrix_real Phi_transformed(1,1);
//    Phi_transformed[0] = Phi;
//    Phi_transformed[0] = 0.5*(1.0-std::cos(Phi))*M_PI;
    Phi = activation_function( Phi, limit );
    Phi_transformed[0] = Phi;


/*
Phi = Phi + M_PI;
Phi = (1.0-std::cos(Phi/2))*M_PI;
Phi_transformed[0] = Phi - M_PI;
*/



    CRY::apply_to( Phi_transformed, input, parallel );



}



/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
Adaptive::apply_from_right( Matrix_real& parameters, Matrix& input ) {


    if (input.cols != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in Adaptive apply_from_right" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }

    double Phi = parameters[0];

    Matrix_real Phi_transformed(1,1);
//    Phi_transformed[0] = Phi;
//    Phi_transformed[0] = 0.5*(1.0-std::cos(Phi))*M_PI;
    Phi = activation_function( Phi, limit );
    Phi_transformed[0] = Phi;
/*
Phi = Phi + M_PI;
Phi = (1.0-std::cos(Phi/2))*M_PI;
Phi_transformed[0] = Phi - M_PI;
*/

    CRY::apply_from_right( Phi_transformed, input );


}



/**
@brief Call to apply the gate on the input array/matrix.
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
std::vector<Matrix>
Adaptive::apply_derivate_to( Matrix_real& parameters, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in Adaptive gate apply" << std::endl;
        print(sstream, 0);	     
        exit(-1);
    }

    double Phi = parameters[0];

    Matrix_real Phi_transformed(1,1);
//    Phi_transformed[0] = Phi;
//    Phi_transformed[0] = 0.5*(1.0-std::cos(Phi))*M_PI;
    Phi = activation_function( Phi, limit );
    Phi_transformed[0] = Phi;


/*
Phi = Phi + M_PI;
Phi = (1.0-std::cos(Phi/2))*M_PI;
Phi_transformed[0] = Phi - M_PI;
*/



    return CRY::apply_derivate_to( Phi_transformed, input, parallel );



}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
Adaptive* Adaptive::clone() {

    Adaptive* ret = new Adaptive(qbit_num, target_qbit, control_qbit);

    if ( parameters.size() > 0 ) {
        ret->set_optimized_parameters(parameters[0]);
    }
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    ret->set_limit( limit );


    return ret;

}



/**
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is embedded.
@param parameters The parameter array corresponding to the circuit in which the gate is embedded
@return Returns with the array of the extracted parameters.
*/
Matrix_real 
Adaptive::extract_parameters( Matrix_real& parameters ) {

    if ( get_parameter_start_idx() + get_parameter_num() > parameters.size()  ) {
        std::string err("Adaptive::extract_parameters: Cant extract parameters, since the dinput arary has not enough elements.");
        throw err;     
    }

    Matrix_real extracted_parameters(1,1);

    extracted_parameters[0] = std::fmod( 2*parameters[ get_parameter_start_idx() ], 4*M_PI);

    return extracted_parameters;

}



/**
@brief ???????????
*/
void 
Adaptive::set_limit( int limit_in ) {

    limit = limit_in;

}


/**
@brief ???????????
*/
int 
Adaptive::get_limit() {

    return limit;
}

