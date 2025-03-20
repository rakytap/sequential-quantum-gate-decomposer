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
/*! \file RZ_P.cpp
    \brief Class representing a RZ_P gate.
*/

#include "RZ_P.h"



//static tbb::spin_mutex my_mutex;
/**
@brief Nullary constructor of the class.
*/
RZ_P::RZ_P() {

        // number of qubits spanning the matrix of the gate
        qbit_num = -1;
        // the size of the matrix
        matrix_size = -1;
        // A string describing the type of the gate
        type = RZ_P_OPERATION;

        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = -1;
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = -1;

        // logical value indicating whether the matrix creation takes an argument theta
        theta = false;
        // logical value indicating whether the matrix creation takes an argument phi
        phi = false;
        // logical value indicating whether the matrix creation takes an argument lambda
        lambda = false;

        // set static values for the angles
        theta0 = 0.0;
        lambda0 = 0.0;


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
RZ_P::RZ_P(int qbit_num_in, int target_qbit_in) {

	//The stringstream input to store the output messages.
	std::stringstream sstream;

	//Integer value to set the verbosity level of the output messages.
	int verbose_level;

        // number of qubits spanning the matrix of the gate
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the gate
        type = RZ_P_OPERATION;


        if (target_qbit_in >= qbit_num) {
	   verbose_level=1;
	   sstream << "The index of the target qubit is larger than the number of qubits" << std::endl;
	   print(sstream,verbose_level);	    	
	            
	   throw "The index of the target qubit is larger than the number of qubits";
        }
	
        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = target_qbit_in;
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = -1;

        // logical value indicating whether the matrix creation takes an argument theta
        theta = false;
        // logical value indicating whether the matrix creation takes an argument phi
        phi = true;
        // logical value indicating whether the matrix creation takes an argument lambda
        lambda = false;

        // set static values for the angles
        theta0 = 0.0;
        lambda0 = 0.0;

        parameter_num = 1;

        // Parameters theta, phi, lambda of the U3 gate after the decomposition of the unitay is done
        parameters = Matrix_real(1, parameter_num);

}


/**
@brief Destructor of the class
*/
RZ_P::~RZ_P() {


}




/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
RZ_P::apply_to( Matrix_real& parameters, Matrix& input, int parallel ) {


    if (input.rows != matrix_size ) {
        std::string err("Wrong matrix size in RZ_P gate apply");
        throw err;
    }


    double Phi = parameters[0];

    

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(theta0, Phi, lambda0 );

    apply_kernel_to( u3_1qbit, input, false, parallel );


}



/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
RZ_P::apply_from_right( Matrix_real& parameters, Matrix& input ) {


    if (input.cols != matrix_size ) {
        std::string err("Wrong matrix size in RZ_P apply_from_right");
        throw err;    
    }

    double Phi = parameters[0];

    

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(theta0, Phi, lambda0 );


    apply_kernel_from_right(u3_1qbit, input);



}



/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
std::vector<Matrix> 
RZ_P::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
        std::string err("Wrong matrix size in RZ_P apply_derivate_to");
        throw err;
    }

    double Phi = parameters[0] + M_PI/2;

    

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(theta0, Phi, lambda0 ); 
    u3_1qbit[0].real = 0.0;
    u3_1qbit[0].imag = 0.0;

    Matrix&& res_mtx = input.copy();
    apply_kernel_to( u3_1qbit, res_mtx, parallel );

    std::vector<Matrix> ret;
    ret.push_back(res_mtx);


    return ret;


}




/**
@brief Call to set the final optimized parameters of the gate.
@param Theta Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
*/
void RZ_P::set_optimized_parameters(double PhiOver2 ) {

    parameters = Matrix_real(1, parameter_num);

    parameters[0] = PhiOver2;

}


/**
@brief Call to get the final optimized parameters of the gate.
@param parameters_in Preallocated pointer to store the parameters Theta, Phi and Lambda of the U3 gate.
*/
Matrix_real RZ_P::get_optimized_parameters() {

    return parameters.copy();

}

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on a single qbit space.
@param Theta Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
void 
RZ_P::parameters_for_calc_one_qubit( double& ThetaOver2, double& Phi, double& Lambda){

    ThetaOver2 = 0;
    Lambda = 0;
}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
RZ_P* RZ_P::clone() {

    RZ_P* ret = new RZ_P(qbit_num, target_qbit);

    if ( parameters.size() > 0 ) {
        ret->set_optimized_parameters(parameters[0]);
    }
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );


    return ret;

}


