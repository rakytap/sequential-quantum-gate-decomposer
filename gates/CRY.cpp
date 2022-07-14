/*
Created on Fri Jun 26 14:13:26 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

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
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/

void 
CRY::apply_to( Matrix_real& parameters, Matrix& input, const double scale ) {


    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CRY gate apply" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }


    double Theta, Phi, Lambda;

    Theta = parameters[0];
    Phi = phi0;
    Lambda = lambda0;
/*
    Theta = theta0;
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
    Matrix u3_1qbit = calc_one_qubit_u3(Theta, Phi, Lambda );


    // apply the computing kernel on the matrix
    apply_kernel_to(u3_1qbit, input);

}



/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
CRY::apply_from_right( Matrix_real& parameters, Matrix& input ) {


    if (input.cols != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CRY apply_from_right" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }

    double Theta, Phi, Lambda;

    Theta = parameters[0];
    Phi = phi0;
    Lambda = lambda0;
/*
    Theta = theta0;
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
    Matrix u3_1qbit = calc_one_qubit_u3(Theta, Phi, Lambda );

    // apply the computing kernel on the matrix
    apply_kernel_from_right(u3_1qbit, input);

}


/**
@brief ???????????????
*/
std::vector<Matrix> 
CRY::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input ) {

    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CRY gate apply" << std::endl;
        print(sstream, 0);	   
        exit(-1);
    }

    std::vector<Matrix> ret;

    double Theta, Phi, Lambda;

    Theta = parameters_mtx[0]+M_PI;
    Phi = phi0;
    Lambda = lambda0;

    // the resulting matrix
    Matrix res_mtx = input.copy();   


    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(Theta, Phi, Lambda, 0.5 );


    // apply the computing kernel on the matrix
    bool deriv = true;
    apply_kernel_to(u3_1qbit, res_mtx, deriv);

    ret.push_back(res_mtx);
    return ret;


}



/**
@brief Call to set the final optimized parameters of the gate.
@param Theta Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
*/
void CRY::set_optimized_parameters(double Theta ) {

    parameters = Matrix_real(1, parameter_num);

    parameters[0] = Theta;

}


/**
@brief Call to get the final optimized parameters of the gate.
@param parameters_in Preallocated pointer to store the parameters Theta, Phi and Lambda of the U3 gate.
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


    return ret;

}
