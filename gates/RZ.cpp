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
/*! \file RZ.cpp
    \brief Class representing a RZ gate.
*/

#include "RZ.h"

//setting local_verbose_parameter 
std::stringstream ss;
int verbose_level;


//static tbb::spin_mutex my_mutex;
/**
@brief NullaRZ constructor of the class.
*/
RZ::RZ() {

        // number of qubits spanning the matrix of the gate
        qbit_num = -1;
        // the size of the matrix
        matrix_size = -1;
        // A string describing the type of the gate
        type = RZ_OPERATION;

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
RZ::RZ(int qbit_num_in, int target_qbit_in) {

        // number of qubits spanning the matrix of the gate
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the gate
        type = RZ_OPERATION;


        if (target_qbit_in >= qbit_num) {
            printf("The index of the target qubit is larger than the number of qubits");
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

        // Parameters theta, phi, lambda of the U3 gate after the decomposition of the unitaRZ is done
        parameters = Matrix_real(1, parameter_num);

}


/**
@brief Destructor of the class
*/
RZ::~RZ() {


}




/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
RZ::apply_to( Matrix_real& parameters, Matrix& input, const double scale=1.0 ) {

    if (input.rows != matrix_size ) {
	ss << "Wrong matrix size in RZ gate apply" << std::endl;
	verbose_level=1;
        logging::printnewsq(ss, verbose_level);	
	ss.str("");
        
        exit(-1);
    }

    double Theta, Phi, Lambda;

    Theta = theta0;
    Phi = parameters[0];
    Lambda = lambda0;
    

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(Theta, Phi, Lambda, scale );


    apply_kernel_to( u3_1qbit, input );


}



/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
RZ::apply_from_right( Matrix_real& parameters, Matrix& input ) {


    if (input.cols != matrix_size ) {
	ss << "Wrong matrix size in U3 apply_from_right" << std::endl;
	verbose_level=1;
        logging::printnewsq(ss, verbose_level);	
	ss.str("");
        
        exit(-1);
    }

    double Theta, Phi, Lambda;

    Theta = theta0;
    Phi = parameters[0];
    Lambda = lambda0;
    

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(Theta, Phi, Lambda );


    apply_kernel_from_right(u3_1qbit, input);



}



/**
@brief ???????????????
*/
std::vector<Matrix> 
RZ::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input ) {

    if (input.rows != matrix_size ) {
	ss << "Wrong matrix size in RZ apply_derivate_to" << std::endl;
	verbose_level=1;
        logging::printnewsq(ss, verbose_level);	
	ss.str("");
        
        exit(-1);
    }

    double Theta, Phi, Lambda;

    Theta = theta0;
    Phi = parameters_mtx[0] + M_PI/2;
    Lambda = lambda0;

    Matrix&& res_mtx = input.copy();
    

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(Theta, Phi, Lambda, 1.0 );
    memset(u3_1qbit.get_data(), 0.0, 2*sizeof(QGD_Complex16));


    apply_kernel_to( u3_1qbit, res_mtx );

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
void RZ::set_optimized_parameters(double Phi ) {

    parameters = Matrix_real(1, parameter_num);

    parameters[0] = Phi;

}


/**
@brief Call to get the final optimized parameters of the gate.
@param parameters_in Preallocated pointer to store the parameters Theta, Phi and Lambda of the U3 gate.
*/
Matrix_real RZ::get_optimized_parameters() {

    return parameters.copy();

}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
RZ* RZ::clone() {

    RZ* ret = new RZ(qbit_num, target_qbit);

    if ( parameters.size() > 0 ) {
        ret->set_optimized_parameters(parameters[0]);
    }


    return ret;

}


