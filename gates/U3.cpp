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
/*! \file U3.cpp
    \brief Class representing a U3 gate.
*/

#include "U3.h"

// pi/2
static double M_PIOver2 = M_PI/2;


//static tbb::spin_mutex my_mutex;
/**
@brief Nullary constructor of the class.
*/
U3::U3() {

        // number of qubits spanning the matrix of the gate
        qbit_num = -1;
        // the size of the matrix
        matrix_size = -1;
        // A string describing the type of the gate
        type = U3_OPERATION;

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
U3::U3(int qbit_num_in, int target_qbit_in, bool theta_in, bool phi_in, bool lambda_in) {

        // number of qubits spanning the matrix of the gate
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the gate
        type = U3_OPERATION;


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

        // logical value indicating whether the matrix creation takes an argument theta
        theta = theta_in;
        // logical value indicating whether the matrix creation takes an argument phi
        phi = phi_in;
        // logical value indicating whether the matrix creation takes an argument lambda
        lambda = lambda_in;


        // The number of free parameters
        if (theta && !phi && lambda) {
            parameter_num = 2;
        }

        else if (theta && phi && lambda) {
            parameter_num = 3;
        }

        else if (!theta && phi && lambda) {
            parameter_num = 2;
        }

        else if (theta && phi && !lambda) {
            parameter_num = 2;
        }

        else if (!theta && !phi && lambda) {
            parameter_num = 1;
        }

        else if (!theta && phi && !lambda) {
            parameter_num = 1;
        }

        else if (theta && !phi && !lambda) {
            parameter_num = 1;
        }

        else {
            parameter_num = 0;
        }

        // Parameters theta, phi, lambda of the U3 gate after the decomposition of the unitary is done
        parameters = Matrix_real(1, parameter_num);

}


/**
@brief Destructor of the class
*/
U3::~U3() {

}



/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@return Returns with a matrix of the gate
*/
Matrix
U3::get_matrix( Matrix_real& parameters ) {

        Matrix U3_matrix = create_identity(matrix_size);
        apply_to(parameters, U3_matrix);

#ifdef DEBUG
        if (U3_matrix.isnan()) {
            std::stringstream sstream;
	    sstream << "U3::get_matrix: U3_matrix contains NaN." << std::endl;
	    verbose_level=1;
            print(sstream,verbose_level);	  
        }
#endif

        return U3_matrix;

}

/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
U3::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& input ) {


    for ( std::vector<Matrix>::iterator it=input.begin(); it != input.end(); it++ ) {
        apply_to( parameters_mtx, *it );
    }

}



/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
U3::apply_to( Matrix_real& parameters_mtx, Matrix& input ) {


    if (input.rows != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in U3 gate apply" << std::endl;
        print(sstream, 0);	        
        exit(-1);
    }


    double ThetaOver2, Phi, Lambda;

    if (theta && !phi && lambda) {
        ThetaOver2 = parameters_mtx[0];
        Phi = 0.0;
        Lambda = parameters_mtx[1];
    }

    else if (theta && phi && lambda) {
        ThetaOver2 = parameters_mtx[0];
        Phi = parameters_mtx[1];
        Lambda = parameters_mtx[2];
    }

    else if (!theta && phi && lambda) {
        ThetaOver2 = 0.0;
        Phi = parameters_mtx[0];
        Lambda = parameters_mtx[1];
    }

    else if (theta && phi && !lambda) {
        ThetaOver2 = parameters_mtx[0];
        Phi = parameters_mtx[1];
        Lambda = 0.0;
    }

    else if (!theta && !phi && lambda) {
        ThetaOver2 = 0.0;
        Phi = 0.0;
        Lambda = parameters_mtx[0];
    }

    else if (!theta && phi && !lambda) {
        ThetaOver2 = 0.0;
        Phi = parameters_mtx[0];
        Lambda = 0.0;
    }

    else if (theta && !phi && !lambda) {
        ThetaOver2 = parameters_mtx[0];
        Phi = 0.0;
        Lambda = 0.0;
    }

    else {
        ThetaOver2 = 0.0;
        Phi = 0.0;
        Lambda = 0.0;
    }


    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );


    apply_kernel_to( u3_1qbit, input );


}








/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
U3::apply_from_right( Matrix_real& parameters_mtx, Matrix& input ) {

    if (input.cols != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in U3 apply_from_right" << std::endl;
        print(sstream, 1);	      
        exit(-1);
    }

    double ThetaOver2, Phi, Lambda;

    if (theta && !phi && lambda) {
        ThetaOver2 = parameters_mtx[0];
        Phi = 0.0;
        Lambda = parameters_mtx[1];
    }

    else if (theta && phi && lambda) {
        ThetaOver2 = parameters_mtx[0];
        Phi = parameters_mtx[1];
        Lambda = parameters_mtx[2];
    }

    else if (!theta && phi && lambda) {
        ThetaOver2 = 0.0;
        Phi = parameters_mtx[0];
        Lambda = parameters_mtx[1];
    }

    else if (theta && phi && !lambda) {
        ThetaOver2 = parameters_mtx[0];
        Phi = parameters_mtx[1];
        Lambda = 0.0;
    }

    else if (!theta && !phi && lambda) {
        ThetaOver2 = 0.0;
        Phi = 0.0;
        Lambda = parameters_mtx[0];
    }

    else if (!theta && phi && !lambda) {
        ThetaOver2 = 0.0;
        Phi = parameters_mtx[0];
        Lambda = 0.0;
    }

    else if (theta && !phi && !lambda) {
        ThetaOver2 = parameters_mtx[0];
        Phi = 0.0;
        Lambda = 0.0;
    }

    else {
        ThetaOver2 = 0.0;
        Phi = 0.0;
        Lambda = 0.0;
    }

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );


    apply_kernel_from_right(u3_1qbit, input);


}




/**
@brief ???????????????
*/
std::vector<Matrix> 
U3::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input ) {

    if (input.rows != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in U3 gate apply" << std::endl;
        print(sstream, 1);	   
        exit(-1);
    }


    std::vector<Matrix> ret;

    double ThetaOver2, Phi, Lambda;

    if (theta && !phi && lambda) {

        ThetaOver2 = parameters_mtx[0];
        Phi = 0.0;
        Lambda = parameters_mtx[1];        
    }

    else if (theta && phi && lambda) {
        ThetaOver2 = parameters_mtx[0];
        Phi = parameters_mtx[1];
        Lambda = parameters_mtx[2];
    }

    else if (!theta && phi && lambda) {
        ThetaOver2 = 0.0;
        Phi = parameters_mtx[0];
        Lambda = parameters_mtx[1];
    }

    else if (theta && phi && !lambda) {
        ThetaOver2 = parameters_mtx[0];
        Phi = parameters_mtx[1];
        Lambda = 0.0;
    }

    else if (!theta && !phi && lambda) {
        ThetaOver2 = 0.0;
        Phi = 0.0;
        Lambda = parameters_mtx[0];
    }

    else if (!theta && phi && !lambda) {
        ThetaOver2 = 0.0;
        Phi = parameters_mtx[0];
        Lambda = 0.0;
    }

    else if (theta && !phi && !lambda) {
        ThetaOver2 = parameters_mtx[0];
        Phi = 0.0;
        Lambda = 0.0;
    }

    else {
        ThetaOver2 = 0.0;
        Phi = 0.0;
        Lambda = 0.0;
    }



    if (theta) {

        Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2+M_PIOver2, Phi, Lambda);
        Matrix res_mtx = input.copy();
        apply_kernel_to( u3_1qbit, res_mtx );
        ret.push_back(res_mtx);

    }



    if (phi) {

        Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi+M_PIOver2, Lambda );
        memset(u3_1qbit.get_data(), 0.0, 2*sizeof(QGD_Complex16) );

        Matrix res_mtx = input.copy();
        apply_kernel_to( u3_1qbit, res_mtx );
        ret.push_back(res_mtx);

    }



    if (lambda) {

        Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda+M_PIOver2 );
        memset(u3_1qbit.get_data(), 0.0, sizeof(QGD_Complex16) );
        memset(u3_1qbit.get_data()+2, 0.0, sizeof(QGD_Complex16) );

        Matrix res_mtx = input.copy();
        apply_kernel_to( u3_1qbit, res_mtx );
        ret.push_back(res_mtx);

    }


    return ret;


}

/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
void U3::set_qbit_num(int qbit_num_in) {

        // setting the number of qubits
        Gate::set_qbit_num(qbit_num_in);
}



/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void U3::reorder_qubits( std::vector<int> qbit_list) {

    Gate::reorder_qubits(qbit_list);

}



/**
@brief Call to check whether theta is a free parameter of the gate
@return Returns with true if theta is a free parameter of the gate, or false otherwise.
*/
bool U3::is_theta_parameter() {
    return theta;
}


/**
@brief Call to check whether Phi is a free parameter of the gate
@return Returns with true if Phi is a free parameter of the gate, or false otherwise.
*/
bool U3::is_phi_parameter() {
    return phi;
}

/**
@brief Call to check whether Lambda is a free parameter of the gate
@return Returns with true if Lambda is a free parameter of the gate, or false otherwise.
*/
bool U3::is_lambda_parameter() {
    return lambda;
}



/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on a single qbit space.
@param ThetaOver2 Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
Matrix U3::calc_one_qubit_u3(double ThetaOver2, double Phi, double Lambda ) {

    Matrix u3_1qbit = Matrix(2,2);

#ifdef DEBUG
    if (isnan(ThetaOver2)) {
        std::stringstream sstream;
	sstream << "Matrix U3::calc_one_qubit_u3: ThetaOver2 is NaN." << std::endl;
        print(sstream, 1);	    
    }
    if (isnan(Phi)) {
        std::stringstream sstream;
	sstream << "Matrix U3::calc_one_qubit_u3: Phi is NaN." << std::endl;
        print(sstream, 1);	     
    }
    if (isnan(Lambda)) {
        std::stringstream sstream;
	sstream << "Matrix U3::calc_one_qubit_u3: Lambda is NaN." << std::endl;
        print(sstream, 1);	   
    }
#endif // DEBUG


    double cos_theta = theta ? cos(ThetaOver2) : 1.0;
    double sin_theta = theta ? sin(ThetaOver2) : 0.0;
    double cos_phi = phi ? cos(Phi) : 1.0;
    double sin_phi = phi ? sin(Phi) : 0.0;
    double cos_lambda = lambda ? cos(Lambda) : 1.0;
    double sin_lambda = lambda ? sin(Lambda) : 0.0;

    // the 1,1 element
    u3_1qbit[0].real = cos_theta;
    u3_1qbit[0].imag = 0;
    // the 1,2 element
    u3_1qbit[1].real = -cos_lambda*sin_theta;
    u3_1qbit[1].imag = -sin_lambda*sin_theta;
    // the 2,1 element
    u3_1qbit[2].real = cos_phi*sin_theta;
    u3_1qbit[2].imag = sin_phi*sin_theta;
    // the 2,2 element
    //cos(a+b)=cos(a)cos(b)-sin(a)sin(b)
    //sin(a+b)=sin(a)cos(b)+cos(a)sin(b)
    u3_1qbit[3].real = (cos_phi*cos_lambda-sin_phi*sin_lambda)*cos_theta;
    u3_1qbit[3].imag = (sin_phi*cos_lambda+cos_phi*sin_lambda)*cos_theta;
    //u3_1qbit[3].real = cos(Phi+Lambda)*cos_theta;
    //u3_1qbit[3].imag = sin(Phi+Lambda)*cos_theta;


    return u3_1qbit;

}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
U3* U3::clone() {

    U3* ret = new U3(qbit_num, target_qbit, theta, phi, lambda);

    if ( parameters.size() > 0 ) {
        ret->set_optimized_parameters(parameters[0], parameters[1], parameters[2]);
    }


    return ret;

}



/**
@brief Call to set the final optimized parameters of the gate.
@param ThetaOver2 Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
*/
void U3::set_optimized_parameters(double ThetaOver2, double Phi, double Lambda ) {

    parameters = Matrix_real(1, 3);

    parameters[0] = ThetaOver2;
    parameters[1] = Phi;
    parameters[2] = Lambda;

}


/**
@brief Call to get the final optimized parameters of the gate.
@param parameters_in Preallocated pointer to store the parameters ThetaOver2, Phi and Lambda of the U3 gate.
*/
Matrix_real U3::get_optimized_parameters() {

    return parameters.copy();

}



/**
@brief Call to set the parameter theta0
@param theta_in The value for the parameter theta0
*/
void U3::set_theta(double theta_in ) {

    theta0 = theta_in;

}

/**
@brief Call to set the parameter phi0
@param theta_in The value for the parameter theta0
*/
void U3::set_phi(double phi_in ) {

    phi0 = phi_in;

}


/**
@brief Call to set the parameter lambda0
@param theta_in The value for the parameter theta0
*/
void U3::set_lambda(double lambda_in ) {

    lambda0 = lambda_in;

}
