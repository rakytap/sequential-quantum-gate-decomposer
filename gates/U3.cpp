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

	//The stringstream input to store the output messages.
	std::stringstream sstream;

	//Integer value to set the verbosity level of the output messages.
	int verbose_level;

        // number of qubits spanning the matrix of the gate
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the gate
        type = U3_OPERATION;


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


	//The stringstream input to store the output messages.
	std::stringstream sstream;

	//Integer value to set the verbosity level of the output messages.
	int verbose_level;

        Matrix U3_matrix = create_identity(matrix_size);
        apply_to(parameters, U3_matrix);

#ifdef DEBUG
        if (U3_matrix.isnan()) {
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
U3::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& input, const double scale=1.0 ) {


    for ( std::vector<Matrix>::iterator it=input.begin(); it != input.end(); it++ ) {
        apply_to( parameters_mtx, *it, scale );
    }

}



/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
U3::apply_to( Matrix_real& parameters_mtx, Matrix& input, const double scale=1.0 ) {

	//The stringstream input to store the output messages.
	std::stringstream sstream;

	//Integer value to set the verbosity level of the output messages.
	int verbose_level;

    if (input.rows != matrix_size ) {
	sstream << "Wrong matrix size in U3 gate apply" << std::endl;
	verbose_level=1;
        print(sstream,verbose_level);	
	
        
        exit(-1);
    }


    double Theta, Phi, Lambda;

    if (theta && !phi && lambda) {
        Theta = parameters_mtx[0];
        Phi = 0.0;
        Lambda = parameters_mtx[1];
    }

    else if (theta && phi && lambda) {
        Theta = parameters_mtx[0];
        Phi = parameters_mtx[1];
        Lambda = parameters_mtx[2];
    }

    else if (!theta && phi && lambda) {
        Theta = 0.0;
        Phi = parameters_mtx[0];
        Lambda = parameters_mtx[1];
    }

    else if (theta && phi && !lambda) {
        Theta = parameters_mtx[0];
        Phi = parameters_mtx[1];
        Lambda = 0.0;
    }

    else if (!theta && !phi && lambda) {
        Theta = 0.0;
        Phi = 0.0;
        Lambda = parameters_mtx[0];
    }

    else if (!theta && phi && !lambda) {
        Theta = 0.0;
        Phi = parameters_mtx[0];
        Lambda = 0.0;
    }

    else if (theta && !phi && !lambda) {
        Theta = parameters_mtx[0];
        Phi = 0.0;
        Lambda = 0.0;
    }

    else {
        Theta = 0.0;
        Phi = 0.0;
        Lambda = 0.0;
    }


    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(Theta, Phi, Lambda, scale );


    apply_kernel_to( u3_1qbit, input );


}



/**
@brief ???????????
*/
void 
U3::apply_kernel_to( Matrix& u3_1qbit, Matrix& input ) {


    int index_step = Power_of_2(target_qbit);
    int current_idx = 0;
    int current_idx_pair = current_idx+index_step;

//std::cout << "target qbit: " << target_qbit << std::endl;


    while ( current_idx_pair < matrix_size ) {


        //tbb::parallel_for(0, index_step, 1, [&](int idx) {  
        for( int idx=0; idx<index_step; idx++ )  {

            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            int row_offset = current_idx_loc*input.stride;
            int row_offset_pair = current_idx_pair_loc*input.stride;

            for ( int col_idx=0; col_idx<matrix_size; col_idx++) {
                int index      = row_offset+col_idx;
                int index_pair = row_offset_pair+col_idx;

                QGD_Complex16 element      = input[index];
                QGD_Complex16 element_pair = input[index_pair];

                QGD_Complex16 tmp1 = mult(u3_1qbit[0], element);
                QGD_Complex16 tmp2 = mult(u3_1qbit[1], element_pair);
                input[index].real = tmp1.real + tmp2.real;
                input[index].imag = tmp1.imag + tmp2.imag;

                tmp1 = mult(u3_1qbit[2], element);
                tmp2 = mult(u3_1qbit[3], element_pair);
                input[index_pair].real = tmp1.real + tmp2.real;
                input[index_pair].imag = tmp1.imag + tmp2.imag;

            };         

//std::cout << current_idx << " " << current_idx_pair << std::endl;
        }
        //});


        current_idx = current_idx + 2*index_step;
        current_idx_pair = current_idx_pair + 2*index_step;


    }

}




/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
U3::apply_from_right( Matrix_real& parameters_mtx, Matrix& input ) {

	//The stringstream input to store the output messages.
	std::stringstream sstream;

	//Integer value to set the verbosity level of the output messages.
	int verbose_level;

    if (input.cols != matrix_size ) {
	sstream << "Wrong matrix size in U3 apply_from_right" << std::endl;
	verbose_level=1;
        print(sstream,verbose_level);	
	
        
        exit(-1);
    }

    double Theta, Phi, Lambda;

    if (theta && !phi && lambda) {
        Theta = parameters_mtx[0];
        Phi = 0.0;
        Lambda = parameters_mtx[1];
    }

    else if (theta && phi && lambda) {
        Theta = parameters_mtx[0];
        Phi = parameters_mtx[1];
        Lambda = parameters_mtx[2];
    }

    else if (!theta && phi && lambda) {
        Theta = 0.0;
        Phi = parameters_mtx[0];
        Lambda = parameters_mtx[1];
    }

    else if (theta && phi && !lambda) {
        Theta = parameters_mtx[0];
        Phi = parameters_mtx[1];
        Lambda = 0.0;
    }

    else if (!theta && !phi && lambda) {
        Theta = 0.0;
        Phi = 0.0;
        Lambda = parameters_mtx[0];
    }

    else if (!theta && phi && !lambda) {
        Theta = 0.0;
        Phi = parameters_mtx[0];
        Lambda = 0.0;
    }

    else if (theta && !phi && !lambda) {
        Theta = parameters_mtx[0];
        Phi = 0.0;
        Lambda = 0.0;
    }

    else {
        Theta = 0.0;
        Phi = 0.0;
        Lambda = 0.0;
    }

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(Theta, Phi, Lambda );


    apply_kernel_from_right(u3_1qbit, input);


}


/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
U3::apply_kernel_from_right( Matrix& u3_1qbit, Matrix& input ) {

    int index_step = Power_of_2(target_qbit);
    int current_idx = 0;
    int current_idx_pair = current_idx+index_step;


    while ( current_idx_pair < matrix_size ) {


        //tbb::parallel_for(0, index_step, 1, [&](int idx) {  
        for( int idx=0; idx<index_step; idx++ )  {

            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;


            for ( int row_idx=0; row_idx<matrix_size; row_idx++) {

                int row_offset = row_idx*input.stride;


                int index      = row_offset+current_idx_loc;
                int index_pair = row_offset+current_idx_pair_loc;

                QGD_Complex16 element      = input[index];
                QGD_Complex16 element_pair = input[index_pair];

                QGD_Complex16 tmp1 = mult(u3_1qbit[0], element);
                QGD_Complex16 tmp2 = mult(u3_1qbit[2], element_pair);
                input[index].real = tmp1.real + tmp2.real;
                input[index].imag = tmp1.imag + tmp2.imag;

                tmp1 = mult(u3_1qbit[1], element);
                tmp2 = mult(u3_1qbit[3], element_pair);
                input[index_pair].real = tmp1.real + tmp2.real;
                input[index_pair].imag = tmp1.imag + tmp2.imag;

            };         

//std::cout << current_idx << " " << current_idx_pair << std::endl;
        }
        //});


        current_idx = current_idx + 2*index_step;
        current_idx_pair = current_idx_pair + 2*index_step;


    }

}


/**
@brief ???????????????
*/
std::vector<Matrix> 
U3::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input ) {

	//The stringstream input to store the output messages.
	std::stringstream sstream;

	//Integer value to set the verbosity level of the output messages.
	int verbose_level;

    if (input.rows != matrix_size ) {
	sstream << "Wrong matrix size in U3 gate apply" << std::endl;
	verbose_level=1;
        print(sstream,verbose_level);	
	
        
        exit(-1);
    }


    std::vector<Matrix> ret;

    double Theta, Phi, Lambda;

    if (theta && !phi && lambda) {

        Theta = parameters_mtx[0];
        Phi = 0.0;
        Lambda = parameters_mtx[1];        
    }

    else if (theta && phi && lambda) {
        Theta = parameters_mtx[0];
        Phi = parameters_mtx[1];
        Lambda = parameters_mtx[2];
    }

    else if (!theta && phi && lambda) {
        Theta = 0.0;
        Phi = parameters_mtx[0];
        Lambda = parameters_mtx[1];
    }

    else if (theta && phi && !lambda) {
        Theta = parameters_mtx[0];
        Phi = parameters_mtx[1];
        Lambda = 0.0;
    }

    else if (!theta && !phi && lambda) {
        Theta = 0.0;
        Phi = 0.0;
        Lambda = parameters_mtx[0];
    }

    else if (!theta && phi && !lambda) {
        Theta = 0.0;
        Phi = parameters_mtx[0];
        Lambda = 0.0;
    }

    else if (theta && !phi && !lambda) {
        Theta = parameters_mtx[0];
        Phi = 0.0;
        Lambda = 0.0;
    }

    else {
        Theta = 0.0;
        Phi = 0.0;
        Lambda = 0.0;
    }



    if (theta) {

        Matrix u3_1qbit = calc_one_qubit_u3(Theta+M_PI, Phi, Lambda, 0.5);
        Matrix res_mtx = input.copy();
        apply_kernel_to( u3_1qbit, res_mtx );
        ret.push_back(res_mtx);

    }



    if (phi) {

        Matrix u3_1qbit = calc_one_qubit_u3(Theta, Phi+M_PI/2, Lambda, 1.0 );
        memset(u3_1qbit.get_data(), 0.0, 2*sizeof(QGD_Complex16) );

        Matrix res_mtx = input.copy();
        apply_kernel_to( u3_1qbit, res_mtx );
        ret.push_back(res_mtx);

    }



    if (lambda) {

        Matrix u3_1qbit = calc_one_qubit_u3(Theta, Phi, Lambda+M_PI/2, 1.0 );
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
@param Theta Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
Matrix U3::calc_one_qubit_u3(double Theta, double Phi, double Lambda, const double scale=1.0 ) {

	//The stringstream input to store the output messages.
	std::stringstream sstream;

	//Integer value to set the verbosity level of the output messages.
	int verbose_level;

    Matrix u3_1qbit = Matrix(2,2);

#ifdef DEBUG
    if (isnan(Theta)) {
	sstream << "Matrix U3::calc_one_qubit_u3: Theta is NaN." << std::endl;
	verbose_level=1;
        print(sstream,verbose_level);	
	
        
    }
    if (isnan(Phi)) {
	sstream << "Matrix U3::calc_one_qubit_u3: Phi is NaN." << std::endl;
	verbose_level=1;
        print(sstream,verbose_level);	
	
        
    }
    if (isnan(Lambda)) {
	sstream << "Matrix U3::calc_one_qubit_u3: Lambda is NaN." << std::endl;
	verbose_level=1;
        print(sstream,verbose_level);	
	
        
    }
#endif // DEBUG


    double cos_theta = cos(Theta/2)*scale;
    double sin_theta = sin(Theta/2)*scale;


    // the 1,1 element
    u3_1qbit[0].real = cos_theta;
    u3_1qbit[0].imag = 0;
    // the 1,2 element
    u3_1qbit[1].real = -cos(Lambda)*sin_theta;
    u3_1qbit[1].imag = -sin(Lambda)*sin_theta;
    // the 2,1 element
    u3_1qbit[2].real = cos(Phi)*sin_theta;
    u3_1qbit[2].imag = sin(Phi)*sin_theta;
    // the 2,2 element
    u3_1qbit[3].real = cos(Phi+Lambda)*cos_theta;
    u3_1qbit[3].imag = sin(Phi+Lambda)*cos_theta;


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
@param Theta Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
*/
void U3::set_optimized_parameters(double Theta, double Phi, double Lambda ) {

    parameters = Matrix_real(1, 3);

    parameters[0] = Theta;
    parameters[1] = Phi;
    parameters[2] = Lambda;

}


/**
@brief Call to get the final optimized parameters of the gate.
@param parameters_in Preallocated pointer to store the parameters Theta, Phi and Lambda of the U3 gate.
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
