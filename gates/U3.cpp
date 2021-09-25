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
        // the base indices of the target qubit for state |0>
        indexes_target_qubit_0 = NULL;
        // the base indices of the target qubit for state |1>
        indexes_target_qubit_1 = NULL;

        // logical value indicating whether the matrix creation takes an argument theta
        theta = false;
        // logical value indicating whether the matrix creation takes an argument phi
        phi = false;
        // logical value indicating whether the matrix creation takes an argument lambda
        lambda = false;

        parameter_num = 0;

        // Parameters theta, phi, lambda of the U3 gate after the decomposition of the unitary is done
        parameters = NULL;


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
            printf("The index of the target qubit is larger than the number of qubits");
            throw "The index of the target qubit is larger than the number of qubits";
        }
        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = target_qbit_in;
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = -1;
        // the base indices of the target qubit for state |0>
        indexes_target_qubit_0 = NULL;
        // the base indices of the target qubit for state |1>
        indexes_target_qubit_1 = NULL;

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
        parameters = NULL;

        // determine the basis indices of the |0> and |1> states of the target qubit
        determine_base_indices();
}


/**
@brief Destructor of the class
*/
U3::~U3() {

    if ( indexes_target_qubit_0 != NULL ) {
        qgd_free(indexes_target_qubit_0);
        indexes_target_qubit_0 = NULL;
    }

    if ( indexes_target_qubit_1 != NULL ) {
        qgd_free(indexes_target_qubit_1);
        indexes_target_qubit_1 = NULL;
    }

    if ( parameters != NULL ) {
        qgd_free(parameters);
        parameters = NULL;
    }

}



/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@return Returns with a matrix of the gate
*/
Matrix
U3::get_matrix( const double* parameters ) {
/*
        // preallocate array for the composite u3 gate
        Matrix U3_matrix = Matrix(matrix_size, matrix_size);

        if (theta && !phi && lambda) {
            // function handle to calculate the gate on the target qubit
            composite_u3_Theta_Lambda( parameters, U3_matrix );
        }

        else if (theta && phi && lambda) {
            // function handle to calculate the gate on the target qubit
            composite_u3_Theta_Phi_Lambda( parameters, U3_matrix );
        }

        else if (!theta && phi && lambda) {
            // function handle to calculate the gate on the target qubit
            composite_u3_Phi_Lambda( parameters, U3_matrix );
        }

        else if (theta && phi && !lambda) {
            // function handle to calculate the gate on the target qubit
            composite_u3_Theta_Phi( parameters, U3_matrix );
        }

        else if (!theta && !phi && lambda) {
            // function handle to calculate the gate on the target qubit
            composite_u3_Lambda( parameters, U3_matrix );
        }

        else if (!theta && phi && !lambda) {
            // function handle to calculate the gate on the target qubit
            composite_u3_Phi( parameters, U3_matrix );
        }

        else if (theta && !phi && !lambda) {
            // function handle to calculate the gate on the target qubit
            composite_u3_Theta( parameters, U3_matrix );
        }

        else {
            composite_u3(0, 0, 0, U3_matrix );
        }
*/


        Matrix U3_matrix = create_identity(matrix_size);
        apply_to(parameters, U3_matrix);

#ifdef DEBUG
        if (U3_matrix.isnan()) {
            std::cout << "U3::get_matrix: U3_matrix contains NaN." << std::endl;
        }
#endif






        return U3_matrix;

}


/**
@brief Call to apply the gate on the input array/matrix
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
U3::apply_to( const double* parameters, Matrix& input ) {

    if (input.rows != matrix_size ) {
        std::cout<< "Wrong matrix size in U3 gate apply" << std::endl;
        exit(-1);
    }


    double Theta, Phi, Lambda;

    if (theta && !phi && lambda) {
        Theta = parameters[0];
        Phi = 0.0;
        Lambda = parameters[1];
    }

    else if (theta && phi && lambda) {
        Theta = parameters[0];
        Phi = parameters[1];
        Lambda = parameters[2];
    }

    else if (!theta && phi && lambda) {
        Theta = 0.0;
        Phi = parameters[0];
        Lambda = parameters[1];
    }

    else if (theta && phi && !lambda) {
        Theta = parameters[0];
        Phi = parameters[1];
        Lambda = 0.0;
    }

    else if (!theta && !phi && lambda) {
        Theta = 0.0;
        Phi = 0.0;
        Lambda = parameters[0];
    }

    else if (!theta && phi && !lambda) {
        Theta = 0.0;
        Phi = parameters[0];
        Lambda = 0.0;
    }

    else if (theta && !phi && !lambda) {
        Theta = parameters[0];
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


    int index_step = Power_of_2(target_qbit);
    int current_idx = 0;
    int current_idx_pair = current_idx+index_step;

//std::cout << "target qbit: " << target_qbit << std::endl;

    while ( current_idx_pair < matrix_size ) {


        tbb::parallel_for(0, index_step, 1, [&](int idx) {  

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

        });


        current_idx = current_idx + 2*index_step;
        current_idx_pair = current_idx_pair + 2*index_step;


    }


 


}

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on the space of qbit_num qubits.
@param parameters An array containing the parameters of the U3 gate.
@param U3_matrix A pointer to the preallocated array of the gate matrix.
@return Returns with 0 on success.
*/
int U3::composite_u3_Theta_Phi_Lambda( const double* parameters, Matrix& U3_matrix ) {
        return composite_u3( parameters[0], parameters[1], parameters[2], U3_matrix );
}

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on the space of qbit_num qubits.
@param parameters An array containing the parameters of the U3 gate.
@param U3_matrix A pointer to the preallocated array of the gate matrix.
@return Returns with 0 on success.
*/
int U3::composite_u3_Phi_Lambda( const double* parameters, Matrix& U3_matrix ) {
        return composite_u3( 0, parameters[0], parameters[1], U3_matrix );
}

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on the space of qbit_num qubits.
@param parameters An array containing the parameters of the U3 gate.
@param U3_matrix A pointer to the preallocated array of the gate matrix.
@return Returns with 0 on success.
*/
int U3::composite_u3_Theta_Lambda( const double* parameters, Matrix& U3_matrix ) {
        return composite_u3( parameters[0], 0, parameters[1], U3_matrix );
}

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on the space of qbit_num qubits.
@param parameters An array containing the parameters of the U3 gate.
@param U3_matrix A pointer to the preallocated array of the gate matrix.
@return Returns with 0 on success.
*/
int U3::composite_u3_Theta_Phi( const double* parameters, Matrix& U3_matrix ){
        return composite_u3( parameters[0], parameters[1], 0, U3_matrix );
}

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on the space of qbit_num qubits.
@param parameters An array containing the parameters of the U3 gate.
@param U3_matrix A pointer to the preallocated array of the gate matrix.
@return Returns with 0 on success.
*/
int U3::composite_u3_Lambda( const double* parameters, Matrix& U3_matrix ) {
        return composite_u3( 0, 0, parameters[0], U3_matrix );
}

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on the space of qbit_num qubits.
@param parameters An array containing the parameters of the U3 gate.
@param U3_matrix A pointer to the preallocated array of the gate matrix.
@return Returns with 0 on success.
*/
int U3::composite_u3_Phi( const double* parameters, Matrix& U3_matrix ) {
        return composite_u3( 0, parameters[0], 0, U3_matrix );
}

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on the space of qbit_num qubits.
@param parameters An array containing the parameters of the U3 gate.
@param U3_matrix A pointer to the preallocated array of the gate matrix.
@return Returns with 0 on success.
*/
int U3::composite_u3_Theta( const double* parameters, Matrix& U3_matrix ) {
        return composite_u3( parameters[0], 0, 0, U3_matrix );
}


/**
@brief Calculate the matrix of a U3 gate gate corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
@param Theta Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
@param U3_matrix A pointer to the preallocated array of the gate matrix.
@return Returns with 0 on success.
*/
int U3::composite_u3(double Theta, double Phi, double Lambda, Matrix& U3_matrix ) {

        // set to zero all the elements of the matrix
        memset(U3_matrix.get_data(), 0, U3_matrix.size()*sizeof(QGD_Complex16) );

        // get the U3 gate of one qubit
        Matrix u3_1qbit = calc_one_qubit_u3(Theta, Phi, Lambda );

#ifdef DEBUG
        if (u3_1qbit.isnan()) {
            std::cout << "U3::composite_u3: u3_1qbit contains NaN." << std::endl;
        }
#endif

        // setting the gate elements
        for(int idx = 0; idx < matrix_size/2; ++idx)
        {
            int element_index;

            // element |0> -> |0>
            element_index = indexes_target_qubit_0[idx]*matrix_size + indexes_target_qubit_0[idx];

            U3_matrix[element_index] = u3_1qbit[0];

            // element |1> -> |0>
            element_index = indexes_target_qubit_0[idx]*matrix_size + indexes_target_qubit_1[idx];
            U3_matrix[element_index] = u3_1qbit[1];

            // element |0> -> |1>
            element_index = indexes_target_qubit_1[idx]*matrix_size + indexes_target_qubit_0[idx];
            U3_matrix[element_index] = u3_1qbit[2];

            // element |1> -> |1>
            element_index = indexes_target_qubit_1[idx]*matrix_size + indexes_target_qubit_1[idx];
            U3_matrix[element_index] = u3_1qbit[3];


        }


        return 0;
}


/**
@brief Determine the base indices corresponding to the target qubit states |0> and |1>
*/
void U3::determine_base_indices() {


        // free the previously allocated memories
        if ( indexes_target_qubit_1 != NULL ) {
            qgd_free( indexes_target_qubit_1 );
            indexes_target_qubit_1 = NULL;
        }
        if ( indexes_target_qubit_0 != NULL ) {
            qgd_free( indexes_target_qubit_0 );
            indexes_target_qubit_0 = NULL;
        }


        indexes_target_qubit_1 = (int*)qgd_calloc(matrix_size/2,sizeof(int), 64);
        indexes_target_qubit_0 = (int*)qgd_calloc(matrix_size/2,sizeof(int), 64);



        int target_qbit_power = Power_of_2(target_qbit);
        int indexes_target_qubit_1_idx = 0;
        int indexes_target_qubit_0_idx = 0;

        // generate the reordered  basis set
        for(int idx = 0; idx<matrix_size; ++idx)
        {
            int bit = int(idx/target_qbit_power) % 2;
            if (bit == 0) {
                indexes_target_qubit_0[indexes_target_qubit_0_idx] = idx;
                indexes_target_qubit_0_idx++;
            }
            else {
                indexes_target_qubit_1[indexes_target_qubit_1_idx] = idx;
                indexes_target_qubit_1_idx++;
            }
        }

/*
        // print the result
        printf("\n");
        printf("qbit num: %d, target qubit: %d\n", qbit_num, target_qbit);
        for(int idx = 0; idx<matrix_size/2; ++idx) {
            printf ("base indexes for target bit state 0 and 1 are: %d and %d \n", indexes_target_qubit_0[idx], indexes_target_qubit_1[idx]);
        }
*/
}

/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
void U3::set_qbit_num(int qbit_num_in) {
        // setting the number of qubits
        Gate::set_qbit_num(qbit_num_in);

        // get the base indices of the target qubit
        determine_base_indices();

}



/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void U3::reorder_qubits( std::vector<int> qbit_list) {

    Gate::reorder_qubits(qbit_list);

    // get the base indices of the target qubit
    determine_base_indices();

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
Matrix U3::calc_one_qubit_u3(double Theta, double Phi, double Lambda ) {

    Matrix u3_1qbit = Matrix(2,2);

#ifdef DEBUG
    if (isnan(Theta)) {
        std::cout << "Matrix U3::calc_one_qubit_u3: Theta is NaN." << std::endl;
    }
    if (isnan(Phi)) {
        std::cout << "Matrix U3::calc_one_qubit_u3: Phi is NaN." << std::endl;
    }
    if (isnan(Lambda)) {
        std::cout << "Matrix U3::calc_one_qubit_u3: Lambda is NaN." << std::endl;
    }
#endif // DEBUG


    double cos_theta = cos(Theta/2);
    double sin_theta = sin(Theta/2);


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

    if ( parameters != NULL ) {
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

    if ( parameters == NULL ) {
        parameters = (double*)qgd_calloc( 3, sizeof(double), 16 );
    }

    memset( parameters, 0, 3*sizeof(double) );

    parameters[0] = Theta;
    parameters[1] = Phi;
    parameters[2] = Lambda;

}


/**
@brief Call to get the final optimized parameters of the gate.
@param parameters_in Preallocated pointer to store the parameters Theta, Phi and Lambda of the U3 gate.
*/
void U3::get_optimized_parameters(double *parameters_in ) {

    memcpy( parameters_in, parameters, 3*sizeof(double) );

}
