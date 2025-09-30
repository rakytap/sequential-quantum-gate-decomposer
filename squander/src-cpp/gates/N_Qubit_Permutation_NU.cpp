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
/*! \file UN.cpp
    \brief Class for the representation of general unitary operation on the first qbit_num-1 qubits.
*/


#include "N_Qubit_Permutation_NU.h"
#include "common.h"
#include "Random_Unitary.h"
#include "dot.h"




/**
@brief Deafult constructor of the class.
@return An instance of the class
*/
N_Qubit_Permutation_NU::N_Qubit_Permutation_NU() {

    // A string labeling the gate operation
    name = "N_Qubit_Permutation_NU";
 
    // number of qubits spanning the matrix of the operation
    qbit_num = -1;
    // The size N of the NxN matrix associated with the operations.
    matrix_size = -1;
    // The type of the operation (see enumeration gate_type)
    type = N_QUBIT_PERMUTATION_NU_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // the number of free parameters of the operation
    parameter_num = 0;
}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the unitaries
@return An instance of the class
*/
N_Qubit_Permutation_NU::N_Qubit_Permutation_NU(int qbit_num_in) {

    // A string labeling the gate operation
    name = "N_Qubit_Permutation_NU";
    // number of qubits spanning the matrix of the operation
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the operation
    type = N_QUBIT_PERMUTATION_NU_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // The number of parameters
    parameter_num = 1;
    all_patterns = construct_all_possible_patterns(qbit_num_in);
    n_perm = (int)all_patterns.size();
    centers = std::vector<double> {};
    //load up center vector 
    for (int idx=0; idx<n_perm;idx++){
        centers.push_back(idx);
    }
}


/**
@brief Destructor of the class
*/
N_Qubit_Permutation_NU::~N_Qubit_Permutation_NU() {
}


/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix of the UN gate.
@return Returns with a matrix of the operation
*/
Matrix
N_Qubit_Permutation_NU::get_matrix(Matrix_real& parameters_mtx ) {


        return get_matrix( parameters_mtx, false );
}


/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix of the UN gate.
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the operation
*/
Matrix
N_Qubit_Permutation_NU::get_matrix(Matrix_real& parameters_mtx, int parallel ) {

        Matrix P_matrix = create_identity(matrix_size);
        apply_to(parameters_mtx,P_matrix, parallel);

#ifdef DEBUG
        if (P_matrix.isnan()) {
            std::stringstream sstream;
	    sstream << "U3::get_matrix: P contains NaN." << std::endl;
            print(sstream, 1);	         
        }
#endif

        return P_matrix;
}


/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
N_Qubit_Permutation_NU::apply_to(Matrix_real& parameters_mtx, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
        std::string err("UN::apply_to: Wrong input size in UN gate apply");     
        throw(err);
    }

    Matrix NU_matrix(matrix_size,matrix_size);
    memset( NU_matrix.get_data(), 0.0, 2*NU_matrix.size()*sizeof(double) );
    double x = parameters_mtx[0];

    for (int idx=0; idx<n_perm;idx++){
        Matrix P_idx = construct_matrix_from_pattern(all_patterns[idx]);
        QGD_Complex16 factor;
        factor.imag = 0;
        factor.real = g_k(x,idx);
        mult(factor, P_idx);
        matrix_addition(NU_matrix,P_idx);
    }

    Matrix transformed_input = dot(NU_matrix,input);
    
    memcpy( input.get_data(), transformed_input.get_data(), transformed_input.size()*sizeof(QGD_Complex16) );
     
    return;
}


/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
void 
N_Qubit_Permutation_NU::apply_from_right(Matrix_real& parameters_mtx, Matrix& input ) {

    if (input.rows != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in UN gate apply" << std::endl;
        print(sstream, 0);	        
        exit(-1);
    }

    Matrix NU_matrix(matrix_size,matrix_size);
    memset( NU_matrix.get_data(), 0.0, 2*NU_matrix.size()*sizeof(double) );
    double x = parameters_mtx[0];

    for (int idx=0; idx<n_perm;idx++){
        Matrix P_idx = construct_matrix_from_pattern(all_patterns[idx]);
        QGD_Complex16 factor;
        factor.imag = 0;
        factor.real = g_k(x,idx);
        mult(factor, P_idx);
        matrix_addition(NU_matrix,P_idx);
    }

    Matrix transformed_input = dot(input,NU_matrix);
    
    memcpy( input.get_data(), transformed_input.get_data(), transformed_input.size()*sizeof(QGD_Complex16) );
    
    return;
}


/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
std::vector<Matrix> 
N_Qubit_Permutation_NU::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
        std::string err("Wrong matrix size in RZ apply_derivate_to");
        throw err;
    }

    std::vector<Matrix> ret;
    Matrix res_mtx = input.copy();
    Matrix NU_matrix(matrix_size,matrix_size);
    memset( NU_matrix.get_data(), 0.0, 2*NU_matrix.size()*sizeof(double) );
    double x = parameters_mtx[0];

    for (int idx=0; idx<n_perm;idx++){
        Matrix P_idx = construct_matrix_from_pattern(all_patterns[idx]);
        QGD_Complex16 factor;
        factor.imag = 0;
        factor.real = g_k_derivative(x,idx);
        mult(factor, P_idx);
        matrix_addition(NU_matrix,P_idx);
    }

    Matrix transformed_input = dot(NU_matrix,res_mtx);

    memcpy( res_mtx.get_data(), transformed_input.get_data(), transformed_input.size()*sizeof(QGD_Complex16) );
    ret.push_back(res_mtx);

    return ret;

}

/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void N_Qubit_Permutation_NU::reorder_qubits( std::vector<int> qbit_list ) {

    return;
}
 


/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
gate_type N_Qubit_Permutation_NU::get_type() {
    return type;
}




/**
@brief Call to get the number of qubits composing the unitary
@return Return with the number of qubits composing the unitary
*/
int N_Qubit_Permutation_NU::get_qbit_num() {
    return qbit_num;
}


std::vector<std::vector<int>> N_Qubit_Permutation_NU::construct_all_possible_patterns(int qbit_num){
    std::vector<int> initial(qbit_num);
    for (int i = 0; i < qbit_num; i++) initial[i] = i;

    std::vector<std::vector<int>> all_perms;
    do {
        all_perms.push_back(initial);
    } while (std::next_permutation(initial.begin(), initial.end()));

    return all_perms;
    
}

Matrix N_Qubit_Permutation_NU::construct_matrix_from_pattern(std::vector<int> pattern){

    Matrix perm_matrix(matrix_size,matrix_size);
    memset( perm_matrix.get_data(), 0.0, 2*perm_matrix.size()*sizeof(double) );

    for (int r = 0; r < matrix_size; r++) {
        int pr = 0;
        for (int out = 0; out < qbit_num; out++) {
            int in = pattern[out];
            int bit = (r >> in) & 1;
            pr |= (bit << out);
        }
        perm_matrix[pr*matrix_size+r].real = 1;
        }
    
    return perm_matrix;
}

// Evaluate Lagrange basis function f_k(x)
double N_Qubit_Permutation_NU::f_k(double x, int k) {
    return std::exp(-10.*(x-centers[k])*(x-centers[k]));
}

// Derivative of f_k(x) using optimized formula
double N_Qubit_Permutation_NU::f_k_derivative(double x, int k) {
    double fk = f_k(x, k);
    return fk * -20.0 * (x-centers[k]);
}

// Normalized non-negative Lagrange function g_k(x)
double N_Qubit_Permutation_NU::g_k(double x, int k) {
    double sum = 0.0;
    for (int j = 0; j < n_perm; j++) {
        double fj = f_k(x, j);
        sum += fj ;
    }
    double fk = f_k(x, k);
    return (fk) / sum;
}

// Derivative of g_k(x) using chain rule
double N_Qubit_Permutation_NU::g_k_derivative(double x, int k) {
    double fk = f_k(x, k);
    double fk_prime = f_k_derivative(x, k);

    double sum = 0.0;

    for (int j = 0; j < n_perm; j++) {
        double fj = f_k(x, j);
        sum += fj;
    }

    return (fk_prime * sum - fk * fk_prime) / (sum * sum);
}

void N_Qubit_Permutation_NU::matrix_addition(Matrix& lhs, Matrix rhs){
    for (int rdx=0; rdx<lhs.rows;rdx++){
        for (int cdx=0; cdx<lhs.cols; cdx++){
            int idx = rdx*lhs.rows+cdx;
            double real = lhs[idx].real+rhs[idx].real;
            double imag = lhs[idx].imag+rhs[idx].imag;
            lhs[idx].real = real;
            lhs[idx].imag = imag;
        }
    }
    return;
}
/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
N_Qubit_Permutation_NU* N_Qubit_Permutation_NU::clone() {

    N_Qubit_Permutation_NU* ret = new N_Qubit_Permutation_NU( qbit_num );
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}

Matrix_real 
N_Qubit_Permutation_NU::extract_parameters( Matrix_real& parameters ) {

    if ( get_parameter_start_idx() + get_parameter_num() > parameters.size()  ) {
        std::string err("RY::extract_parameters: Cant extract parameters, since the dinput arary has not enough elements.");
        throw err;     
    }

    Matrix_real extracted_parameters(1,1);

    extracted_parameters[0] = parameters[ get_parameter_start_idx() ];

    return extracted_parameters;

}


