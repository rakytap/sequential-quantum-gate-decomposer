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
/*! \file U3.cpp
    \brief Class representing a U3 gate.
*/

#include "U3.h"
#include "tbb/tbb.h"

// pi/2
static double M_PIOver2 = M_PI/2;

//static tbb::spin_mutex my_mutex;
/**
@brief Nullary constructor of the class.
*/
U3::U3() {

    // A string labeling the gate operation
    name = "U3";

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

    parameter_num = 0;

}

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
*/
U3::U3(int qbit_num_in, int target_qbit_in) {
    
    // A string labeling the gate operation
    name = "U3";

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

    parameter_num = 3;

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
        
        return get_matrix( parameters, false );
}


/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the gate
*/
Matrix
U3::get_matrix( Matrix_real& parameters, int parallel ) {

        Matrix U3_matrix = create_identity(matrix_size);
        apply_to(parameters, U3_matrix, parallel);

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
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
U3::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& inputs, int parallel ) {

    int work_batch = 1;
    if ( parallel == 0 ) {
        work_batch = inputs.size();
    }
    else {
        work_batch = 1;
    }


    tbb::parallel_for( tbb::blocked_range<int>(0,inputs.size(),work_batch), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 

            Matrix* input = &inputs[idx];

            apply_to( parameters_mtx, *input, parallel );

        }

    });

}


/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
U3::apply_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {
    
    if (input.rows != matrix_size ) {
        std::string err("U3::apply_to: Wrong input size in U3 gate apply.");
        throw err;    
    }

    double ThetaOver2 = parameters_mtx[0];
    double Phi = parameters_mtx[1];
    double Lambda = parameters_mtx[2];

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );

    apply_kernel_to( u3_1qbit, input, false, parallel );
}

/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
U3::apply_from_right( Matrix_real& parameters_mtx, Matrix& input ) {
    
    if (input.cols != matrix_size ) {
        std::string err("U3::apply_from_right: Wrong matrix size in U3 apply_from_right.");
        throw err;    
    }

    double ThetaOver2 = parameters_mtx[0];
    double Phi = parameters_mtx[1];
    double Lambda = parameters_mtx[2];

    // get the U3 gate of one qubit
    Matrix u3_1qbit = calc_one_qubit_u3(ThetaOver2, Phi, Lambda );

    apply_kernel_from_right(u3_1qbit, input);
}

/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
std::vector<Matrix> 
U3::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {
    
    if (input.rows != matrix_size ) {
        std::string err("U3::apply_derivate_to: Wrong matrix size in U3 gate apply.");
        throw err;    
    }

    std::vector<Matrix> ret;

    double ThetaOver2 = parameters_mtx[0];
    double Phi = parameters_mtx[1];
    double Lambda = parameters_mtx[2];
    bool deriv = true;


    Matrix u3_1qbit_theta = calc_one_qubit_u3(ThetaOver2+M_PIOver2, Phi, Lambda);
    Matrix res_mtx_theta = input.copy();
    apply_kernel_to( u3_1qbit_theta, res_mtx_theta, deriv, parallel );
    ret.push_back(res_mtx_theta);


    Matrix u3_1qbit_phi = calc_one_qubit_u3(ThetaOver2, Phi+M_PIOver2, Lambda );
    memset(u3_1qbit_phi.get_data(), 0.0, 2*sizeof(QGD_Complex16) );
    Matrix res_mtx_phi = input.copy();
    apply_kernel_to( u3_1qbit_phi, res_mtx_phi, deriv, parallel );
    ret.push_back(res_mtx_phi);


    Matrix u3_1qbit_lambda = calc_one_qubit_u3(ThetaOver2, Phi, Lambda+M_PIOver2 );
    memset(u3_1qbit_lambda.get_data(), 0.0, sizeof(QGD_Complex16) );
    memset(u3_1qbit_lambda.get_data()+2, 0.0, sizeof(QGD_Complex16) );
    Matrix res_mtx_lambda = input.copy();
    apply_kernel_to( u3_1qbit_lambda, res_mtx_lambda, deriv, parallel );
    ret.push_back(res_mtx_lambda);


    return ret;
}

/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
void U3::set_qbit_num(int qbit_num_in) {
    Gate::set_qbit_num(qbit_num_in); // setting the number of qubits
}

/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void U3::reorder_qubits( std::vector<int> qbit_list) {
    Gate::reorder_qubits(qbit_list);
}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
U3* U3::clone() {

    U3* ret = new U3(qbit_num, target_qbit);
    
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
U3::extract_parameters( Matrix_real& parameters ) {

    if ( get_parameter_start_idx() + get_parameter_num() > parameters.size()  ) {
        std::string err("U3::extract_parameters: Cant extract parameters, since the dinput arary has not enough elements.");
        throw err;     
    }

    Matrix_real extracted_parameters(1, 3);

    extracted_parameters[0] = std::fmod( 2*parameters[ get_parameter_start_idx() ], 4*M_PI);
    extracted_parameters[1] = std::fmod( parameters[ get_parameter_start_idx()+1 ], 2*M_PI);
    extracted_parameters[2] = std::fmod( parameters[ get_parameter_start_idx()+2 ], 2*M_PI);

    return extracted_parameters;
}