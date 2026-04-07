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
/*! \file Composite.cpp
    \brief Class for the representation of composite gate operation.
*/


#include "N_Qubit_Phase_Gate.h"
#include "common.h"
#include "dot.h"
#include "Random_Unitary.h"
#include "apply_dedicated_gate_kernel_to_input.h"

static double M_PIOver2 = M_PI/2;
/**
@brief Deafult constructor of the class.
@return An instance of the class
*/
N_Qubit_Phase_Gate::N_Qubit_Phase_Gate() {

    // A string labeling the gate operation
    name = "N_QUBIT_PHASE";

    // number of qubits spanning the matrix of the operation
    qbit_num = -1;
    // The size N of the NxN matrix associated with the operations.
    matrix_size = -1;
    // The type of the operation (see enumeration gate_type)
    type = N_QUBIT_PHASE_OPERATION;
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
N_Qubit_Phase_Gate::N_Qubit_Phase_Gate(int qbit_num_in) {

    // A string labeling the gate operation
    name = "N_QUBIT_PHASE";

    // number of qubits spanning the matrix of the operation
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the operation
    type = N_QUBIT_PHASE_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // The number of parameters
    parameter_num = matrix_size;

    for (int i = 0; i < qbit_num; i++) {
        target_qbits.push_back(i);
    }
}


/**
@brief Destructor of the class
*/
N_Qubit_Phase_Gate::~N_Qubit_Phase_Gate() {
}

/**
@brief Set the number of qubits spanning the matrix of the operation
@param qbit_num_in The number of qubits spanning the matrix
*/
void N_Qubit_Phase_Gate::set_qbit_num( int qbit_num_in ) {
    // setting the number of qubits
    qbit_num = qbit_num_in;

    // update the size of the matrix
    matrix_size = Power_of_2(qbit_num);

    // Update the number of the parameters
    parameter_num = matrix_size;


}

/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix of the composite gate.
@return Returns with a matrix of the operation
*/
Matrix
N_Qubit_Phase_Gate::get_matrix( Matrix_real& parameters ) {

        return get_matrix( parameters, false );
}

/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
N_Qubit_Phase_Gate::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& input ) {


    for ( std::vector<Matrix>::iterator it=input.begin(); it != input.end(); it++ ) {
        apply_to( parameters_mtx, *it, 0);
    }

}

/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
N_Qubit_Phase_Gate::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& inputs, int parallel ) {

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
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix of the composite gate.
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the operation
*/
Matrix
N_Qubit_Phase_Gate::get_matrix( Matrix_real& parameters, int parallel ) {

    Matrix com_matrix(matrix_size,1);
    memset(com_matrix.get_data(),0.0,(com_matrix.size()*2)*sizeof(double));
    for (int idx = 0; idx<matrix_size; idx++){
        com_matrix[idx].real = std::cos(parameters[idx]);
        com_matrix[idx].imag = std::sin(parameters[idx]);
    }
//com_matrix.print_matrix();
#ifdef DEBUG
        if (com_matrix.isnan()) {
	    std::stringstream sstream;
	    sstream << "Composite::get_matrix: UN_matrix contains NaN." << std::endl;
            print(sstream, 1);	           
        }
#endif

        return com_matrix;
}


/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
N_Qubit_Phase_Gate::apply_to( Matrix_real& parameters, Matrix& input, int parallel ) {


    if (input.rows != matrix_size ) {
        std::string err("Composite::apply_to: Wrong matrix size in Composite gate apply.");
        throw err;    
    }

    if (parameters.size() < parameter_num) {
	std::stringstream sstream;
	sstream << "Not enough parameters given for the Composite gate" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }


    Matrix com_matrix = get_matrix( parameters );
    apply_diagonal_gate_to_matrix_input(com_matrix,input,input.rows);

//std::cout << "Composite::apply_to" << std::endl;
//exit(-1);
}


/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
void 
N_Qubit_Phase_Gate::apply_from_right( Matrix_real& parameters, Matrix& input ) {


    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in Composite gate apply" << std::endl;
        print(sstream, 0);	        
        exit(-1);
    }

    if (parameters.size() < parameter_num) {
	std::stringstream sstream;
        sstream << "Not enough parameters given for the Composite gate" << std::endl;
        print(sstream, 0);	 
        exit(-1);
    }

    Matrix com_matrix = get_matrix( parameters );
    apply_diagonal_gate_to_matrix_input(com_matrix,input,input.rows);


//std::cout << "Composite::apply_to" << std::endl;
//exit(-1);

}

std::vector<Matrix> 
N_Qubit_Phase_Gate::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {

    std::vector<Matrix> ret;
    for (int idx=0;idx<parameter_num;idx++){
        Matrix res_mtx = input.copy();
        Matrix com_matrix(matrix_size,1);
        memset(com_matrix.get_data(),0.0,(com_matrix.size()*2)*sizeof(double));
        double param_shifted = parameters_mtx[idx] + M_PIOver2;
        com_matrix[idx].real = std::cos(param_shifted);
        com_matrix[idx].imag = std::sin(param_shifted);
        apply_diagonal_gate_to_matrix_input(com_matrix,res_mtx,res_mtx.rows);
        ret.push_back(res_mtx);
        com_matrix.release_data();
    }

    return ret;

}



/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void N_Qubit_Phase_Gate::reorder_qubits( std::vector<int> qbit_list ) {

    // check the number of qubits
    if ((int)qbit_list.size() != qbit_num ) {
	std::stringstream sstream;
	sstream << "Wrong number of qubits" << std::endl;
	print(sstream, 0);	    	
        exit(-1);
    }


    int control_qbit_new = control_qbit;
    int target_qbit_new = target_qbit;

    // setting the new value for the target qubit
    for (int idx=0; idx<qbit_num; idx++) {
        if (target_qbit == qbit_list[idx]) {
            target_qbit_new = qbit_num-1-idx;
        }
        if (control_qbit == qbit_list[idx]) {
            control_qbit_new = qbit_num-1-idx;
        }
    }

    control_qbit = control_qbit_new;
    target_qbit = target_qbit_new;
}

/**
@brief Call to set the final optimized parameters of the gate.
@param parameters_ Real array of the optimized parameters
*/
void 
N_Qubit_Phase_Gate::set_optimized_parameters( Matrix_real parameters_ ) {

    parameters = parameters_.copy();

}


/**
@brief Call to get the final optimized parameters of the gate.
*/
Matrix_real 
N_Qubit_Phase_Gate::get_optimized_parameters() {

    return parameters.copy();

}

/**
@brief Call to get the number of free parameters
@return Return with the number of the free parameters
*/
int 
N_Qubit_Phase_Gate::get_parameter_num() {
    return parameter_num;
}


/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
gate_type 
N_Qubit_Phase_Gate::get_type() {
    return type;
}


/**
@brief Call to get the number of qubits composing the unitary
@return Return with the number of qubits composing the unitary
*/
int 
N_Qubit_Phase_Gate::get_qbit_num() {
    return qbit_num;
}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
N_Qubit_Phase_Gate* N_Qubit_Phase_Gate::clone() {

    N_Qubit_Phase_Gate* ret = new N_Qubit_Phase_Gate( qbit_num );

    if ( parameters.size() > 0 ) {
        ret->set_optimized_parameters( parameters );
    }
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}



