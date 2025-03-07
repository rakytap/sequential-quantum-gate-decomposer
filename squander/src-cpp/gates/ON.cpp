/*
Created on Fri JON 26 14:13:26 2020
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
/*! \file ON.cpp
    \brief Class for the representation of general ONitary operation on the first qbit_num-1 qubits.
*/


#include "ON.h"
#include "common.h"
#include "Random_Orthogonal.h"
#include "dot.h"



/**
@brief Deafult constructor of the class.
@return An instance of the class
*/
ON::ON() {

    // number of qubits spanning the matrix of the operation
    qbit_num = -1;
    // The size N of the NxN matrix associated with the operations.
    matrix_size = -1;
    // The type of the operation (see enumeration gate_type)
    type = ON_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // the number of free parameters of the operation
    parameter_num = 0;
}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the ONitaries
@return An instance of the class
*/
ON::ON(int qbit_num_in) {

    // number of qubits spanning the matrix of the operation
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the operation
    type = ON_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // The number of parameters
    parameter_num = 0;
}


/**
@brief Destructor of the class
*/
ON::~ON() {
}

/**
@brief Set the number of qubits spanning the matrix of the operation
@param qbit_num_in The number of qubits spanning the matrix
*/
void ON::set_qbit_num( int qbit_num_in ) {
    // setting the number of qubits
    qbit_num = qbit_num_in;

    // update the size of the matrix
    matrix_size = Power_of_2(qbit_num);

    // Update the number of the parameters
    parameter_num = (matrix_size/2)*(matrix_size/2-1)/2;


}



/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix of the ON gate.
@return Returns with a matrix of the operation
*/
Matrix
ON::get_matrix( Matrix_real& parameters ) {



        return get_matrix( parameters, false );
}



/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix of the ON gate.
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the operation
*/
Matrix
ON::get_matrix( Matrix_real& parameters, int parallel ) {


        Matrix ON_matrix = create_identity(matrix_size);
        apply_to(parameters, ON_matrix, parallel);

#ifdef DEBUG
        if (ON_matrix.isnan()) {
            std::stringstream sstream;
	    sstream << "U3::get_matrix: ON_matrix contains NaN." << std::endl;
            print(sstream, 1);	            
        }
#endif

        return ON_matrix;
}


/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
ON::apply_to( Matrix_real& parameters, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in ON gate apply" << std::endl;
        print(sstream, 1);	
        exit(-1);
    }

    if (parameters.size() < parameter_num) {
        std::stringstream sstream;
	sstream << "Not enough parameters given for the ON gate" << std::endl;
        print(sstream, 1);	
        exit(-1);
    }


    Matrix &&Umtx = get_submatrix( parameters );
     
    // get horizontal strided blocks of the input matrix
    Matrix Block0 = Matrix( input.get_data(), input.rows/2, input.cols, input.stride );
    Matrix Block1 = Matrix( input.get_data()+input.rows/2*input.stride, input.rows/2, input.cols, input.stride );

    // get the transformation of the blocks
    Matrix Transformed_Block0 = dot( Umtx, Block0 );
    Matrix Transformed_Block1 = dot( Umtx, Block1 );

    // put back the transformed data into input
    memcpy( input.get_data(), Transformed_Block0.get_data(), Transformed_Block0.size()*sizeof(QGD_Complex16) );
    memcpy( input.get_data()+input.rows/2*input.stride, Transformed_Block1.get_data(), Transformed_Block0.size()*sizeof(QGD_Complex16) );
    

}


/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
void 
ON::apply_from_right( Matrix_real& parameters, Matrix& input ) {


    if (input.rows != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in ON gate apply" << std::endl;
        print(sstream, 0);	        
        exit(-1);
    }

    if (parameters.size() < parameter_num) {
        std::stringstream sstream;
	sstream << "Not enough parameters given for the ON gate" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }

    Matrix &&Umtx = get_submatrix( parameters );

     
    // get vertical strided blocks of the input matrix
    Matrix Block0 = Matrix( input.get_data(), input.rows, input.cols/2, input.stride );
    Matrix Block1 = Matrix( input.get_data()+input.cols/2, input.rows, input.cols/2, input.stride );

    // get the transformation of the blocks
    Matrix Transformed_Block0 = dot( Block0, Umtx );
    Matrix Transformed_Block1 = dot( Block1, Umtx );

    // put back the transformed data into input
    for (int row_idx=0; row_idx<input.rows; row_idx++) {
        memcpy( input.get_data()+row_idx*input.stride, Transformed_Block0.get_data() + row_idx*Transformed_Block0.stride, Transformed_Block0.cols*sizeof(QGD_Complex16) );
        memcpy( input.get_data()+row_idx*input.stride + input.cols/2, Transformed_Block1.get_data() + row_idx*Transformed_Block1.stride, Transformed_Block1.cols*sizeof(QGD_Complex16) );
    }
}



/**
@brief ?????
*/
Matrix
ON::get_submatrix( Matrix_real& parameters ) {

    // create array of random parameters to construct random ONitary
    int matrix_size_loc = matrix_size/2;

    Random_Orthogonal ro( matrix_size_loc );
    Matrix Umtx = ro.Construct_Orthogonal_Matrix( parameters );

    return Umtx;

}


/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void ON::reorder_qubits( std::vector<int> qbit_list ) {


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
ON::set_optimized_parameters( Matrix_real parameters_ ) {

    parameters = parameters_.copy();

}


/**
@brief Call to get the final optimized parameters of the gate.
*/
Matrix_real ON::get_optimized_parameters() {

    return parameters.copy();

}

/**
@brief Call to get the number of free parameters
@return Return with the number of the free parameters
*/
int 
ON::get_parameter_num() {
    return parameter_num;
}


/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
gate_type ON::get_type() {
    return type;
}


/**
@brief Call to get the number of qubits composing the ONitary
@return Return with the number of qubits composing the ONitary
*/
int ON::get_qbit_num() {
    return qbit_num;
}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
ON* ON::clone() {

    ON* ret = new ON( qbit_num );

    if ( parameters.size() > 0 ) {
        ret->set_optimized_parameters( parameters );
    }
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}



