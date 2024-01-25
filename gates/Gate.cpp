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
/*! \file Gate.cpp
    \brief Class for the representation of general gate operations.
*/


#include "Gate.h"
#include "common.h"

#ifdef USE_AVX 
#include "apply_kernel_to_input_AVX.h"
#include "apply_kernel_to_state_vector_input_AVX.h"
#endif

#include "apply_kernel_to_input.h"
#include "apply_kernel_to_state_vector_input.h"

/**
@brief Deafult constructor of the class.
@return An instance of the class
*/
Gate::Gate() {

    // number of qubits spanning the matrix of the operation
    qbit_num = -1;
    // The size N of the NxN matrix associated with the operations.
    matrix_size = -1;
    // The type of the operation (see enumeration gate_type)
    type = GENERAL_OPERATION;
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
Gate::Gate(int qbit_num_in) {

    if (qbit_num_in > 30) {
        std::string err("Gate::Gate: Number of qubits supported up to 30"); 
        throw err;        
    }

    // number of qubits spanning the matrix of the operation
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the operation
    type = GENERAL_OPERATION;
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
Gate::~Gate() {
}

/**
@brief Set the number of qubits spanning the matrix of the operation
@param qbit_num_in The number of qubits spanning the matrix
*/
void Gate::set_qbit_num( int qbit_num_in ) {

    if (qbit_num_in > 30) {
        std::string err("Gate::set_qbit_num: Number of qubits supported up to 30"); 
        throw err;        
    }


    // setting the number of qubits
    qbit_num = qbit_num_in;

    // update the size of the matrix
    matrix_size = Power_of_2(qbit_num);

}


/**
@brief Call to retrieve the operation matrix
@return Returns with a matrix of the operation
*/
Matrix
Gate::get_matrix() {

    return matrix_alloc;
}

/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
Gate::apply_to_list( std::vector<Matrix>& input ) {


    for ( std::vector<Matrix>::iterator it=input.begin(); it != input.end(); it++ ) {
        apply_to( *it );
    }

}


/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set true to apply parallel kernels, false otherwise (optional)
*/
void 
Gate::apply_to( Matrix& input, bool parallel ) {

    Matrix ret = dot(matrix_alloc, input);
    memcpy( input.get_data(), ret.get_data(), ret.size()*sizeof(QGD_Complex16) );
    //input = ret;
}


/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
void 
Gate::apply_from_right( Matrix& input ) {

    Matrix ret = dot(input, matrix_alloc);
    memcpy( input.get_data(), ret.get_data(), ret.size()*sizeof(QGD_Complex16) );

}


/**
@brief Call to set the stored matrix in the operation.
@param input The operation matrix to be stored. The matrix is stored by attribute matrix_alloc.
@return Returns with 0 on success.
*/
void
Gate::set_matrix( Matrix input ) {
    matrix_alloc = input;
}


/**
@brief Call to set the control qubit for the gate operation
@param control_qbit_in The control qubit. Should be: 0 <= control_qbit_in < qbit_num
*/
void Gate::set_control_qbit(int control_qbit_in){
    control_qbit = control_qbit_in;
}


/**
@brief Call to set the target qubit for the gate operation
@param target_qbit_in The target qubit on which the gate is applied. Should be: 0 <= target_qbit_in < qbit_num
*/
void Gate::set_target_qbit(int target_qbit_in){
    target_qbit = target_qbit_in;
}

/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void Gate::reorder_qubits( std::vector<int> qbit_list ) {

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
@brief Call to get the index of the target qubit
@return Return with the index of the target qubit (return with -1 if target qubit was not set)
*/
int Gate::get_target_qbit() {
    return target_qbit;
}

/**
@brief Call to get the index of the control qubit
@return Return with the index of the control qubit (return with -1 if control qubit was not set)
*/
int Gate::get_control_qbit()  {
    return control_qbit;
}

/**
@brief Call to get the number of free parameters
@return Return with the number of the free parameters
*/
int Gate::get_parameter_num() {
    return parameter_num;
}


/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
gate_type Gate::get_type() {
    return type;
}


/**
@brief Call to get the number of qubits composing the unitary
@return Return with the number of qubits composing the unitary
*/
int Gate::get_qbit_num() {
    return qbit_num;
}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
Gate* Gate::clone() {

    Gate* ret = new Gate( qbit_num );
    ret->set_matrix( matrix_alloc );

    return ret;

}



/**
@brief Call to apply the gate kernel on the input state or unitary with optional AVX support
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise (optional)
@param deriv Set true to apply parallel kernels, false otherwise (optional)
*/
void 
Gate::apply_kernel_to(Matrix& u3_1qbit, Matrix& input, bool deriv, bool parallel) {

#ifdef USE_AVX

    // apply kernel on state vector
    if ( input.cols == 1 && (qbit_num<10 || !parallel) ) {
        apply_kernel_to_state_vector_input_AVX(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }
    else if ( input.cols == 1 ) {
        apply_kernel_to_state_vector_input_parallel_AVX(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }




    if ( qbit_num < 4 ) {
        apply_kernel_to_input_AVX_small(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }
    else if ( qbit_num < 10 || !parallel) {
        apply_kernel_to_input_AVX(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
     }
    else {
        apply_kernel_to_input_AVX_parallel(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
     }


#else

    // apply kernel on state vector
    if ( input.cols == 1 && (qbit_num<10 || !parallel) ) {
        apply_kernel_to_state_vector_input(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }
    else if ( input.cols == 1 ) {
        apply_kernel_to_state_vector_input_parallel(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }


    // apply kernel on unitary
    apply_kernel_to_input(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size); 


   


#endif // USE_AVX


}





/**
@brief Call to apply the gate kernel on the input state or unitary from right (no AVX support)
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
*/
void 
Gate::apply_kernel_from_right( Matrix& u3_1qbit, Matrix& input ) {

   
    int index_step_target = 1 << target_qbit;
    int current_idx = 0;
    int current_idx_pair = current_idx+index_step_target;

//std::cout << "target qbit: " << target_qbit << std::endl;

    while ( current_idx_pair < input.cols ) {

        for(int idx=0; idx<index_step_target; idx++) { 
        //tbb::parallel_for(0, index_step_target, 1, [&](int idx) {  

            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            // determine the action according to the state of the control qubit
            if ( control_qbit<0 || ((current_idx_loc >> control_qbit) & 1) ) {

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

                }

            }
            else {
                // leave the state as it is
                continue;
            }        


//std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;


        //});
        }


        current_idx = current_idx + (index_step_target << 1);
        current_idx_pair = current_idx_pair + (index_step_target << 1);


    }


}

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on a single qbit space.
@param ThetaOver2 Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
Matrix Gate::calc_one_qubit_u3(double ThetaOver2, double Phi, double Lambda ) {

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


        double cos_theta = cos(ThetaOver2);
        double sin_theta = sin(ThetaOver2);
        double cos_phi = cos(Phi);
        double sin_phi = sin(Phi);
        double cos_lambda = cos(Lambda);
        double sin_lambda = sin(Lambda);

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
@brief Calculate the matrix of the constans gates.
@return Returns with the matrix of the one-qubit matrix.
*/
Matrix Gate::calc_one_qubit_u3( ) {

  Matrix u3_1qbit = Matrix(2,2); 
  return u3_1qbit;

}

/**
@brief Set static values for the angles and constans parameters for calculating the matrix of the gates.
@param ThetaOver2 Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
*/
void
Gate::parameters_for_calc_one_qubit(double& ThetaOver2, double& Phi, double& Lambda  ) {

 return;

}




