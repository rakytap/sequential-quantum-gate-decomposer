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
    // the index in the parameter array (corrensponding to the encapsulated circuit) where the gate parameters begin (if gates are placed into a circuit a single parameter array is used to execute the whole circuit)
    parameter_start_idx = 0;
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
    // the index in the parameter array (corrensponding to the encapsulated circuit) where the gate parameters begin (if gates are placed into a circuit a single parameter array is used to execute the whole circuit)
    parameter_start_idx = 0;
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
@brief Call to apply the gate on a list of inputs
@param inputs The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_to_list( std::vector<Matrix>& inputs, int parallel ) {

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

            apply_to( *input, parallel );

        }

    });




}



/**
@brief Abstract function to be overriden in derived classes to be used to transform a list of inputs upon a parametric gate operation
@param parameter_mtx An array conatining the parameters of the gate
@param inputs The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& inputs, int parallel ) {

    return;

}



/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_to( Matrix& input, int parallel ) {

    Matrix ret = dot(matrix_alloc, input);
    memcpy( input.get_data(), ret.get_data(), ret.size()*sizeof(QGD_Complex16) );
    //input = ret;
}


/**
@brief Abstract function to be overriden in derived classes to be used to transform an input upon a parametric gate operation
@param parameter_mtx An array conatining the parameters
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_to( Matrix_real& parameter_mtx, Matrix& input, int parallel ) {

    return;
}



/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP (NOT IMPLEMENTED YET) and 2 for parallel with TBB (optional)
*/
std::vector<Matrix> 
Gate::apply_derivate_to( Matrix_real& parameters_mtx_in, Matrix& input, int parallel ) {

    std::vector<Matrix> ret;
    return ret;

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
        std::string err("Gate::reorder_qubits: Wrong number of qubits.");
        throw err;
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
@brief Call to get the qubits involved in the gate operation.
@return Return with a list of the involved qubits
*/
std::vector<int> Gate::get_involved_qubits() {

    std::vector<int> involved_qbits;
    
    if( target_qbit != -1 ) {
        involved_qbits.push_back( target_qbit );
    }
    
    if( control_qbit != -1 ) {
        involved_qbits.push_back( control_qbit );
    }    
    
    
    return involved_qbits;
    

}


/**
@brief Call to add a parent gate to the current gate 
@param parent The parent gate of the current gate.
*/
void Gate::add_parent( Gate* parent ) {

    // check if parent already present in th elist of parents
    if ( std::count(parents.begin(), parents.end(), parent) > 0 ) {
        return;
    }
    
    parents.push_back( parent );

}



/**
@brief Call to add a child gate to the current gate 
@param child The parent gate of the current gate.
*/
void Gate::add_child( Gate* child ) {

    // check if parent already present in th elist of parents
    if ( std::count(children.begin(), children.end(), child) > 0 ) {
        return;
    }
    
    children.push_back( child );

}


/**
@brief Call to erase data on children.
*/
void Gate::clear_children() {

    children.clear();

}


/**
@brief Call to erase data on parents.
*/
void Gate::clear_parents() {

    parents.clear();

}



/**
@brief Call to get the parents of the current gate
@return Returns with the list of the parents
*/
std::vector<Gate*> Gate::get_parents() {

    return parents;

}


/**
@brief Call to get the children of the current gate
@return Returns with the list of the children
*/
std::vector<Gate*> Gate::get_children() {

    return children;

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
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}



/**
@brief Call to apply the gate kernel on the input state or unitary with optional AVX support
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise (optional)
@param deriv Set true to apply parallel kernels, false otherwise (optional)
@param parallel Set 0 for sequential execution (default), 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_kernel_to(Matrix& u3_1qbit, Matrix& input, bool deriv, int parallel) {

#ifdef USE_AVX

    // apply kernel on state vector
    if ( input.cols == 1 && (qbit_num<14 || !parallel) ) {
        apply_kernel_to_state_vector_input_AVX(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }
    else if ( input.cols == 1 ) {
        if ( parallel == 1 ) {
            apply_kernel_to_state_vector_input_parallel_OpenMP_AVX(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        }
        else if ( parallel == 2 ) {
            apply_kernel_to_state_vector_input_parallel_AVX(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        }
        else {
            std::string err("Gate::apply_kernel_to: the argument parallel should be either 0,1 or 2. Set 0 for sequential execution (default), 1 for parallel execution with OpenMP and 2 for parallel with TBB"); 
            throw err;
        }
        return;
    }



    // unitary transform kernels
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

#ifdef _WIN32
void sincos(double x, double *s, double *c)
{
	*s = sin(x), *c = cos(x);
}
#elif defined(__APPLE__)
#define sincos __sincos
#endif

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
		
		double cos_theta = 1.0, sin_theta = 0.0;
		double cos_phi = 1.0, sin_phi = 0.0;
		double cos_lambda = 1.0, sin_lambda = 0.0;

		if (ThetaOver2!=0.0) sincos(ThetaOver2, &sin_theta, &cos_theta);
		if (Phi!=0.0) sincos(Phi, &sin_phi, &cos_phi);
		if (Lambda!=0.0) sincos(Lambda, &sin_lambda, &cos_lambda);

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


/**
@brief Call to set the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated
@param start_idx The starting index
*/
void 
Gate::set_parameter_start_idx(int start_idx) {

    parameter_start_idx = start_idx;

}

/**
@brief Call to set the parents of the current gate
@param parents_ the list of the parents
*/
void 
Gate::set_parents( std::vector<Gate*>& parents_ ) {

    parents = parents_;

}


/**
@brief Call to set the children of the current gate
@param children_ the list of the children
*/
void 
Gate::set_children( std::vector<Gate*>& children_ ) {

    children = children_;

}


/**
@brief Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated
@param start_idx The starting index
*/
int 
Gate::get_parameter_start_idx() {

    return parameter_start_idx;
    
}



/**
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is incorporated in.
@return Returns with the array of the extracted parameters.
*/
Matrix_real 
Gate::extract_parameters( Matrix_real& parameters ) {

    return Matrix_real(0,0);

}




