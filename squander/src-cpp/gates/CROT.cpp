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
/*! \file CROT.cpp
    \brief Class representing a controlled Y rotattion gate.
*/

#include "CROT.h"
#include "apply_large_kernel_to_input.h"

#ifdef USE_AVX
#include "apply_large_kernel_to_input_AVX.h"
#endif

static double M_PIOver2 = M_PI/2;

//static tbb::spin_mutex my_mutex;
/**
@brief Nullary constructor of the class.
*/
CROT::CROT(){

        // A string describing the type of the gate
        type = CROT_OPERATION;
        
         // number of qubits spanning the matrix of the gate
        qbit_num = -1;
        // the size of the matrix
        matrix_size = -1;
        
        target_qbit = -1;
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = -1;
        parameter_num = 0;

        name = "CROT";

}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
CROT::CROT(int qbit_num_in, int target_qbit_in, int control_qbit_in) {

        name = "CROT";

        // number of qubits spanning the matrix of the gate
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the gate

        // A string describing the type of the gate
        type = CROT_OPERATION;
        



        if (control_qbit_in >= qbit_num) {
	    std::stringstream sstream;
	    sstream << "The index of the control qubit is larger than the number of qubits in CROT gate." << std::endl;
	    print(sstream, 0);	  
            throw sstream.str();
        }

        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = control_qbit_in;


        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = target_qbit_in;
        

        parameter_num=2;

        parameters = Matrix_real(1, parameter_num);
}

/**
@brief Destructor of the class
*/
CROT::~CROT() {

}

/**
@brief Call to retrieve the gate matrix.
@param parameters_mtx Parameters of the CROT gate.
@return Returns with a matrix of the gate.
*/
Matrix
CROT::get_matrix( Matrix_real& parameters_mtx ) {
    return get_matrix(parameters_mtx, 0);
}

/**
@brief Call to retrieve the gate matrix.
@param parameters_mtx Parameters of the CROT gate.
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB.
@return Returns with a matrix of the gate.
*/
Matrix
CROT::get_matrix( Matrix_real& parameters_mtx, int parallel ) {

    Matrix CROT_matrix = create_identity(matrix_size);
    apply_to(parameters_mtx, CROT_matrix, parallel);

    return CROT_matrix;
}


/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
CROT::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& input ) {


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
CROT::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& inputs, int parallel ) {

    int work_batch = 1;
    if ( parallel == 0 ) {
        work_batch = static_cast<int>(inputs.size());
    }
    else {
        work_batch = 1;
    }


    tbb::parallel_for( tbb::blocked_range<int>(0,static_cast<int>(inputs.size()),work_batch), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 

            Matrix* input = &inputs[idx];

            apply_to( parameters_mtx, *input, parallel );

        }

    });

}

/**
@brief Call to apply the gate on the input array/matrix by CROT3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
CROT::apply_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {


    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CROT gate apply" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }


    double ThetaOver2, Phi;
    

   ThetaOver2 = parameters_mtx[0];
   Phi = parameters_mtx[1];
   
    Matrix U3_matrix = calc_one_qubit_u3(ThetaOver2, Phi-M_PIOver2, -1*Phi+M_PIOver2 );
    Matrix U3_matrix2 = calc_one_qubit_u3(-1.*ThetaOver2, Phi-M_PIOver2, -1*Phi+M_PIOver2 );

    if(parallel){
#ifdef USE_AVX
      apply_crot_kernel_to_matrix_input_AVX_parallel(U3_matrix2,U3_matrix, input, target_qbit, control_qbit, input.rows);
#else
      apply_crot_kernel_to_matrix_input(U3_matrix2,U3_matrix, input, target_qbit, control_qbit, input.rows);
#endif
    }
    else{
#ifdef USE_AVX
      apply_crot_kernel_to_matrix_input_AVX(U3_matrix2,U3_matrix, input, target_qbit, control_qbit, input.rows);
#else
      apply_crot_kernel_to_matrix_input(U3_matrix2,U3_matrix, input, target_qbit, control_qbit, input.rows);
#endif
    }

    

}



/**
@brief Call to apply the gate on the input array/matrix by input*CROT
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
CROT::apply_from_right( Matrix_real& parameters, Matrix& input ) {



}


/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
*/
std::vector<Matrix> 
CROT::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CROT gate apply" << std::endl;
        print(sstream, 0);	   
        exit(-1);
    }

    std::vector<Matrix> ret;

    double ThetaOver2, Phi;


   ThetaOver2 = parameters_mtx[0];

   Phi = parameters_mtx[1];

    



    // the resulting matrix

    //Theta derivative
    Matrix res_mtx = input.copy();
    Matrix U3_matrix = calc_one_qubit_u3(ThetaOver2+M_PIOver2, Phi-M_PIOver2, -1*Phi+M_PIOver2 );
    Matrix U3_matrix2 = calc_one_qubit_u3(-1.*(ThetaOver2+M_PIOver2), Phi-M_PIOver2, -1*Phi+M_PIOver2 );

    apply_crot_kernel_to_matrix_input(U3_matrix2, U3_matrix, res_mtx, target_qbit, control_qbit, res_mtx.rows);
    ret.push_back(res_mtx);

    //Phi derivative
    Matrix res_mtx1 = input.copy();
    U3_matrix = calc_one_qubit_u3(ThetaOver2, Phi, -1*Phi );
    U3_matrix2 = calc_one_qubit_u3(-1.*ThetaOver2, Phi, -1*Phi );
    U3_matrix[0].real = 0;
    U3_matrix[0].imag = 0;
    U3_matrix[3].real = 0;
    U3_matrix[3].imag = 0;
    U3_matrix2[0].real = 0;
    U3_matrix2[0].imag = 0;
    U3_matrix2[3].real = 0;
    U3_matrix2[3].imag = 0;
    apply_crot_kernel_to_matrix_input(U3_matrix2, U3_matrix, res_mtx1, target_qbit, control_qbit, res_mtx1.rows);
    ret.push_back(res_mtx1);
    

    return ret;


}

/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
void CROT::set_qbit_num(int qbit_num_in) {

        // setting the number of qubits
        Gate::set_qbit_num(qbit_num_in);
}



/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void CROT::reorder_qubits( std::vector<int> qbit_list) {

    Gate::reorder_qubits(qbit_list);

}



/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CROT* CROT::clone() {

    CROT* ret = new CROT(qbit_num, target_qbit, control_qbit);

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
CROT::extract_parameters( Matrix_real& parameters ) {

    if ( get_parameter_start_idx() + get_parameter_num() > parameters.size()  ) {
        std::string err("CROT::extract_parameters: Cant extract parameters, since the dinput arary has not enough elements.");
        throw err;     
    }

    Matrix_real extracted_parameters(1,2);

    extracted_parameters[0] = std::fmod( 2*parameters[ get_parameter_start_idx() ], 4*M_PI);
    extracted_parameters[1] = std::fmod( parameters[ get_parameter_start_idx() + 1 ], 2*M_PI);

    return extracted_parameters;

}
