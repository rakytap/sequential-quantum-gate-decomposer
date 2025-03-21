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
/*! \file CZ_NU.cpp
    \brief Class representing a non-unitary, parametric CZ gate.
*/

#include "CZ_NU.h"



using namespace std;


/**
@brief Nullary constructor of the class.
*/
CZ_NU::CZ_NU() {

        // number of qubits spanning the matrix of the gate
        qbit_num = -1;
        // the size of the matrix
        matrix_size = -1;
        // A string describing the type of the gate
        type = CZ_NU_OPERATION;
        // The number of free parameters
        parameter_num = 0;

        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = -1;

        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = -1;


}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
CZ_NU::CZ_NU(int qbit_num_in,  int target_qbit_in, int control_qbit_in) {


        // number of qubits spanning the matrix of the gate
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the gate
        type = CZ_NU_OPERATION;
        // The number of free parameters
        parameter_num = 1;

        if (target_qbit_in >= qbit_num) {
            std::stringstream sstream;
	    sstream << "The index of the target qubit is larger than the number of qubits" << std::endl;
	    print(sstream, 0);	    	            
            throw sstream.str();
        }
        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = target_qbit_in;


        if (control_qbit_in >= qbit_num) {
            std::stringstream sstream;
	    sstream << "The index of the control qubit is larger than the number of qubits" << std::endl;
	    print(sstream, 0);	    	
            throw sstream.str();
        }
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = control_qbit_in;

        // Parameter of the gate after the decomposition of the unitary is done
        parameters = Matrix_real(1, parameter_num);

}

/**
@brief Destructor of the class
*/
CZ_NU::~CZ_NU() {
}

/**
@brief Call to retrieve the gate matrix
@return Returns with the matrix of the gate
*/
Matrix
CZ_NU::get_matrix( Matrix_real& parameters ) {

    return get_matrix( parameters, false );
}


/**
@brief Call to retrieve the gate matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with the matrix of the gate
*/
Matrix
CZ_NU::get_matrix( Matrix_real& parameters, int parallel) {

    Matrix CZ_matrix = create_identity(matrix_size);
    apply_to(  parameters, CZ_matrix, parallel);

    return CZ_matrix;
}




/**
@brief Call to apply the gate on the input array/matrix CZ*input
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
CZ_NU::apply_to( Matrix_real& parameters, Matrix& input, int parallel ) {

    double param = parameters[0];

    Matrix u3_1qbit = calc_one_qubit_u3( param );
    apply_kernel_to(u3_1qbit, input, false, parallel);

}



/**
@brief Call to apply the gate on the input array/matrix by input*CZ
@param input The input array on which the gate is applied
*/
void 
CZ_NU::apply_from_right( Matrix_real& parameters, Matrix& input ) {

    if (input.cols != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CRY apply_from_right" << std::endl;
        print(sstream, 0);	
        throw "Wrong matrix size in CRY apply_from_right";
    }
    
    double param = parameters[0];
    
    Matrix u3_1qbit = calc_one_qubit_u3( param );
    apply_kernel_from_right(u3_1qbit, input);

}


/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param inputs The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
CZ_NU::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& inputs, int parallel ) {

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
@brief Call to evaluate the derivate of the circuit on an input with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
std::vector<Matrix> 
CZ_NU::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CRY gate apply" << std::endl;
        print(sstream, 0);	   
        throw "Wrong matrix size in CRY gate apply";
    }

    std::vector<Matrix> ret;

    double param = parameters_mtx[0]+M_PI/2;

    // the resulting matrix
    Matrix res_mtx = input.copy();   


    // get the gate kernel
    Matrix u3_1qbit = calc_one_qubit_u3( param );


    // apply the computing kernel on the matrix
    bool deriv = true;
    apply_kernel_to(u3_1qbit, res_mtx, deriv, parallel);

    ret.push_back(res_mtx);
    return ret;


}




/**
@brief Call to set the final optimized parameters of the gate.
@param param Real parameter of the gate.
*/
void CZ_NU::set_optimized_parameters(double param ) {

    parameters = Matrix_real(1, parameter_num);

    parameters[0] = param;

}


/**
@brief Call to get the final optimized parameters of the gate.
@return Returns with an array containing the optimized parameter
*/
Matrix_real CZ_NU::get_optimized_parameters() {

    return parameters.copy();

}



/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num The number of qubits
*/
void CZ_NU::set_qbit_num(int qbit_num) {
        // setting the number of qubits
        Gate::set_qbit_num(qbit_num);

}



/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void CZ_NU::reorder_qubits( vector<int> qbit_list) {

        Gate::reorder_qubits(qbit_list);

}

/**
@brief Set static values for matrix of the gates.
@param u3_1qbit Matrix parameter for the gate.
*/
Matrix 
CZ_NU::calc_one_qubit_u3( double& param ){

    Matrix u3_1qbit = Matrix(2,2);
    u3_1qbit[0].real = 1.0;         u3_1qbit[0].imag = 0.0; 
    u3_1qbit[1].real = 0.0;         u3_1qbit[1].imag = 0.0;
    u3_1qbit[2].real = 0.0;         u3_1qbit[2].imag = 0.0;
    u3_1qbit[3].real = cos( param); u3_1qbit[3].imag = 0.0;
    return u3_1qbit;

}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CZ_NU* CZ_NU::clone() {

    CZ_NU* ret = new CZ_NU( qbit_num, target_qbit, control_qbit );

    return ret;

}



