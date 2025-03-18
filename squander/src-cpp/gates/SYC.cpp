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
/*! \file SYC.cpp
    \brief Class representing a SYC gate.
*/

#include "SYC.h"



using namespace std;


/**
@brief Nullary constructor of the class.
*/
SYC::SYC() {

        // number of qubits spanning the matrix of the gate
        qbit_num = -1;
        // the size of the matrix
        matrix_size = -1;
        // A string describing the type of the gate
        type = SYC_OPERATION;
        // The number of free parameters
        parameter_num = 0;

        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = -1;

        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
        control_qbit = -1;


}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
SYC::SYC(int qbit_num_in,  int target_qbit_in, int control_qbit_in) {

        // number of qubits spanning the matrix of the gate
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the gate
        type = SYC_OPERATION;
        // The number of free parameters
        parameter_num = 0;

        if (target_qbit_in >= qbit_num) {
            std::stringstream sstream;
	    sstream << "The index of the target qubit is larger than the number of qubits" << std::endl;
	    print(sstream, 0);
	    throw "The index of the target qubit is larger than the number of qubits";
        }
	
        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = target_qbit_in;


        if (control_qbit_in >= qbit_num) {
            std::stringstream sstream;
	    sstream << "The index of the control qubit is larger than the number of qubits" << std::endl;
	    print(sstream, 0);	
	    throw "The index of the control qubit is larger than the number of qubits";
        }
	
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
        control_qbit = control_qbit_in;

}

/**
@brief Destructor of the class
*/
SYC::~SYC() {
}

/**
@brief Call to retrieve the gate matrix
@return Returns with the matrix of the gate
*/
Matrix
SYC::get_matrix() {


    return get_matrix( false );

}


/**
@brief Call to retrieve the gate matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with the matrix of the gate
*/
Matrix
SYC::get_matrix( int parallel) {

    Matrix SYC_matrix = create_identity(matrix_size);
    apply_to(SYC_matrix, parallel);

    return SYC_matrix;

}


/**
@brief Call to apply the gate on the input array/matrix SYC*input
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
SYC::apply_to( Matrix& input, int parallel ) {

    int index_step_target = Power_of_2(target_qbit);
    int index_step_control = Power_of_2(control_qbit);

    int loopSize   = index_step_target < index_step_control ? index_step_target : index_step_control;
    int iterations = control_qbit < target_qbit ? Power_of_2(target_qbit-control_qbit-1) : Power_of_2(control_qbit-target_qbit-1);

    // |control, target>
    int idx00 = 0;
    int idx01 = index_step_target;
    int idx10 = index_step_control;
    int idx11 = index_step_target + index_step_control;

/*
std::cout << "target qubit: " << target_qbit << std::endl;
std::cout << "control qubit: " << control_qbit << std::endl;
std::cout << "iterations: " << iterations << std::endl;
std::cout << "loopSize: " << loopSize << std::endl;
*/
    while ( idx11 < matrix_size ) {

        for ( int jdx=0; jdx<iterations; jdx++ ) {

            tbb::parallel_for(0, loopSize, 1, [&](int idx) {  

                // |control, target>
                //int idx00_loc = idx00 + idx;
                int idx01_loc = idx01 + idx;
                int idx10_loc = idx10 + idx;
                int idx11_loc = idx11 + idx;


                //int offset00 = idx00_loc*input.stride;
                int offset01 = idx01_loc*input.stride;
                int offset10 = idx10_loc*input.stride;
                int offset11 = idx11_loc*input.stride;


                for (int col_idx=0; col_idx<input.cols; col_idx++) {

                    // transform elements 10 and 01
                    QGD_Complex16 element01 = input[ offset01 + col_idx ];
                    QGD_Complex16 element10 = input[ offset10 + col_idx ];
                    input[ offset01 + col_idx ].real = element10.imag;
                    input[ offset01 + col_idx ].imag = -element10.real;

                    input[ offset10 + col_idx ].real = element01.imag;
                    input[ offset10 + col_idx ].imag = -element01.real;


                    // transform element 11
                    QGD_Complex16 element11 = input[ offset11 + col_idx ];
                    QGD_Complex16 factor;
                    factor.real = sqrt(3)/2;
                    factor.imag = -0.5;
                    input[ offset11 + col_idx ] = mult(element11, factor);
                }

            

    //std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;


            });

        

            idx00 = idx00 + 2*loopSize;
            idx01 = idx01 + 2*loopSize;
            idx10 = idx10 + 2*loopSize;
            idx11 = idx11 + 2*loopSize;


        }


        idx00 = idx00 + 2*loopSize*iterations;
        idx01 = idx01 + 2*loopSize*iterations;
        idx10 = idx10 + 2*loopSize*iterations;
        idx11 = idx11 + 2*loopSize*iterations;


    }


}


/**
@brief Call to apply the gate on the input array/matrix by input*SYC
@param input The input array on which the gate is applied
*/
void 
SYC::apply_from_right( Matrix& input ) {
/*
Matrix SYC_gate = get_matrix();
SYC_gate.print_matrix();
Matrix res = dot(input, SYC_gate);
memcpy( input.get_data(), res.get_data(), res.size()*sizeof(QGD_Complex16) );
return;
*/
    int index_step_target = Power_of_2(target_qbit);
    int index_step_control = Power_of_2(control_qbit);

    int loopSize   = index_step_target < index_step_control ? index_step_target : index_step_control;
    int iterations = control_qbit < target_qbit ? Power_of_2(target_qbit-control_qbit-1) : Power_of_2(control_qbit-target_qbit-1);



    // loop over the rows of the input matrix
    tbb::parallel_for(0, matrix_size, 1, [&](int idx) {  

        int offset = idx*input.stride;
        
        // |control, target>
        int idx00 = 0;
        int idx01 = index_step_target;
        int idx10 = index_step_control;
        int idx11 = index_step_target + index_step_control;

        while ( idx11 < matrix_size ) {

            for ( int jdx=0; jdx<iterations; jdx++ ) {


                for ( int idx=0; idx<loopSize; idx++ ) {

                    // |control, target>
                    //int idx00_loc = idx00 + idx;
                    int idx01_loc = idx01 + idx;
                    int idx10_loc = idx10 + idx;
                    int idx11_loc = idx11 + idx;

                    // transform elements 10 and 01
                    QGD_Complex16 element01 = input[ offset + idx01_loc ];
                    QGD_Complex16 element10 = input[ offset + idx10_loc ];
                    input[ offset + idx01_loc ].real = element10.imag;
                    input[ offset + idx01_loc ].imag = -element10.real;

                    input[ offset + idx10_loc ].real = element01.imag;
                    input[ offset + idx10_loc ].imag = -element01.real;


                    // transform element 11
                    QGD_Complex16 element11 = input[ offset + idx11_loc ];
                    QGD_Complex16 factor;
                    factor.real = sqrt(3)/2;
                    factor.imag = -0.5;
                    input[ offset + idx11_loc ] = mult(element11, factor);

                }

                idx00 = idx00 + 2*loopSize;
                idx01 = idx01 + 2*loopSize;
                idx10 = idx10 + 2*loopSize;
                idx11 = idx11 + 2*loopSize;


            }

            idx00 = idx00 + 2*loopSize*iterations;
            idx01 = idx01 + 2*loopSize*iterations;
            idx10 = idx10 + 2*loopSize*iterations;
            idx11 = idx11 + 2*loopSize*iterations;

        }

    });


}



/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num The number of qubits
*/
void SYC::set_qbit_num(int qbit_num) {
        // setting the number of qubits
        Gate::set_qbit_num(qbit_num);

}



/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void SYC::reorder_qubits( vector<int> qbit_list) {

        Gate::reorder_qubits(qbit_list);

}



/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
SYC* SYC::clone() {

    SYC* ret = new SYC( qbit_num, target_qbit, control_qbit );
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}



