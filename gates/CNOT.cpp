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
/*! \file CNOT.cpp
    \brief Class representing a CNOT gate.
*/

#include "CNOT.h"


using namespace std;


/**
@brief Nullary constructor of the class.
*/
CNOT::CNOT() {

        // number of qubits spanning the matrix of the gate
        qbit_num = -1;
        // the size of the matrix
        matrix_size = -1;
        // A string describing the type of the gate
        type = CNOT_OPERATION;
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
CNOT::CNOT(int qbit_num_in,  int target_qbit_in, int control_qbit_in) {

 
        // number of qubits spanning the matrix of the gate
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the gate
        type = CNOT_OPERATION;
        // The number of free parameters
        parameter_num = 0;

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
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
        control_qbit = control_qbit_in;


}

/**
@brief Destructor of the class
*/
CNOT::~CNOT() {
}


/**
@brief Call to retrieve the gate matrix
@return Returns with the matrix of the gate
*/
Matrix
CNOT::get_matrix() {

    Matrix CNOT_matrix = create_identity(matrix_size);
    apply_to(CNOT_matrix);

    return CNOT_matrix;

}


/**
@brief Call to apply the gate on the input array/matrix CNOT*input
@param input The input array on which the gate is applied
*/
void 
CNOT::apply_to( Matrix& input ) {

    int index_step_target = Power_of_2(target_qbit);
    int current_idx = 0;
    int current_idx_pair = current_idx+index_step_target;

    int index_step_control = Power_of_2(control_qbit);

//std::cout << "target qbit: " << target_qbit << std::endl;

    while ( current_idx_pair < matrix_size ) {

        tbb::parallel_for(0, index_step_target, 1, [&](int idx) {  

            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            int row_offset = current_idx_loc*input.stride;
            int row_offset_pair = current_idx_pair_loc*input.stride;

            // determine the action according to the state of the control qubit
            if ( (current_idx_loc/index_step_control) % 2 == 0) {
                // leave the state as it is
                return;
            }
            else {
                for ( int col_idx=0; col_idx<matrix_size; col_idx++) {
                    int index      = row_offset+col_idx;
                    int index_pair = row_offset_pair+col_idx;                

                    QGD_Complex16 element      = input[index];
                    QGD_Complex16 element_pair = input[index_pair];              

                    input[index] = element_pair;
                    input[index_pair] = element;

                }                     

            }


//std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;


        });


        current_idx = current_idx + 2*index_step_target;
        current_idx_pair = current_idx_pair + 2*index_step_target;


    }


}


/**
@brief Call to apply the gate on the input array/matrix by input*CNOT
@param input The input array on which the gate is applied
*/
void 
CNOT::apply_from_right( Matrix& input ) {



    int index_step_target = Power_of_2(target_qbit);
    int current_idx = 0;
    int current_idx_pair = current_idx+index_step_target;

    int index_step_control = Power_of_2(control_qbit);

//std::cout << "target qbit: " << target_qbit << std::endl;

    while ( current_idx_pair < matrix_size ) {

        tbb::parallel_for(0, index_step_target, 1, [&](int idx) {  

            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            // determine the action according to the state of the control qubit
            if ( (current_idx_loc/index_step_control) % 2 == 0) {
                // leave the state as it is
                return;
            }
            else {
                for ( int row_idx=0; row_idx<matrix_size; row_idx++) {

                    int row_offset = row_idx*input.stride;

                    int index      = row_offset+current_idx_loc;
                    int index_pair = row_offset+current_idx_pair_loc;                

                    QGD_Complex16 element      = input[index];
                    QGD_Complex16 element_pair = input[index_pair];              

                    input[index] = element_pair;
                    input[index_pair] = element;

                }                     

            }


//std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;


        });


        current_idx = current_idx + 2*index_step_target;
        current_idx_pair = current_idx_pair + 2*index_step_target;


    }








}



/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num The number of qubits
*/
void CNOT::set_qbit_num(int qbit_num) {
        // setting the number of qubits
        Gate::set_qbit_num(qbit_num);

}



/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void CNOT::reorder_qubits( vector<int> qbit_list) {

        Gate::reorder_qubits(qbit_list);

}



/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CNOT* CNOT::clone() {

    CNOT* ret = new CNOT( qbit_num, target_qbit, control_qbit );

    return ret;

}



