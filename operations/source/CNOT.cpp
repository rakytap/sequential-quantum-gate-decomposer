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

//
// @brief A base class responsible for constructing matrices of C-NOT gates
// gates acting on the N-qubit space

#include "CNOT.h"


using namespace std;




    ////
    // @brief Constructor of the class.
    // @param qbit_num The number of qubits in the unitaries
    // @param parameter_labels A list of strings 'Theta', 'Phi' or 'Lambda' indicating the free parameters of the U3 operations. (Paremetrs which are not labeled are set to zero)
CNOT::CNOT(int qbit_num_in, int target_qbit_in,  int control_qbit_in) {

        // number of qubits spanning the matrix of the operation
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the operation
        type = "cnot";
        // The number of free parameters
        parameter_num = 0;

        if (target_qbit_in >= qbit_num) {
            printf("The index of the target qubit is larger than the number of qubits");
            throw "The index of the target qubit is larger than the number of qubits";
        }
        // The index of the qubit on which the operation acts (target_qbit >= 0) 
        target_qbit = target_qbit_in;


        if (control_qbit_in >= qbit_num) {
            printf("The index of the control qubit is larger than the number of qubits");
            throw "The index of the control qubit is larger than the number of qubits";
        }
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
        control_qbit = control_qbit_in;

        // Contruct the matrix of the operation
        matrix_alloc = composite_cnot();

}

//
// @brief Destructor of the class
CNOT::~CNOT() {
}


        

////    
// @brief Calculate the matrix of a CNOT gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
// @return Returns with the matrix of the CNOT gate.
MKL_Complex16* CNOT::composite_cnot() {


        // preallocate array for the composite u3 operation
        MKL_Complex16* CNOT_mtx = (MKL_Complex16*)mkl_calloc(matrix_size*matrix_size, sizeof(MKL_Complex16), 64);

        int target_qbit_power = Power_of_2(target_qbit);
        int control_qbit_power = Power_of_2(control_qbit);


        // setting the operation elements
        #pragma omp parallel for
        for(int idx = 0; idx < matrix_size*matrix_size; ++idx)
        {

            int col_idx = idx % matrix_size;
            int row_idx = (idx-col_idx)/matrix_size;


            
            // determine the row state of the control and target qubits corresponding to the given thread
            int target_qubit_state_row = int(row_idx / target_qbit_power) % 2;
            int control_qubit_state_row = int(row_idx / control_qbit_power) % 2;
            int state_row_remaining = row_idx;
            if (target_qubit_state_row == 1) {
                state_row_remaining = state_row_remaining - target_qbit_power;
            }
            if (control_qubit_state_row == 1) {
                state_row_remaining = state_row_remaining - control_qbit_power;
            }


            // determine the col state of the control and target qubits corresponding to the given thread
            int target_qubit_state_col = int(col_idx / target_qbit_power) % 2;
            int control_qubit_state_col = int(col_idx / control_qbit_power) % 2;
            int state_col_remaining = col_idx;
            if (target_qubit_state_col == 1) {
                state_col_remaining = state_col_remaining - target_qbit_power;
            }
            if (control_qubit_state_col == 1) {
                state_col_remaining = state_col_remaining - control_qbit_power;
            }

            
            // setting the col_idx-th element in the row
            if (control_qubit_state_row == 0 && control_qubit_state_col == 0 && target_qubit_state_row == target_qubit_state_col && state_row_remaining == state_col_remaining) {
                CNOT_mtx[idx].real = 1;
            }
            //else if (control_qubit_state_row == 0 && control_qubit_state_col == 0 && target_qubit_state_row != target_qubit_state_col && state_row_remaining == state_col_remaining) {
            //    CNOT_mtx[idx] = 0;
            //}
            //else if (control_qubit_state_row == 1 && control_qubit_state_col == 1 && target_qubit_state_row == target_qubit_state_col && state_row_remaining == state_col_remaining) {
            //    CNOT_mtx[idx] = 0;
            //}
            else if (control_qubit_state_row == 1 && control_qubit_state_col == 1 && target_qubit_state_row != target_qubit_state_col && state_row_remaining == state_col_remaining) {
                CNOT_mtx[idx].real = 1;
            }
            //else: {
            //     CNOT_mtx[idx] = 0;
            //}


        }

        return CNOT_mtx;
}


////
// @brief Sets the number of qubits spanning the matrix of the operation
// @param qbit_num The number of qubits
void CNOT::set_qbit_num(int qbit_num_in) {
        // setting the number of qubits
        Operation::set_qbit_num(qbit_num_in);

        if ( matrix_alloc != NULL ) {
            mkl_free( matrix_alloc );
        }

        // Contruct the matrix of the operation
        matrix_alloc = composite_cnot();

}



////
// @brief Call to reorder the qubits in the matrix of the operation
// @param qbit_list The list of qubits spanning the matrix
void CNOT::reorder_qubits( vector<int> qbit_list) {

        Operation::reorder_qubits(qbit_list);

        // setting the control qubit
        if ( control_qbit != -1 ) {
            control_qbit = qbit_list[qbit_list.size()-control_qbit-1];
        }

        if ( matrix_alloc != NULL ) {
            mkl_free( matrix_alloc );
        }

        // Contruct the matrix of the operation
        matrix_alloc = composite_cnot();
}



//
// @brief Create a clone of the present class
// @return Return with a pointer pointing to the cloned object
CNOT* CNOT::clone() {

    CNOT* ret = new CNOT( qbit_num, target_qbit, control_qbit );
 
    if (matrix_alloc != NULL) {
        MKL_Complex16* mtx = (MKL_Complex16*)mkl_malloc( matrix_size*matrix_size*sizeof(MKL_Complex16), 64);
        memcpy( mtx, matrix_alloc, matrix_size*matrix_size*sizeof(MKL_Complex16) );
        ret->set_matrix( mtx);
    }
    

    return ret;

}


                   
