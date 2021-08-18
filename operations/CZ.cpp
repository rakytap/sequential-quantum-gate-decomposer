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
/*! \file CZ.cpp
    \brief Class representing a CZ operation.
*/

#include "CZ.h"


using namespace std;


/**
@brief Nullary constructor of the class.
*/
CZ::CZ() {

        // number of qubits spanning the matrix of the operation
        qbit_num = -1;
        // the size of the matrix
        matrix_size = -1;
        // A string describing the type of the operation
        type = CZ_OPERATION;
        // The number of free parameters
        parameter_num = 0;

        // The index of the qubit on which the operation acts (target_qbit >= 0)
        target_qbit = -1;

        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
        control_qbit = -1;


}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
CZ::CZ(int qbit_num_in,  int control_qbit_in, int target_qbit_in) {

        // number of qubits spanning the matrix of the operation
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the operation
        type = CZ_OPERATION;
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


}

/**
@brief Destructor of the class
*/
CZ::~CZ() {
}


/**
@brief Call to retrieve the operation matrix
@return Returns with the matrix of the operation
*/
Matrix
CZ::get_matrix() {
    return composite_cz();
}

/**
@brief Calculate the matrix of a CZ gate operation acting on the space of qbit_num qubits.
@return Returns with the operation matrix
*/
Matrix CZ::composite_cz() {


        // preallocate array for the composite u3 operation
        Matrix CZ_mtx = Matrix(matrix_size, matrix_size);
        QGD_Complex16* CZ_mtx_data = CZ_mtx.get_data();

        // set to zero all the elements of the matrix
        memset(CZ_mtx_data, 0, matrix_size*matrix_size*sizeof(QGD_Complex16) );


        int target_qbit_power = Power_of_2(target_qbit);
        int control_qbit_power = Power_of_2(control_qbit);


        // setting the operation elements
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
                CZ_mtx[idx].real = 1;
                //CZ_mtx[idx].imag = 0;
            }
            /*else if (control_qubit_state_row == 0 && control_qubit_state_col == 0 && target_qubit_state_row != target_qubit_state_col && state_row_remaining == state_col_remaining) {
                CZ_mtx[idx].real = 0;
                CZ_mtx[idx].imag = 0;
            }
            else if (control_qubit_state_row == 1 && control_qubit_state_col == 1 && target_qubit_state_row == target_qubit_state_col && state_row_remaining == state_col_remaining) {
                CZ_mtx[idx].real = 0;
                CZ_mtx[idx].imag = 0;
            }*/
            else if (control_qubit_state_row == 1 && control_qubit_state_col == 1 && target_qubit_state_row == target_qubit_state_col && state_row_remaining == state_col_remaining) {
                CZ_mtx[idx].real = 1-2*target_qubit_state_row;
                //CZ_mtx[idx].imag = 0;
            }
            /*else {
                 CZ_mtx[idx].real = 0;
                 CZ_mtx[idx].imag = 0;
            }*/


        }

#ifdef DEBUG
        if (CZ_mtx.isnan()) {
            std::cout << "Matrix CZ::composite_cnot: CZ_mtx contains NaN. Exiting" << std::endl;
            exit(-1);
        }
#endif

        return CZ_mtx;
}


/**
@brief Call to set the number of qubits spanning the matrix of the operation
@param qbit_num The number of qubits
*/
void CZ::set_qbit_num(int qbit_num) {
        // setting the number of qubits
        Operation::set_qbit_num(qbit_num);

}



/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void CZ::reorder_qubits( vector<int> qbit_list) {

        Operation::reorder_qubits(qbit_list);

}



/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CZ* CZ::clone() {

    CZ* ret = new CZ( qbit_num, control_qbit, target_qbit );

    return ret;

}



