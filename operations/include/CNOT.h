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

#include "Operation.h"
#include <math.h>

using namespace std;


class CNOT: public Operation {

protected:

// the base indices of the target qubit for state |0>
int* indexes_target_qubit_0;
// the base indices of the target qubit for state |1>
int* indexes_target_qubit_1;    
        

public: 
////
// @brief Constructor of the class.
// @param qbit_num The number of qubits in the unitaries
// @param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
// @param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
CNOT(int, int, int);

//
// @brief Destructor of the class
~CNOT();

//
// @brief Call to terive the operation matrix
MKL_Complex16* matrix();



////
// @brief Sets the number of qubits spanning the matrix of the operation
// @param qbit_num The number of qubits
void set_qbit_num(int qbit_num);


// @brief Calculate the matrix of a C_NOT gate operation acting on the space of qbit_num qubits.
// @param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
// @param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
// @return Returns with the matrix of the C-NOT gate.
MKL_Complex16* composite_cnot();



    ////
    // @brief Call to reorder the qubits in the matrix of the operation
    // @param qbit_list The list of qubits spanning the matrix
void reorder_qubits( vector<int> qbit_list);



};

                   
