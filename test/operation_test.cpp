/*
Created on Fri Jun 26 14:14:12 2020
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

#include <iostream>

#include <stdio.h>
#include <complex.h>
#include <mkl.h>
#include <vector>

#include "Operation.h"
#include "U3.h"

using namespace std;


extern "C" {
void test_general_operation(int qbit_num) {
    
    printf("****************************************\n");
    printf("Test of general operation\n");
    printf("\n");

    // creating gereal operation
    Operation op = Operation( qbit_num );   
    
    // reorder qubits
    vector<int> qbit_list(qbit_num);

    qbit_list[0] = 2;
    qbit_list[1] = 1;
    qbit_list[2] = 0;
    qbit_list[3] = 3;

    op.reorder_qubits( qbit_list );
};



void test_U3_operation() {
    
    printf("****************************************\n");
    printf("Test of operation U3\n\n");

    // define the nmumber of qubits spanning the matrices
    int qbit_num = 2;
    
    // the target qbit of the U3 operation
    int target_qbit = 1;
        

    // creating gereal operation
    bool Theta = true;
    bool Phi = false;
    bool Lambda = true;
    U3 op = U3( qbit_num, target_qbit, Theta, Phi, Lambda ); 
        
    // check the matrix
    printf("The matrix of %d qubit U3 operator acting on target qubit %d\n", qbit_num, target_qbit );
    double parameters[2];
    parameters[0] = 1;
    parameters[1] = 2;
    
    // construct the matrix of the U3 operation
    MKL_Complex16* matrix = op.matrix( parameters );

    // print the matrix
    print_mtx( matrix, Power_of_2(qbit_num));
   

    // reorder qubits, and test the modified target qubit (1,0) -> (0,1)
    vector<int> qbit_list;
    qbit_list.push_back(0);
    qbit_list.push_back(1);
    op.reorder_qubits( qbit_list );        
    
    // check the reordered matrix
    printf("The matrix of %d qubit U3 operator acting on target qubit %d\n", qbit_num, op.get_target_qbit() );
    matrix = op.matrix(parameters);
    print_mtx( matrix, Power_of_2(qbit_num));

};

}
