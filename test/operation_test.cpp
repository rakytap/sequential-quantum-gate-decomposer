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
#include "CNOT.h"
#include "Operations.h"
#include "Operation_block.h"

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
        

    // creating U3 operation
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
    print_mtx( matrix, Power_of_2(qbit_num), Power_of_2(qbit_num));
   

    // reorder qubits, and test the modified target qubit (1,0) -> (0,1)
    vector<int> qbit_list;
    qbit_list.push_back(0);
    qbit_list.push_back(1);
    op.reorder_qubits( qbit_list );        
    
    // check the reordered matrix
    printf("The matrix of %d qubit U3 operator acting on target qubit %d\n", qbit_num, op.get_target_qbit() );
    matrix = op.matrix(parameters);
    print_mtx( matrix, Power_of_2(qbit_num), Power_of_2(qbit_num));

};


void test_CNOT_operation() {
 
    printf("****************************************\n");
    printf("Test of operation CNOT\n\n");


    // define the nmumber of qubits spanning the matrices
    int qbit_num = 3;
    
    // the target qbit of the U3 operation
    int target_qbit = 0;
    
    // the control qbit of the U3 operation
    int control_qbit = 1;
        

    // creating gereal operation
    CNOT op = CNOT( qbit_num, target_qbit, control_qbit );  
    
    // check the CNOT matrix
    MKL_Complex16* matrix = op.matrix();
    printf("The matrix of %d qubit CNOT operator acting on target qubit %d with control qubit %d\n", qbit_num, op.get_target_qbit(), op.get_control_qbit() );
    print_CNOT( matrix, Power_of_2(qbit_num));
    
    // reorder qubits, and test the modified target qubit
    vector<int> qbit_list;
    qbit_list.push_back(2);
    qbit_list.push_back(0);
    qbit_list.push_back(1);
    op.reorder_qubits( qbit_list );
    if (op.get_target_qbit() != qbit_list[qbit_list.size()-target_qbit-1] ) {
        printf("Reordering qubits does not work properly");
        //throw "Reordering qubits does not work properly";
    }
        
    
    // check the reordered CNOT matrix
    matrix = op.matrix();
    printf("The matrix of %d qubit CNOT operator acting on target qubit %d with control qubit %d\n", qbit_num, op.get_target_qbit(), op.get_control_qbit() );
    print_CNOT( matrix, Power_of_2(qbit_num));
};



void test_operations() {
    
    printf("****************************************\n");
    printf("Test of operations\n\n");

    // define the nmumber of qubits spanning the matrices
    int qbit_num = 3;
    
    // create class intance storing quantum gate operations
    Operations operations = Operations( qbit_num );


    // adding operations to the list
    bool Theta = true;
    bool Phi = false;
    bool Lambda = true;
    operations.add_u3_to_end(1, Theta, Phi, Lambda);

    int target_qbit = 1;
    int control_qbit = 2;
    operations.add_cnot_to_end(target_qbit, control_qbit);

    
    // get the number of parameters
    printf( "The number of parameters in the list of operations is %d\n", operations.get_parameter_num());
    
};




void test_operation_block() {
    
    printf("****************************************\n");
    printf("Test of operation_block\n\n");


    // define the nmumber of qubits spanning the matrices
    int qbit_num = 2;
    
    // create class intance storing quantum gate operations
    Operation_block* op_block = new Operation_block( qbit_num );
    
    // adding operations to the list
    bool Theta = true;
    bool Phi = false;
    bool Lambda = true;
    // adding a U3 operation on target qubit 1
    printf("adding a U3 operation on target qubit %d \n", 1);
    op_block->add_u3_to_end(1, Theta, Phi, Lambda);
    // adding a U3 operation on target qubit 0
    printf("adding a U3 operation on target qubit %d \n", 0);
    op_block->add_u3_to_end(0, Theta, Phi, Lambda);

    int target_qbit = 0;
    int control_qbit = 1;
    // adding CNOT operation
    printf("adding a CNOT operation on target qubit %d and control qubit %d\n", 0, 1);
    op_block->add_cnot_to_end(target_qbit, control_qbit);
    
    // get the number of parameters
    printf( "The number of parameters in the list of operations is %d\n", op_block->get_parameter_num());


    // construct parameters for the two U3 operations
    double parameters[4];
    parameters[0] = 0;
    parameters[1] = 1.2;
    parameters[2] = 0.3;
    parameters[3] = 0.67;

    // calculate the matrix product stored in the operation block
    MKL_Complex16* mtx = op_block->matrix( parameters );


    // check the block matrix
    printf("The matrix of %d qubit operators consisting of 2 U3 operations and 1 CNOT operation is:\n", qbit_num);
    print_mtx( mtx, Power_of_2(qbit_num), Power_of_2(qbit_num));

    mkl_free(mtx);
    delete( op_block );




    // define the nmumber of qubits spanning the matrices
    qbit_num = 5;


    printf("Testing getting involved qubits for operation block consisting of %d qubits\n", qbit_num);
    
    // create class intance storing quantum gate operations
    op_block = new Operation_block( qbit_num );
    
    // adding operations to the list
    // adding a U3 operation on target qubit 1
    printf("adding a U3 operation on target qubit %d \n", 1);
    op_block->add_u3_to_end(1, Theta, Phi, Lambda);
    printf("adding a U3 operation on target qubit %d \n", 0);
    // adding a U3 operation on target qubit 0
    op_block->add_u3_to_end(0, Theta, Phi, Lambda);
    printf("adding a U3 operation on target qubit %d \n", 3);
    // adding a U3 operation on target qubit 3
    op_block->add_u3_to_end(3, Theta, Phi, Lambda);
    

    target_qbit = 0;
    control_qbit = 1;
    // adding CNOT operation
    printf("adding a CNOT operation on target qubit %d and control qubit %d\n", 0, 1);
    op_block->add_cnot_to_end(target_qbit, control_qbit);


    // get the number of parameters
    printf( "The number of parameters in the list of operations is %d\n", op_block->get_parameter_num());

    printf("The involved qubits are:\n");
    std::vector<int> involved_qbits = op_block->get_involved_qubits();
    for(std::vector<int>::iterator it = involved_qbits.begin(); it != involved_qbits.end(); ++it) {
        int current_val = *it;
        printf("%d\n", current_val);
    }


    delete( op_block );
  

};


}
