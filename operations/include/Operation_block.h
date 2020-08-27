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


#pragma once
#include <vector> 
#include "qgd/common.h"
#include "qgd/Operation.h"


using namespace std;

//
// @brief A base class responsible for grouping C-NOT and U3 operations into layers
class Operation_block :  public Operation {
    

protected:    
    // The list of stored operations
    vector<Operation*> operations;
    // number of operation layers
    int layer_num;

public:
//
// @brief Deafult constructor of the class.
Operation_block();

//
// @brief Constructor of the class.
// @param qbit_num The number of qubits in the unitaries
Operation_block(int);


//
// @brief Destructor of the class.
~Operation_block();

//
// @brief Call to terive the operation matrix
// @return Returns with a pointer to the operation matrix
MKL_Complex16* matrix( const double* parameters );


//
// @brief Call to terive the operation matrix
// @param free_after_used Logical value indicating whether the cteated matrix can be freed after it was used. (For example U3 allocates the matrix on demand, but CNOT is returning with a pointer to the stored matrix in attribute matrix_allocate)
// @return Returns with a pointer to the operation matrix
int matrix( const double* parameters, MKL_Complex16* block_mtx  );

////
// @brief Call to get the list of matrix representation of the operations grouped in the block.
// @param parameters List of parameters to calculate the matrix of the operation block
// @param free_after_used Array of logical value indicating whether the cteated matrixes can bee freed after they were used or not. (For example U3 allocates the matrix on demand, but CNOT is returning with a pointer to the stored matrix in attribute matrix_allocate)
// @return Returns with a pointer to the operation matrix
std::vector<MKL_Complex16*> get_matrices(const double* parameters );

////
// @brief Append a U3 gate to the list of operations
// @param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
// @param theta_in ...
void add_u3_to_end(int, bool, bool, bool);
    
////
// @brief Add a U3 gate to the front of the list of operations
// @param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
// @param parameter_labels A list of strings 'Theta', 'Phi' or 'Lambda' indicating the free parameters of the U3 operations. (Paremetrs which are not labeled are set to zero)
void add_u3_to_front(int, bool, bool, bool);
        
//// 
// @brief Append a C_NOT gate operation to the list of operations
// @param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
// @param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
void add_cnot_to_end( int, int);
        
        
    
//// add_cnot_to_front
// @brief Add a C_NOT gate operation to the front of the list of operations
// @param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
// @param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
void add_cnot_to_front( int, int);
    
////
// @brief Append an array of operations to the list of operations
// @param operations A list of operation class instances.
void add_operations_to_end( vector<Operation*>);
            
    
//// add_operations_to_front
// @brief Add an array of operations to the front of the list of operations
// @param operations A list of operation class instances.
void add_operations_to_front( vector<Operation*>);
    
    
//// add_operation_to_end
// @brief Append an operation to the list of operations
// @param operation An instance of class describing an operation.
void add_operation_to_end( Operation* );
    
//// add_operation_to_front
// @brief Add an operation to the front of the list of operations
// @param operation A class describing an operation.
void add_operation_to_front( Operation* );

            
            
////
// @brief Call to get the number of specific gates in the decomposition
// @return Returns with a dictionary containing the number of specific gates. 
gates_num get_gate_nums();


//
// @brief Call to get the number of free parameters
// @return Return with the index of the target qubit (return with -1 if target qubit was not set)
int get_parameter_num();


//
// @brief Call to get the number of operations grouped in the block
// @return Return with the number of the operations
int get_operation_num();
   
    
////
// @brief Lists the operations decomposing the initial unitary. (These operations are the inverse operations of the operations bringing the intial matrix into unity.)
// @param parameters The parameters of the operations that should be inverted
// @param start_index The index of the first inverse operation
void list_operations( const double*, int );
    
    
////
// @brief Call to reorder the qubits in the in the stored operations
// @param qbit_list A list of the permutation of the qubits (for example [1 3 0 2])
void reorder_qubits( vector<int> );


////
// @biref Call to get the involved qubits in the operations stored in the block
// @return Return with an array of the invovled qubits
std::vector<int> get_involved_qubits();

////
// @biref Call to get the operations stored in the block
// @return Return with a vector of Operations.
std::vector<Operation*> get_operations();

////
// @biref Call to append the operations of an operation bolck to the current block
// @param an instance of class @operation_block
void combine(Operation_block* op_block);


//// 
// @brief Set the number of qubits spanning the matrix of the operation stored in the block
// @param qbit_num_in The number of qubits spanning the matrix
void set_qbit_num( int qbit_num_in );


//
// @brief Create a clone of the present class
// @return Return with a pointer pointing to the cloned object
Operation_block* clone();

};


        
