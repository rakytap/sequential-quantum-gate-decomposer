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
/*! \file qgd/Operation_block.h
    \brief Header file for a class responsible for grouping CNOT and U3 operations into layers
*/


#pragma once
#include <vector> 
#include "qgd/common.h"
#include "qgd/Operation.h"


using namespace std;

/**
@brief A class responsible for grouping CNOT and U3 operations into layers
*/
class Operation_block :  public Operation {
    

protected:    
    /// The list of stored operations
    vector<Operation*> operations;
    /// number of operation layers
    int layer_num;

public:

/**
@brief Deafult constructor of the class.
*/
Operation_block();

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
*/
Operation_block(int qbit_num_in);


/**
@brief Destructor of the class.
*/
virtual ~Operation_block();

/**
@brief Call to release the stored operations
*/
void release_operations();

/**
@brief Call to retrieve the operation matrix (Which is the product of all the operation matrices stored in the operation block)
@param parameters An array pointing to the parameters of the operations
@return Returns with a pointer to the operation matrix
*/
QGD_Complex16* matrix( const double* parameters );


/**
@brief Call to retrieve the operation matrix (Which is the product of all the operation matrices stored in the operation block)
@param parameters An array pointing to the parameters of the operations
@param block_mtx A preallocated array to store the matrix of the operation block.
@return Returns with 0 on seccess
*/
int matrix( const double* parameters, QGD_Complex16* block_mtx  );

/**
@brief Call to get the list of matrix representation of the operations grouped in the block.
@param parameters Array of parameters to calculate the matrix of the operation block
@return Returns with the list of the operations
*/
std::vector<QGD_Complex16*> get_matrices(const double* parameters );

/**
@brief Append a U3 gate to the list of operations
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param Theta The Theta parameter of the U3 operation
@param Phi The Phi parameter of the U3 operation
@param Lambda The Lambda parameter of the U3 operation
*/
void add_u3_to_end(int target_qbit, bool Theta, bool Phi, bool Lambda);
    
/**
@brief Add a U3 gate to the front of the list of operations
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param Theta The Theta parameter of the U3 operation
@param Phi The Phi parameter of the U3 operation
@param Lambda The Lambda parameter of the U3 operation
*/
void add_u3_to_front(int target_qbit, bool Theta, bool Phi, bool Lambda);
        
/** 
@brief Append a C_NOT gate operation to the list of operations
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cnot_to_end( int control_qbit, int target_qbit);
        
        
    
/**
@brief Add a C_NOT gate operation to the front of the list of operations
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cnot_to_front( int control_qbit, int target_qbit );
    
/**
@brief Append a list of operations to the list of operations
@param operations_in A list of operation class instances.
*/
void add_operations_to_end( vector<Operation*> operations_in );
            
    
/**
@brief Add an array of operations to the front of the list of operations
@param operations_in A list of operation class instances.
*/
void add_operations_to_front( vector<Operation*> operations_in );
    
    
/**
@brief Append a general operation to the list of operations
@param operation A pointer to a class Operation describing an operation.
*/
void add_operation_to_end( Operation* operation );
    
/**
@brief Add an operation to the front of the list of operations
@param operation A pointer to a class Operation describing an operation.
*/
void add_operation_to_front( Operation* operation );

            
            
/**
@brief Call to get the number of the individual gate types in the list of operations
@return Returns with an instance gates_num describing the number of the individual gate types
*/ 
gates_num get_gate_nums();


/**
@brief Call to get the number of free parameters
@return Return with the number of parameters of the operations grouped in the operation block.
*/
int get_parameter_num();


/**
@brief Call to get the number of operations grouped in the class
@return Return with the number of the operations grouped in the operation block.
*/
int get_operation_num();
   
    
/**
@brief Call to print the list of operations stored in the block of operations for a specific set of parameters
@param parameters The parameters of the operations that should be printed.
@param start_index The ordinal number of the first operation.
*/
void list_operations( const double* parameters, int start_index );
    
    
/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void reorder_qubits( vector<int> qbit_list );


/**
@brief Call to get the qubits involved in the operations stored in the block of operations.
@return Return with a list of the invovled qubits
*/
std::vector<int> get_involved_qubits();

/**
@brief Call to get the operations stored in the class.
@return Return with a list of the operations.
*/
std::vector<Operation*> get_operations();

/**
@brief Call to append the operations of an operation block to the current block
@param op_block A pointer to an instance of class Operation_block
*/
void combine(Operation_block* op_block);


/**
@brief Set the number of qubits spanning the matrix of the operations stored in the block of operations.
@param qbit_num_in The number of qubits spanning the matrices.
*/
void set_qbit_num( int qbit_num_in );


/**
@brief Create a clone of the present class.
@return Return with a pointer pointing to the cloned object.
*/
Operation_block* clone();

};


        
