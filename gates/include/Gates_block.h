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
/*! \file Gates_block.h
    \brief Header file for a class responsible for grouping two-qubit (CNOT,CZ,CH) and one-qubit gates into layers
*/

#ifndef GATES_BLOCK_H
#define GATES_BLOCK_H

#include <vector>
#include "common.h"
#include "Gate.h"



/**
@brief A class responsible for grouping two-qubit (CNOT,CZ,CH) and one-qubit gates into layers
*/
class Gates_block :  public Gate {


protected:
    /// The list of stored gates
    std::vector<Gate*> gates;
    /// number of gate layers
    int layer_num;

public:

/**
@brief Default constructor of the class.
*/
Gates_block();

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
*/
Gates_block(int qbit_num_in);


/**
@brief Destructor of the class.
*/
virtual ~Gates_block();

/**
@brief Call to release the stored gates
*/
void release_gates();

/**
@brief Call to retrieve the gate matrix (Which is the product of all the gate matrices stored in the gate block)
@param parameters An array pointing to the parameters of the gates
@return Returns with the gate matrix
*/
Matrix get_matrix( const double* parameters );

/**
@brief Call to apply the gate on the input array/matrix
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void apply_to( const double* parameters, Matrix& input );

/**
@brief Call to get the list of matrix representation of the gates grouped in the block.
@param parameters Array of parameters to calculate the matrix of the gate block
@return Returns with the list of the gates
*/
std::vector<Matrix> get_matrices(const double* parameters );

/**
@brief Append a U3 gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param Theta The Theta parameter of the U3 gate
@param Phi The Phi parameter of the U3 gate
@param Lambda The Lambda parameter of the U3 gate
*/
void add_u3_to_end(int target_qbit, bool Theta, bool Phi, bool Lambda);

/**
@brief Add a U3 gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param Theta The Theta parameter of the U3 gate
@param Phi The Phi parameter of the U3 gate
@param Lambda The Lambda parameter of the U3 gate
*/
void add_u3(int target_qbit, bool Theta, bool Phi, bool Lambda);

/**
@brief Append a CNOT gate gate to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cnot_to_end( int target_qbit, int control_qbit);



/**
@brief Add a C_NOT gate gate to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cnot( int target_qbit, int control_qbit );


/**
@brief Append a CZ gate gate to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cz_to_end( int target_qbit, int control_qbit);



/**
@brief Add a CZ gate gate to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cz( int target_qbit, int control_qbit );


/**
@brief Append a CH gate (i.e. controlled Hadamard gate) gate to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_ch_to_end( int target_qbit, int control_qbit);



/**
@brief Add a CH gate (i.e. controlled Hadamard gate) gate to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_ch( int target_qbit, int control_qbit );

/**
@brief Append a list of gates to the list of gates
@param gates_in A list of gate class instances.
*/
void add_gates_to_end( std::vector<Gate*> gates_in );


/**
@brief Add an array of gates to the front of the list of gates
@param gates_in A list of gate class instances.
*/
void add_gates( std::vector<Gate*> gates_in );


/**
@brief Append a general gate to the list of gates
@param gate A pointer to a class Gate describing an gate.
*/
void add_gate_to_end( Gate* gate );

/**
@brief Add an gate to the front of the list of gates
@param gate A pointer to a class Gate describing an gate.
*/
void add_gate( Gate* gate );



/**
@brief Call to get the number of the individual gate types in the list of gates
@return Returns with an instance gates_num describing the number of the individual gate types
*/
gates_num get_gate_nums();


/**
@brief Call to get the number of free parameters
@return Return with the number of parameters of the gates grouped in the gate block.
*/
int get_parameter_num();


/**
@brief Call to get the number of gates grouped in the class
@return Return with the number of the gates grouped in the gate block.
*/
int get_gate_num();


/**
@brief Call to print the list of gates stored in the block of gates for a specific set of parameters
@param parameters The parameters of the gates that should be printed.
@param start_index The ordinal number of the first gate.
*/
void list_gates( const double* parameters, int start_index );


/**
@brief Call to reorder the qubits in the matrix of the gates
@param qbit_list The reordered list of qubits spanning the matrix
*/
void reorder_qubits( std::vector<int> qbit_list );


/**
@brief Call to get the qubits involved in the gates stored in the block of gates.
@return Return with a list of the invovled qubits
*/
std::vector<int> get_involved_qubits();

/**
@brief Call to get the gates stored in the class.
@return Return with a list of the gates.
*/
std::vector<Gate*> get_gates();

/**
@brief Call to append the gates of an gate block to the current block
@param op_block A pointer to an instance of class Gate_block
*/
void combine(Gates_block* op_block);


/**
@brief Set the number of qubits spanning the matrix of the gates stored in the block of gates.
@param qbit_num_in The number of qubits spanning the matrices.
*/
void set_qbit_num( int qbit_num_in );


/**
@brief Create a clone of the present class.
@return Return with a pointer pointing to the cloned object.
*/
Gates_block* clone();

/**
@brief Call to extract the gates stored in the class.
@param op_block An instance of Gates_block class in which the gates will be stored. (The current gates will be erased)
@return Return with 0 on success.
*/
int extract_gates( Gates_block* op_block );


};

#endif //GATES_BLOCK

