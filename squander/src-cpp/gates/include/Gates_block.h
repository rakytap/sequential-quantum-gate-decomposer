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
/*! \file Gates_block.h
    \brief Header file for a class responsible for grouping gates into subcircuits. (Subcircuits can be nested)
*/

#ifndef GATES_BLOCK_H
#define GATES_BLOCK_H

#include <vector>
#include "common.h"
#include "matrix_real.h"
#include "Gate.h"

#ifdef __DFE__
#include "common_DFE.h"
#endif


/**
@brief A class responsible for grouping two-qubit (CNOT,CZ,CH) and one-qubit gates into layers
*/
class Gates_block :  public Gate {


protected:
    /// The list of stored gates
    std::vector<Gate*> gates;
    /// number of gate layers
    int layer_num;
    
    
    //////// experimental attributes to partition the circuits into subsegments. Advantageous in simulation of larger circuits ///////////űű
    
    /// boolean variable indicating whether the circuit was already partitioned or not
    bool fragmented;
    int fragmentation_type;
    /// maximal number of qubits in partitions
    int max_fusion;
    std::vector< std::vector<int>> involved_qbits;
    std::vector<int> block_end;
    std::vector<int> block_type;
    //////// experimental attributes to partition the circuits into subsegments. Advantageous in simulation of larger circuits ///////////    

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
@brief Call to release one gate in the list
*/
void release_gate( int idx);

/**
@brief Call to retrieve the gate matrix (Which is the product of all the gate matrices stored in the gate block)
@param parameters An array pointing to the parameters of the gates
@return Returns with the gate matrix
*/
Matrix get_matrix( Matrix_real& parameters );

/**
@brief Call to retrieve the gate matrix (Which is the product of all the gate matrices stored in the gate block)
@param parameters An array pointing to the parameters of the gates
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with the gate matrix
*/
Matrix get_matrix( Matrix_real& parameters, int parallel );


/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param inputs The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void apply_to_list( Matrix_real& parameters, std::vector<Matrix>& inputs, int parallel );

/**
@brief Call to apply the gate on the input array/matrix Gates_block*input
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
virtual void apply_to( Matrix_real& parameters_mtx, Matrix& input, int parallel=0 );




/**
@brief Call to apply the gate on the input array/matrix by input*CNOT
@param input The input array on which the gate is applied
*/
virtual void apply_from_right( Matrix_real& parameters_mtx, Matrix& input );


/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP (NOT IMPLEMENTED YET) and 2 for parallel with TBB (optional)
*/
virtual std::vector<Matrix> apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel );




/**
@brief Append a U3 gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param Theta The Theta parameter of the U3 gate
@param Phi The Phi parameter of the U3 gate
@param Lambda The Lambda parameter of the U3 gate
*/
void add_u3(int target_qbit, bool Theta, bool Phi, bool Lambda);

/**
@brief Add a U3 gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param Theta The Theta parameter of the U3 gate
@param Phi The Phi parameter of the U3 gate
@param Lambda The Lambda parameter of the U3 gate
*/
void add_u3_to_front(int target_qbit, bool Theta, bool Phi, bool Lambda);


/**
@brief Append a RX gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_rx(int target_qbit);

/**
@brief Add a RX gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_rx_to_front(int target_qbit);


/**
@brief Append a RY gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_ry(int target_qbit);

/**
@brief Add a RY gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_ry_to_front(int target_qbit);




/**
@brief Append a CRY gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cry(int target_qbit, int control_qbit);

/**
@brief Add a CRY gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cry_to_front(int target_qbit, int control_qbit);

/**
@brief Append a CZ_NU gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cz_nu(int target_qbit, int control_qbit);



/**
@brief Add a CZ_NU gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cz_nu_to_front(int target_qbit, int control_qbit );



/**
@brief Append a RZ gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_rz(int target_qbit);

/**
@brief Add a RZ gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_rz_to_front(int target_qbit);



/**
@brief Append a CNOT gate gate to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cnot( int target_qbit, int control_qbit);



/**
@brief Add a C_NOT gate gate to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cnot_to_front( int target_qbit, int control_qbit );


/**
@brief Append a CZ gate gate to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cz( int target_qbit, int control_qbit);



/**
@brief Add a CZ gate gate to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_cz_to_front( int target_qbit, int control_qbit );


/**
@brief Append a CH gate (i.e. controlled Hadamard gate) gate to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_ch( int target_qbit, int control_qbit);



/**
@brief Add a CH gate (i.e. controlled Hadamard gate) gate to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_ch_to_front( int target_qbit, int control_qbit );

/**
@brief Append a Hadamard gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_h(int target_qbit);

/**
@brief Add a Hadamard gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_h_to_front(int target_qbit);


/**
@brief Append a X gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_x(int target_qbit);

/**
@brief Add a X gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_x_to_front(int target_qbit);


/**
@brief Append a Y gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_y(int target_qbit);

/**
@brief Add a Y gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_y_to_front(int target_qbit);


/**
@brief Append a Z gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_z(int target_qbit);

/**
@brief Add a Z gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_z_to_front(int target_qbit);

/**
@brief Append a SX gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_sx(int target_qbit);

/**
@brief Add a SX gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_sx_to_front(int target_qbit);

/**
@brief Append a Sycamore gate (i.e. controlled Hadamard gate) gate to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_syc( int target_qbit, int control_qbit);



/**
@brief Add a Sycamore gate (i.e. controlled Hadamard gate) gate to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_syc_to_front( int target_qbit, int control_qbit );




/**
@brief Append a UN gate to the list of gates
*/
void add_un();

/**
@brief Add a UN gate to the front of the list of gates
*/
void add_un_to_front();


/**
@brief Append a ON gate to the list of gates
*/
void add_on();

/**
@brief Add a OUN gate to the front of the list of gates
*/
void add_on_to_front();


/**
@brief Append a Composite gate to the list of gates
*/
void add_composite();

/**
@brief Add a Composite gate to the front of the list of gates
*/
void add_composite_to_front();


/**
@brief Append a Adaptive gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_adaptive(int target_qbit, int control_qbit);

/**
@brief Add a Adaptive gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void add_adaptive_to_front(int target_qbit, int control_qbit);


/**
@brief Append a list of gates to the list of gates
@param gates_in A list of gate class instances.
*/
void add_gates( std::vector<Gate*> gates_in );


/**
@brief Add an array of gates to the front of the list of gates
@param gates_in A list of gate class instances.
*/
void add_gates_to_front( std::vector<Gate*> gates_in );


/**
@brief Append a general gate to the list of gates
@param gate A pointer to a class Gate describing an gate.
*/
void add_gate( Gate* gate );

/**
@brief Add an gate to the front of the list of gates
@param gate A pointer to a class Gate describing an gate.
*/
void add_gate_to_front( Gate* gate );


/**
@brief Call to insert a gate at a given position 
@param gate A pointer to a class Gate describing a gate.
@param idx The position where to insert the gate.
*/
void insert_gate( Gate* gate, int idx );



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
void list_gates( const Matrix_real &parameters, int start_index );


/**
@brief Call to reorder the qubits in the matrix of the gates
@param qbit_list The reordered list of qubits spanning the matrix
*/
virtual void reorder_qubits( std::vector<int> qbit_list );


/**
@brief Call to get the qubits involved in the gates stored in the block of gates.
@return Return with a list of the involved qubits
*/
std::vector<int> get_involved_qubits();

/**
@brief Call to get the gates stored in the class.
@return Return with a list of the gates.
*/
std::vector<Gate*> get_gates();


/**
@brief Call to get the gates stored in the class.
@return Return with a list of the gates.
*/
Gate* get_gate(int idx);

/**
@brief Call to append the gates of an gate block to the current block
@param op_block A pointer to an instance of class Gate_block
*/
void combine(Gates_block* op_block);


/**
@brief Set the number of qubits spanning the matrix of the gates stored in the block of gates.
@param qbit_num_in The number of qubits spanning the matrices.
*/
virtual void set_qbit_num( int qbit_num_in );


/**
@brief Create a clone of the present class.
@return Return with a pointer pointing to the cloned object.
*/
virtual Gates_block* clone();

/**
@brief Call to extract the gates stored in the class.
@param op_block An instance of Gates_block class in which the gates will be stored. (The current gates will be erased)
@return Return with 0 on success.
*/
int extract_gates( Gates_block* op_block );



/**
@brief Call to determine, whether the circuit contains daptive gate or not.
@return Return with true if the circuit contains an adaptive gate.
*/
bool contains_adaptive_gate();

/**
@brief Call to determine, whether the sub-circuit at a given position in the circuit contains daptive gate or not.
@param idx The position of the gate to be checked.
@return Return with true if the circuit contains an adaptive gate.
*/
bool contains_adaptive_gate(int idx);



/**
@brief Call to evaluate the reduced densiy matrix.
@param parameters An array of parameters to calculate the entropy
@param input_state The input state on which the gate structure is applied
@param qbit_list Subset of qubits for which the entropy should be calculated. (Should conatin unique elements)
@Return Returns with the reduced density matrix.
*/
Matrix get_reduced_density_matrix( Matrix_real& parameters_mtx, Matrix& input_state, matrix_base<int>& qbit_list_subset );



/**
@brief Call to evaluate the seconf Rényi entropy. The quantum circuit is applied on an input state input. The entropy is evaluated for the transformed state.
@param parameters An array of parameters to calculate the entropy
@param input_state The input state on which the gate structure is applied
@param qbit_list Subset of qubits for which the entropy should be calculated. (Should conatin unique elements)
@Return Returns with the calculated entropy
*/
double get_second_Renyi_entropy( Matrix_real& parameters_mtx, Matrix& input_state, matrix_base<int>& qbit_list );



/**
@brief Call to obtain the parent gates in the circuit. A parent gate needs to be applied prior to the given gate. The parent gates are stored via the "parents" attribute of the gate instance
@param gate The gate for which the parents are determined. 
*/
void determine_parents( Gate* gate );


/**
@brief Call to obtain the child gates in the circuit. A child gate needs to be applied after the given gate. The children gates are stored via the "children" attribute of the gate instance
@param gate The gate for which the children are determined. 
*/
void determine_children( Gate* gate );



//////// experimental attributes to partition the circuits into subsegments. Advantageous in simulation of larger circuits ///////////űű

void fragment_circuit();
void get_parameter_max(Matrix_real &range_max);

//////// experimental attributes to partition the circuits into subsegments. Advantageous in simulation of larger circuits ///////////űű

#ifdef __DFE__

/**
@brief Call to convert gate data (and derivate gate data) to DFE data structure
@return 
*/
DFEgate_kernel_type* convert_to_DFE_gates_with_derivates( Matrix_real& parameters_mtx, int& gatesNum, int& gateSetNum, int& redundantGateSets, bool only_derivates=false );

/**
@brief Call to adjust DFE gate paremeters to evaluate the derivates
@return 
*/
void adjust_parameters_for_derivation( DFEgate_kernel_type* DFEgates, const int  gatesNum, int& gate_idx, int& gate_set_index );

/**
@brief Call to convert gate data to batched DFE data structure
@return 
*/
DFEgate_kernel_type* convert_to_batched_DFE_gates( std::vector<Matrix_real>& parameters_mtx_vec, int& gatesNum, int& gateSetNum, int& redundantGateSets );



/**
@brief Call to convert gate data to DFE data structure
@return 
*/
DFEgate_kernel_type* convert_to_DFE_gates( Matrix_real& parameters_mtx, int& gatesNum );

/**
@brief Call to convert gate data to DFE data structure
@return 
*/
void convert_to_DFE_gates( const Matrix_real& parameters_mtx, DFEgate_kernel_type* DFEgates, int& start_index );

void get_matrices_target_control(std::vector<Matrix> &u3_qbit, std::vector<int> &target_qbit, std::vector<int> &control_qbit, Matrix_real& parameters_mtx);

#endif


/**
@brief Method to reset the parameter start indices of gate operations incorporated in the circuit. (When a gate is inserted into the circuit at other position than the end.)
*/
void reset_parameter_start_indices();



/**
@brief Method to reset the dependency graph of the gates in the circuit
*/
void reset_dependency_graph();


/**
@brief Method to generate a flat circuit. A flat circuit is a circuit does not containing subcircuits: there are no Gates_block instances (containing subcircuits) in the resulting circuit. If the original circuit contains subcircuits, the gates in the subcircuits are directly incorporated in the resulting flat circuit.
*/
Gates_block* get_flat_circuit();


/**
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is embedded.
@param parameters The parameter array corresponding to the circuit in which the gate is embedded
@return Returns with the array of the extracted parameters.
*/
virtual Matrix_real extract_parameters( Matrix_real& parameters );

};



/**
@brief ?????????
@return Return with ?????????
*/
void export_gate_list_to_binary(Matrix_real& parameters, Gates_block* gates_block, const std::string& filename, int verbosity=3);

/**
@brief ?????????
@return Return with ?????????
*/
void export_gate_list_to_binary(Matrix_real& parameters, Gates_block* gates_block, FILE* pFile, int verbosity=3);


/**
@brief ?????????
@return Return with ?????????
*/
Gates_block* import_gate_list_from_binary(Matrix_real& parameters, const std::string& filename, int verbosity=3);


/**
@brief ?????????
@return Return with ?????????
*/
Gates_block* import_gate_list_from_binary(Matrix_real& parameters, FILE* pFile, int verbosity=3);

//////// experimental attributes to partition the circuits into subsegments. Advantageous in simulation of larger circuits ///////////
bool is_qbit_present(std::vector<int> involved_qbits, int new_qbit, int num_of_qbits);
//////// experimental attributes to partition the circuits into subsegments. Advantageous in simulation of larger circuits ///////////



/**
@brief Call to reverse the order of the parameters in an array
@param v_in The real input vector.
@param gates The associated list of gates. (does not get reversed)
@return Return with the reversed array
*/
Matrix_real reverse_parameters( const Matrix_real& v_in, std::vector<Gate*>::iterator gates_it, int num_of_gates );


/**
@brief Call to inverse-reverse the order of the parameters in an array
@param v_in The real input vector.
@param gates The associated list of gates. (does not get reversed)
@return Return with the reversed array
*/
Matrix_real inverse_reverse_parameters( const Matrix_real& v_in, std::vector<Gate*>::iterator gates_it, int num_of_gates );

#endif //GATES_BLOCK,

