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

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file N_Qubit_Decomposition_Tree_Search.h
    \brief Header file for a class implementing the adaptive gate decomposition algorithm of arXiv:2203.04426
*/

#ifndef N_Qubit_Decomposition_Tree_Search_H
#define N_Qubit_Decomposition_Tree_Search_H

#include "N_Qubit_Decomposition_custom.h"
#include "GrayCode.h"




/**
@brief A base class to determine the decomposition of an N-qubit unitary into a sequence of CNOT and U3 gates.
This class contains the non-template implementation of the decomposition class.
*/
class N_Qubit_Decomposition_Tree_Search : public Optimization_Interface {


public:

protected:


    /// The maximal number of adaptive layers used in the decomposition
    int level_limit;
    /// The minimal number of adaptive layers used in the decomposition
    int level_limit_min;
    /// A vector of index pairs encoding the connectivity between the qubits
    std::vector<matrix_base<int>> topology;
    
    /// List of possible target qubits according to the topology -- paired up with possible control qubits
    matrix_base<int> possible_target_qbits;
    /// List of possible control qubits according to the topology -- paired up with possible target qubits
    matrix_base<int> possible_control_qbits;   
    
    

public:

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_Tree_Search();


/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param level_limit_in The maximal number of two-qubit gates in the decomposition
@param config std::map conatining custom config parameters
@param accelerator_num The number of DFE accelerators used in the calculations
@return An instance of the class
*/
N_Qubit_Decomposition_Tree_Search( Matrix Umtx_in, int qbit_num_in, int level_limit_in, std::map<std::string, Config_Element>& config, int accelerator_num=0 );


/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param level_limit_in The maximal number of two-qubit gates in the decomposition
@param topology_in A list of <target_qubit, control_qubit> pairs describing the connectivity between qubits.
@param config std::map conatining custom config parameters
@param accelerator_num The number of DFE accelerators used in the calculations
@return An instance of the class
*/
N_Qubit_Decomposition_Tree_Search( Matrix Umtx_in, int qbit_num_in, int level_limit_in, std::vector<matrix_base<int>> topology_in, std::map<std::string, Config_Element>& config, int accelerator_num=0 );



/**
@brief Destructor of the class
*/
virtual ~N_Qubit_Decomposition_Tree_Search();


/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
@param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
*/
virtual void start_decomposition();



/**
@brief Call determine the gate structrue of the decomposing circuit. (quantum circuit with CRY gates)
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
virtual Gates_block* determine_gate_structure(Matrix_real& optimized_parameters_mtx);



/**
@brief Call to add two-qubit building block (two single qubit rotation blocks and one two-qubit gate) to the circuit
@param gate_structure Appending the two-qubit building block to this circuit
@param target_qbit The target qubit of the two-qubit gate
@param control_qbit The control qubit of the two-qubit gate
*/
void add_two_qubit_block(Gates_block* gate_structure, int target_qbit, int control_qbit);



/**
@brief  Call to construct a gate structure corresponding to the configuration of the two-qubit gates described by the Gray code  
@param gcode The N-ary Gray code describing the configuration of the two-qubit gates.
@return Returns with the generated circuit
*/
Gates_block* 
construct_gate_structure_from_Gray_code( const GrayCode& gcode );

/**
@brief Call to perform tree search over possible gate structures
@param level_mum The number of decomposing levels (i.e. the maximal tree depth)
@return Returns with the best Gray-code corresponding to the best circuit (The associated gate structure can be costructed by function construct_gate_structure_from_Gray_code)
*/
GrayCode tree_search_over_gate_structures( int level_num );

/**
@brief Call to perform the optimization on the given gate structure
@param gate_structure_loc The gate structure to be optimized
*/
N_Qubit_Decomposition_custom perform_optimization(Gates_block* gate_structure_loc);



/**
@brief Call to add finalyzing layer (single qubit rotations on all of the qubits) to the gate structure.
*/
void add_finalyzing_layer( Gates_block* gate_structure );




/** 
@brief Set unitary matrix 
@param matrix to set unitary to
*/
void set_unitary( Matrix& Umtx_new ) ;


/** 
@brief Perform tabu serach over gate structures
@return Returns with the best Gray-code corresponding to the best circuit (The associated gate structure can be costructed by function construct_gate_structure_from_Gray_code)
*/
GrayCode tabu_search_over_gate_structures();




};






#endif
