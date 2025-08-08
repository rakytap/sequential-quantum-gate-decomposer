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
/*! \file N_Qubit_Decomposition_Tabu_Search.h
    \brief Header file for a class implementing the adaptive gate decomposition algorithm of arXiv:2203.04426
*/

#ifndef N_Qubit_Decomposition_Plywood_H
#define N_Qubit_Decomposition_Plywood_H

#include "N_Qubit_Decomposition_Tree_Search.h"
#include "N_Qubit_Decomposition_Tabu_Search.h"
#include "N_Qubit_Decomposition_adaptive.h"
#include "GrayCode.h"
#include "GrayCodeHash.h"
#include <unordered_set>


/**
@brief A base class to determine the decomposition of an N-qubit unitary into a sequence of CNOT and U3 gates.
This class contains the non-template implementation of the decomposition class.
*/
class N_Qubit_Decomposition_Plywood : public N_Qubit_Decomposition_Tree_Search {


public:

protected:
 
    /// the set of already examined gate structures (mapped to n-ary Gray codes)
    std::unordered_set<GrayCode, GrayCodeHash> tested_gate_structures;
    
    std::vector< std::pair<GrayCode, double> > best_solutions;
    
    int minimum_level;

public:

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_Plywood();


/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param level_limit_in The maximal number of two-qubit gates in the decomposition
@param config std::map conatining custom config parameters
@param accelerator_num The number of DFE accelerators used in the calculations
@return An instance of the class
*/
N_Qubit_Decomposition_Plywood( Matrix Umtx_in, int qbit_num_in, int level_limit_in, std::map<std::string, Config_Element>& config, int accelerator_num=0 );


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
N_Qubit_Decomposition_Plywood( Matrix Umtx_in, int qbit_num_in, int level_limit_in, std::vector<matrix_base<int>> topology_in, std::map<std::string, Config_Element>& config, int accelerator_num=0 );



/**
@brief Destructor of the class
*/
virtual ~N_Qubit_Decomposition_Plywood();



/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
@param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
*/
virtual void start_decomposition() override;

/**
@brief  Call to construct a gate structure corresponding to the configuration of the two-qubit gates described by the Gray code  
@param gcode The N-ary Gray code describing the configuration of the two-qubit gates.
@return Returns with the generated circuit
*/
virtual Gates_block* 
construct_gate_structure_from_Gray_code( const GrayCode& gcode ) override;


};


#endif
