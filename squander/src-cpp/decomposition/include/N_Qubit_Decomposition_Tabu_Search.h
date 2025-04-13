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

#ifndef N_Qubit_Decomposition_Tabu_Search_H
#define N_Qubit_Decomposition_Tabu_Search_H

#include "N_Qubit_Decomposition_Tree_Search.h"
#include "GrayCode.h"
#include "GrayCodeHash.h"
#include <unordered_set>


/**
@brief A base class to determine the decomposition of an N-qubit unitary into a sequence of CNOT and U3 gates.
This class contains the non-template implementation of the decomposition class.
*/
class N_Qubit_Decomposition_Tabu_Search : public N_Qubit_Decomposition_Tree_Search {


public:

protected:
 
    /// the set of already examined gate structures (mapped to n-ary Gray codes)
    std::unordered_set<GrayCode, GrayCodeHash> tested_gate_structures;
    
    std::vector< std::pair<GrayCode, double> > best_solutions;
    

public:

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_Tabu_Search();


/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@param compression_enabled_in Optional logical value. If True(1) begin decomposition function will compress the circuit. If False(0) it will not. Compression can still be called in seperate wrapper function. 
@return An instance of the class
*/
N_Qubit_Decomposition_Tabu_Search( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::map<std::string, Config_Element>& config, int accelerator_num=0 );


/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@param compression_enabled Optional logical value. If True(1) begin decomposition function will compress the circuit. If False(0) it will not. Compression can still be called in seperate wrapper function. 
@return An instance of the class
*/
N_Qubit_Decomposition_Tabu_Search( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::vector<matrix_base<int>> topology_in, std::map<std::string, Config_Element>& config, int accelerator_num=0 );



/**
@brief Destructor of the class
*/
virtual ~N_Qubit_Decomposition_Tabu_Search();



/**
@brief Call determine the gate structrue of the decomposing circuit. (quantum circuit with CRY gates)
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
Gates_block* determine_gate_structure(Matrix_real& optimized_parameters_mtx);



/** 
@brief Perform tabu serach over gate structures
@return Returns with the best Gray-code corresponding to the best circuit (The associated gate structure can be costructed by function construct_gate_structure_from_Gray_code)
*/
GrayCode tabu_search_over_gate_structures();


/** 
@brief ????
@param ????
@return Returns with the ????
*/
std::vector<GrayCode> determine_mutated_structures( const GrayCode& gcode );



/** 
@brief ????
@param ????
@return Returns with the ????
*/
GrayCode draw_gate_structure_from_list( const std::vector<GrayCode>& gcodes );


/** 
@brief ????
@param ????
@return Returns with the ????
*/
GrayCode mutate_gate_structure( const GrayCode& gcode );

/** 
@brief ????
@param ????
@return Returns with the ????
*/
void insert_into_best_solution( const GrayCode& gcode_, double minimum_ );


};






#endif
