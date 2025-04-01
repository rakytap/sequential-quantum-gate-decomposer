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
/*! \file N_Qubit_Decomposition_non_unitary_adaptive.h
    \brief Header file for a class implementing the adaptive gate decomposition algorithm of arXiv:2203.04426
*/

#ifndef N_Qubit_Decomposition_non_unitary_adaptive_H
#define N_Qubit_Decomposition_non_unitary_adaptive_H

#include "N_Qubit_Decomposition_custom.h"

#ifdef __cplusplus
extern "C" 
{
#endif

/// Definition of the zggev function from Lapacke to calculate the eigenvalues of a complex matrix
int LAPACKE_zggev 	( 	int  	matrix_layout,
		char  	jobvl,
		char  	jobvr,
		int  	n,
		QGD_Complex16 *  	a,
		int  	lda,
		QGD_Complex16 *  	b,
		int  	ldb,
		QGD_Complex16 *  	alpha,
		QGD_Complex16 *  	beta,
		QGD_Complex16 *  	vl,
		int  	ldvl,
		QGD_Complex16 *  	vr,
		int  	ldvr 
	); 	

#ifdef __cplusplus
}
#endif


/**
@brief A base class to determine the decomposition of an N-qubit unitary into a sequence of CNOT and U3 gates.
This class contains the non-template implementation of the decomposition class.
*/
class N_Qubit_Decomposition_non_unitary_adaptive : public Optimization_Interface {


public:

protected:


    /// The maximal number of adaptive layers used in the decomposition
    int level_limit;
    /// The minimal number of adaptive layers used in the decomposition
    int level_limit_min;
    /// A vector of index pairs encoding the connectivity between the qubits
    std::vector<matrix_base<int>> topology;
    /// Boolean variable to determine whether randomized adaptive layers are used or not
    bool randomized_adaptive_layers;
    
    

public:

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_non_unitary_adaptive();


/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@param compression_enabled_in Optional logical value. If True(1) begin decomposition function will compress the circuit. If False(0) it will not. Compression can still be called in seperate wrapper function. 
@return An instance of the class
*/
N_Qubit_Decomposition_non_unitary_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::map<std::string, Config_Element>& config, int accelerator_num=0 );


/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@param compression_enabled Optional logical value. If True(1) begin decomposition function will compress the circuit. If False(0) it will not. Compression can still be called in seperate wrapper function. 
@return An instance of the class
*/
N_Qubit_Decomposition_non_unitary_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::vector<matrix_base<int>> topology_in, std::map<std::string, Config_Element>& config, int accelerator_num=0 );



/**
@brief Destructor of the class
*/
virtual ~N_Qubit_Decomposition_non_unitary_adaptive();


/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
@param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
*/
virtual void start_decomposition();

/**
@brief get initial circuit
*/
virtual void get_initial_circuit();


/**
@brief Finalize the circuit
@param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
*/
virtual void finalize_circuit();

/**
@brief Call to optimize an imported gate structure
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
Gates_block* optimize_imported_gate_structure(Matrix_real& optimized_parameters_mtx_loc);


/**
@brief Call determine the gate structrue of the decomposing circuit. (quantum circuit with CRY gates)
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
Gates_block* determine_initial_gate_structure(Matrix_real& optimized_parameters_mtx);



/**
@brief ?????????????????????????
*/
void add_two_qubit_block(Gates_block* gate_structure, int target_qbit, int control_qbit);

/**
@brief C????????????????????????????????
@param gate_structure The gate structure to be optimized
@param optimized_parameters A matrix containing the initial parameters
*/
Gates_block* tree_search_over_gate_structures( int level_max );

/**
@brief C????????????????????????????????
@param gate_structure The gate structure to be optimized
@param optimized_parameters A matrix containing the initial parameters
*/
N_Qubit_Decomposition_custom perform_optimization(Gates_block* gate_structure_loc);


/**
@brief Call to replace CZ_NU gates in the circuit that are close to either an identity or to a CNOT gate.
@param gate_structure The gate structure to be optimized
@param optimized_parameters A matrix containing the initial parameters
*/
Gates_block* replace_trivial_CZ_NU_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters );


/**
@brief Call to add adaptive layers to the gate structure stored by the class.
*/
void add_adaptive_layers();

/**
@brief Call to add adaptive layers to the gate structure.
*/
void add_adaptive_layers( Gates_block* gate_structure );

/**
@brief Call to construct adaptive layers.
*/
Gates_block* construct_adaptive_gate_layers();


/**
@brief Call to add finalyzing layer (single qubit rotations on all of the qubits) to the gate structure stored by the class.
*/
void add_finalyzing_layer();


/**
@brief Call to add finalyzing layer (single qubit rotations on all of the qubits) to the gate structure.
*/
void add_finalyzing_layer( Gates_block* gate_structure );



/**
@brief Call to set custom layers to the gate structure that are intended to be used in the decomposition.
@param filename
*/
void set_adaptive_gate_structure( std::string filename );
 
 
 /**
 @brief Set unitary matrix from file
 @param filename file to read unitary from
 */
 void set_unitary_from_file( std::string filename );


 /** 
 @brief Set unitary matrix 
 @param matrix to set unitary to
 */
 void set_unitary( Matrix& Umtx_new ) ;

/**
@brief Call to append custom layers to the gate structure that are intended to be used in the decomposition.
@param filename
*/
void add_adaptive_gate_structure( std::string filename );

/**
@brief Call to apply the imported gate structure on the unitary. The transformed unitary is to be decomposed in the calculations, and the imported gfate structure is released.
*/
void apply_imported_gate_structure();


/**
@brief Call to add an adaptive layer to the gate structure previously imported gate structure
@param filename
*/
void add_layer_to_imported_gate_structure();



/**
@brief 
@param 
@param 
*/
Gates_block* 
CZ_nu_search( int level_max );

};






#endif
