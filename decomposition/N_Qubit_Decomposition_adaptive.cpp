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
/*! \file N_Qubit_Decomposition_adaptive.cpp
    \brief Base class to determine the decomposition of a unitary into a sequence of two-qubit and one-qubit gate gates.
    This class contains the non-template implementation of the decomposition class
*/

#include "N_Qubit_Decomposition_adaptive.h"
#include "N_Qubit_Decomposition_custom.h"
#include "N_Qubit_Decomposition_Cost_Function.h"


/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive::N_Qubit_Decomposition_adaptive() : N_Qubit_Decomposition_Base() {

    // initialize custom gate structure
    gate_structure = NULL;

}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive::N_Qubit_Decomposition_adaptive( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, guess_type initial_guess_in= CLOSE_TO_ZERO ) : N_Qubit_Decomposition_Base(Umtx_in, qbit_num_in, optimize_layer_num_in, initial_guess_in) {


    // initialize custom gate structure
    gate_structure = NULL;

}



/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_adaptive::~N_Qubit_Decomposition_adaptive() {


    if ( gate_structure != NULL ) {
        // release custom gate structure
        delete gate_structure;
    }

}



/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
@param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
*/
void
N_Qubit_Decomposition_adaptive::start_decomposition(bool prepare_export) {



    if (verbose) {
        printf("***************************************************************\n");
        printf("Starting to disentangle %d-qubit matrix\n", qbit_num);
        printf("***************************************************************\n\n\n");
    }

    // temporarily turn off OpenMP parallelism
#if BLAS==0 // undefined BLAS
    num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#elif BLAS==1 // MKL
    num_threads = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS==2 //OpenBLAS
    num_threads = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif

    //measure the time for the decompositin
    tbb::tick_count start_time = tbb::tick_count::now();


    decomposition_tree_node* parent_node = NULL;
    decomposition_tree_node* minimal_root_node = NULL;
    //std::vector<decomposition_tree_node*> children_nodes;
    tbb::concurrent_vector<decomposition_tree_node*> children;
    tbb::concurrent_vector<matrix_base<double>> optmized_parameters_loc;


while ( current_minimum > optimization_tolerance ) {


    decomposition_tree_node* minimal_node = NULL;
    children.clear();


    // setting the gate structure for optimization
    for ( int target_qbit_loc=0; target_qbit_loc<qbit_num; target_qbit_loc++ ) {
            for ( int control_qbit_loc=target_qbit_loc+1; control_qbit_loc<qbit_num; control_qbit_loc++ ) {

                if ( target_qbit_loc == control_qbit_loc ) continue;

                // reset optimization data
                release_gates();

                if (optimized_parameters != NULL ) {
                    qgd_free( optimized_parameters );
                    parameter_num = 0;
                    optimized_parameters = NULL;
                }

                current_minimum = 1e8;

                Gates_block* gate_structure;
                if ( parent_node != NULL ) {
                    // contruct the new gate structure to be optimized
                    gate_structure = create_layers_from_decomposition_tree( minimal_root_node );
                }
                else {
                    gate_structure = new Gates_block(qbit_num);
                }


                // create the new decomposing layer
                Gates_block* layer = construct_gate_layer(target_qbit_loc, control_qbit_loc);
                gate_structure->add_gate( layer );


                // prepare node to be stored in the decomposition tree
                decomposition_tree_node* current_node = new decomposition_tree_node;
                current_node->layer = layer->clone();


                // add the last layer to rotate all of the qubits into the |0> state
                add_finalyzing_layer( gate_structure );

                //combine( gate_structure );

                // solve the optimization problem in isolated optimization process
                N_Qubit_Decomposition_custom cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, initial_guess);
                cDecomp_custom.set_custom_gate_structure( gate_structure );
                cDecomp_custom.start_decomposition(false);
//cDecomp_custom.list_gates(0);

//exit(-1);
/*
                tbb::task_arena ta(32);
                ta.execute([&]() {
                    // final tuning of the decomposition parameters
                    final_optimization();

                });   
 */
                // save the current minimum to the current node of the decomposition tree
                current_node->cost_func_val = cDecomp_custom.get_current_minimum();
current_minimum = cDecomp_custom.get_current_minimum();

                // store the decomposition tree node
                if (parent_node == NULL) {
                    current_node->parent = NULL;
                    root_nodes.push_back(current_node);
                }
                else {
                    current_node->parent = parent_node;
                    children.push_back(current_node);
                }




                if (current_minimum < optimization_tolerance) {
                    combine( gate_structure );
 
                    if ( optimized_parameters != NULL ) {
                        qgd_free(optimized_parameters);
                    }

                    optimized_parameters = cDecomp_custom.get_optimized_parameters();
delete(gate_structure);
                    break;
                }

delete(gate_structure);


        }

        if (current_minimum < optimization_tolerance) break;


    }



    if (parent_node != NULL) {
        // find the minimal node for the next iteration
        minimal_node = find_minimal_child_node( children );
        parent_node->minimal_child =  minimal_node;
        parent_node->children = children;
    }
    else {
        // find the minimal node for the next iteration
        minimal_node = find_minimal_child_node( root_nodes );
        minimal_root_node = minimal_node;
    }



    parent_node = minimal_node;
if ( current_minimum < minimal_node->cost_func_val ) std::cout << "llllllllllllllllllllllllllllllllllllllllllll" << std::endl;
    current_minimum = minimal_node->cost_func_val;

}


    // prepare gates to export
    if (prepare_export) {
        prepare_gates_to_export();
    }

    // calculating the final error of the decomposition
Matrix_real optimized_parameters_mtx(optimized_parameters, 1, parameter_num );
    Matrix matrix_decomposed = get_transformed_matrix(optimized_parameters_mtx, gates.begin(), gates.size(), Umtx );
    calc_decomposition_error( matrix_decomposed );


    // get the number of gates used in the decomposition
    gates_num gates_num = get_gate_nums();

    if (verbose) {
        std::cout << "In the decomposition with error = " << decomposition_error << " were used " << layer_num << " gates with:" << std::endl;
        if ( gates_num.u3>0 ) std::cout << gates_num.u3 << " U3 opeartions," << std::endl;
        if ( gates_num.rx>0 ) std::cout << gates_num.rx << " RX opeartions," << std::endl;
        if ( gates_num.ry>0 ) std::cout << gates_num.ry << " RY opeartions," << std::endl;
        if ( gates_num.rz>0 ) std::cout << gates_num.rz << " RZ opeartions," << std::endl;
        if ( gates_num.cnot>0 ) std::cout << gates_num.cnot << " CNOT opeartions," << std::endl;
        if ( gates_num.cz>0 ) std::cout << gates_num.cz << " CZ opeartions," << std::endl;
        if ( gates_num.ch>0 ) std::cout << gates_num.ch << " CH opeartions," << std::endl;
        if ( gates_num.x>0 ) std::cout << gates_num.x << " X opeartions," << std::endl;
        if ( gates_num.sx>0 ) std::cout << gates_num.sx << " SX opeartions," << std::endl;
        if ( gates_num.syc>0 ) std::cout << gates_num.syc << " Sycamore opeartions," << std::endl;
        std::cout << std::endl;
        tbb::tick_count current_time = tbb::tick_count::now();
        std::cout << "--- In total " << (current_time - start_time).seconds() << " seconds elapsed during the decomposition ---" << std::endl;
    }

#if BLAS==0 // undefined BLAS
    omp_set_num_threads(num_threads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(num_threads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(num_threads);
#endif

}



/**
@brief ??????????????????
*/
decomposition_tree_node* 
N_Qubit_Decomposition_adaptive::find_minimal_child_node( tbb::concurrent_vector<decomposition_tree_node*> &children ) {

    decomposition_tree_node* minimal_node = NULL;


    double cost_func_val = 1e8;
    for ( tbb::concurrent_vector<decomposition_tree_node*>::iterator it = children.begin(); it != children.end(); it++ ) {
        double cost_func_val_tmp = (*it)->cost_func_val;

        if (cost_func_val_tmp < cost_func_val) {
            cost_func_val = cost_func_val_tmp;
            minimal_node = *it;
        }



    }
std::cout << "ppppppppppppppppppp " << minimal_node->cost_func_val << std::endl;
    return minimal_node;



}






/**
@brief ???????????????
*/
Gates_block*
N_Qubit_Decomposition_adaptive::create_layers_from_decomposition_tree( const decomposition_tree_node* minimal_root_node ) {

    Gates_block* gate_structure = new Gates_block(qbit_num);
//std::cout << "N_Qubit_Decomposition_adaptive::create_layers_from_decomposition_tree 0" << std::endl;
    const decomposition_tree_node* current_node = minimal_root_node;
    while ( current_node != NULL ) {

        Gates_block* layer = current_node->layer->clone();

        // adding the opeartion block to the gates
        gate_structure->add_gate( layer );
   

        if ( current_node->children.size() > 0 ) {
            current_node = current_node->minimal_child;
        }
        else {
            return gate_structure;
        }

    }
//std::cout << "N_Qubit_Decomposition_adaptive::create_layers_from_decomposition_tree 1" << std::endl;
    return gate_structure;

}


/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::construct_gate_layer( const int& _target_qbit, const int& _control_qbit) {


    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );

    // adding U3 gate to the block
    bool Theta = true;
    bool Phi = false;
    bool Lambda = true;
    block->add_u3(_target_qbit, Theta, Phi, Lambda);
    block->add_u3(_control_qbit, Theta, Phi, Lambda);


    // add CNOT gate to the block
    block->add_cnot(_target_qbit, _control_qbit);

    return block;

}



/**
@brief ??????????????????
*/
void 
N_Qubit_Decomposition_adaptive::add_finalyzing_layer( Gates_block* gate_structure ) {


    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );

    // adding U3 gate to the block
    bool Theta = true;
    bool Phi = false;
    bool Lambda = true;

    for (int qbit=0; qbit<qbit_num; qbit++) {
        block->add_u3(qbit, Theta, Phi, Lambda);
    }

    // adding the opeartion block to the gates
    gate_structure->add_gate( block );

}







