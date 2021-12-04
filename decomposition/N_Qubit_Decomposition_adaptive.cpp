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


    Decomposition_Tree_Node* parent_node = NULL;
    Decomposition_Tree_Node* minimal_root_node = NULL;
    tbb::concurrent_vector<matrix_base<double>> optmized_parameters_loc;

    int current_level = 0;
    int level_limit = 6;

    Decomposition_Tree_Node* minimal_node = NULL;

    // mutual exclusion to help modification of vector containers
    tbb::spin_mutex vector_mutex;

//opt_method = 1;
//std::cout << optimization_problem(NULL) << std::endl;
//exit(-1);
    current_minimum = std::numeric_limits<double>::max();
    while ( current_minimum > optimization_tolerance ) {


        minimal_node = NULL;


        // setting the gate structure for optimization
        //for ( int target_qbit_loc=0; target_qbit_loc<qbit_num; target_qbit_loc++ ) {
        tbb::parallel_for( 0, qbit_num, 1, [&](int target_qbit_loc) {
                //for ( int control_qbit_loc=target_qbit_loc+1; control_qbit_loc<qbit_num; control_qbit_loc++ ) {
                tbb::parallel_for(target_qbit_loc, qbit_num, 1, [&](int control_qbit_loc) {
   
                    if ( target_qbit_loc == control_qbit_loc ) return;

                    Gates_block* gate_structure;
                    if ( parent_node != NULL ) {
                        // contruct the new gate structure to be optimized
                        gate_structure = create_layers_from_decomposition_tree( minimal_root_node, current_level );
                    }
                    else {
                        gate_structure = new Gates_block(qbit_num);
                    }


                    // create the new decomposing layer
                    Gates_block* layer = construct_gate_layer(target_qbit_loc, control_qbit_loc);
                    gate_structure->add_gate( layer );


                    // prepare node to be stored in the decomposition tree
                    Decomposition_Tree_Node* current_node = new Decomposition_Tree_Node;
                    current_node->layer = layer->clone();


                    // add the last layer to rotate all of the qubits into the |0> state
                    //add_finalyzing_layer( gate_structure );


                    // solve the optimization problem in isolated optimization process
                    N_Qubit_Decomposition_custom cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, initial_guess);
                    cDecomp_custom.set_custom_gate_structure( gate_structure );
                    cDecomp_custom.set_verbose(false);
                    cDecomp_custom.opt_method = 1;
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
                    current_node->optimized_parameters = cDecomp_custom.get_optimized_parameters();

                    // store the decomposition tree node
                    if (parent_node == NULL) {
//std::cout << current_node->cost_func_val << std::endl;
                        current_node->parent = NULL;
                        {
                            tbb::spin_mutex::scoped_lock my_lock{vector_mutex};
                            root_nodes.push_back(current_node);
                        }
                    }
                    else if (current_node->cost_func_val < 1*parent_node->cost_func_val) {
//std::cout << current_node->cost_func_val << " " << parent_node->cost_func_val << std::endl;
                        current_node->parent = parent_node;
                        {
                            tbb::spin_mutex::scoped_lock my_lock{vector_mutex};
                            parent_node->add_Child( current_node );
                        }
                    }
                    else {
                        delete( current_node );

                    }

                    delete(gate_structure);

            });


        });



        if (parent_node != NULL) {
            // find the minimal node for the next iteration
            minimal_node = parent_node->minimal_child;
        }
        else {
            // find the minimal node for the next iteration
            minimal_node = find_minimal_child_node( root_nodes );
            minimal_root_node = minimal_node;
            minimal_root_node->minimal_child = NULL;
        }

        current_level++;

        if ( minimal_node != NULL ) {
            current_minimum = minimal_node->cost_func_val;
            std::cout << "Decomposing the unitary with " << current_level << " layers found minimum " << current_minimum << std::endl;
        }






        if ( current_minimum < optimization_tolerance ) {
            break;
        }


        // find another path over the decomposition tree if the number of the applied layers is larger than the limit or there is no firther path to follow
        if (current_level >= level_limit ||  minimal_node == NULL) {

            // find the parent of the parent node to select another path over the decomposition tree
            Decomposition_Tree_Node* grand_parent_node = parent_node->parent;

            if ( grand_parent_node == NULL ) {

                // need to chose another root node
                delete_root_node( minimal_root_node );

                minimal_node = find_minimal_child_node( root_nodes );
std::cout << "Find new root node . Remaining root nodes: " << root_nodes.size() << std::endl;

                if ( minimal_node == NULL ) break;

                minimal_root_node = minimal_node;
                current_level = 1;
                current_minimum = minimal_root_node->cost_func_val;
            }
            else {

                grand_parent_node->deactivate_Child(  grand_parent_node->minimal_child );
                current_level = current_level - 1;


                while (current_level>0) {
                    //grand_parent_node->print_active_children();

                    if ( grand_parent_node->active_children == 0 ) {
                        grand_parent_node = grand_parent_node->parent;
                        if ( grand_parent_node == NULL ) {

                            // need to chose another root node
                            delete_root_node( minimal_root_node );

                            minimal_node = find_minimal_child_node( root_nodes );
                            if ( minimal_node == NULL ) break;

                            minimal_root_node = minimal_node;
                            current_level = 1;
                            current_minimum = minimal_root_node->cost_func_val;
std::cout << "Find new root node (b). Remaining root nodes: " << root_nodes.size() << std::endl;
                            break;
                        }
                        else {

                            grand_parent_node->deactivate_Child(  grand_parent_node->minimal_child );
                            current_level = current_level - 1;
                            continue;
                        }

                    }
                          
                    minimal_node = grand_parent_node->minimal_child;
                    break;

                }
            }

        }

        parent_node = minimal_node;

        if ( parent_node == NULL ) {
            break;
        }


    } // while end

    std::cout << "lllll" << std::endl;
    if (minimal_node == NULL ) {
        std::cout << "Decomposition was not successfull." << std::endl;
        return;
    }


    //optimized_parameters_mtx = minimal_node->optimized_parameters;
opt_method = 0;
    release_gates();

    // constructing the decomposing gate structure from decomposition tree
    Gates_block* gate_structure = create_layers_from_decomposition_tree( minimal_root_node, current_level );
    
    // add the last layer to rotate all of the qubits into the |0> state
    add_finalyzing_layer( gate_structure );

    // store the decomposing gate structure
    
    combine( gate_structure );

    // final tuning of the decomposition parameters
    final_optimization();


    // prepare gates to export
    if (prepare_export) {
        prepare_gates_to_export();
    }

    // calculating the final error of the decomposition
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
@brief ???????????????
*/
Gates_block*
N_Qubit_Decomposition_adaptive::create_layers_from_decomposition_tree( const Decomposition_Tree_Node* minimal_root_node, int max_level ) {

    Gates_block* gate_structure = new Gates_block(qbit_num);

    const Decomposition_Tree_Node* current_node = minimal_root_node;
    for ( int level=0; level<max_level; level++ )  {

        Gates_block* layer = current_node->layer->clone();

        // adding the opeartion block to the gates
        gate_structure->add_gate( layer );
//std::cout << "   N_Qubit_Decomposition_adaptive::create_layers_from_decomposition_tree " << current_node->cost_func_val << std::endl;

        if ( current_node->children.size() > 0 ) {
            current_node = current_node->minimal_child;
        }
        else {
            return gate_structure;
        }

    }

    return gate_structure;

}





/**
@brief ??????????????????
*/
Decomposition_Tree_Node* 
N_Qubit_Decomposition_adaptive::find_minimal_child_node( std::vector<Decomposition_Tree_Node*> &children ) {

    Decomposition_Tree_Node* minimal_node = NULL;

    double cost_func_val = std::numeric_limits<double>::max();
    for ( std::vector<Decomposition_Tree_Node*>::iterator it = children.begin(); it != children.end(); it++ ) {

        if ( *it == NULL ) continue;

        double cost_func_val_tmp = (*it)->cost_func_val;

        if (cost_func_val_tmp < cost_func_val) {
            cost_func_val = cost_func_val_tmp;
            minimal_node = *it;
        }


    }

    return minimal_node;



}




/**
@brief ??????????????????
*/
void
N_Qubit_Decomposition_adaptive::delete_root_node( Decomposition_Tree_Node* root_node ) {

    for ( std::vector<Decomposition_Tree_Node*>::iterator it = root_nodes.begin(); it != root_nodes.end(); it++ ) {

        if (root_node == *it) {
            delete(*it);
            root_nodes.erase(it);
            return;
        }


    }

    return;



}








/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::construct_gate_layer( const int& _target_qbit, const int& _control_qbit) {


    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );
/*
    // adding U3 gate to the block
    bool Theta = true;
    bool Phi = false;
    bool Lambda = true;
    block->add_u3(_target_qbit, Theta, Phi, Lambda);
    block->add_u3(_control_qbit, Theta, Phi, Lambda);
*/
    block->add_ry(_target_qbit);
    block->add_ry(_control_qbit);

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
        //block->add_u3(qbit, Theta, Phi, Lambda);
        block->add_ry(qbit);
    }

    // adding the opeartion block to the gates
    gate_structure->add_gate( block );

}







