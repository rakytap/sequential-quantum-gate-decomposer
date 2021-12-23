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
#include "N_Qubit_Decomposition_adaptive_Cost_Function.h"
#include "Random_Orthogonal.h"

#include <time.h>
#include <stdlib.h>


Matrix_real create_random_paramaters( Gates_block* gate_structure ) {

    int parameter_num = gate_structure->get_parameter_num();

    Matrix_real parameters(1, parameter_num);

    for(int idx = 0; idx < parameter_num; idx++) {
         if ( idx % 5 == 0 ) {
             if ( rand() % 2 == 0 ) {
                 parameters[idx] = 0.0;
             }
             else {
                 parameters[idx] = M_PI;
             }
         }
         else {
             parameters[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
         }
    }

    return parameters;


}

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive::N_Qubit_Decomposition_adaptive() : N_Qubit_Decomposition_Base() {

    // initialize custom gate structure
    gate_structure = NULL;

    // set the level limit
    level_limit = 0;

    // set decomposition_iterations
    decomposition_iterations = 0;

    srand(time(NULL));   // Initialization, should only be called once.
}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive::N_Qubit_Decomposition_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, guess_type initial_guess_in= CLOSE_TO_ZERO ) : N_Qubit_Decomposition_Base(Umtx_in, qbit_num_in, false, initial_guess_in) {


    // initialize custom gate structure
    gate_structure = NULL;

    // set the level limit
    level_limit = level_limit_in;
    level_limit_min = level_limit_min_in;

    // set decomposition_iterations
    decomposition_iterations = 4;

    srand(time(NULL));   // Initialization, should only be called once.
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

    if (level_limit == 0 ) {
        std::cout << "please increase level limit" << std::endl;
        exit(-1);
    }

    Decomposition_Tree_Node* minimal_node = NULL;

    // mutual exclusion to help modification of vector containers
    tbb::spin_mutex vector_mutex;

/*
opt_method = 1;
std::cout << optimization_problem(NULL) << std::endl;

CRY cry_gate(3,0,2);
Matrix_real params(1,1);
params[0] = 1*M_PI;
Matrix mtx = cry_gate.get_matrix( params );
mtx.print_matrix();
//exit(-1);
*/
/*
Random_Orthogonal ro(8);
Matrix O = ro.Construct_Orthogonal_Matrix();
//O.print_matrix();
Matrix O2 = O.copy();
O2.transpose();
Matrix O3 = dot( O, O2 );
O3.print_matrix();
exit(-1);
*/

/*
    Gates_block* gate_structure = new Gates_block(qbit_num);
    add_static_gate_layers_1( gate_structure );
    add_static_gate_layers_2( gate_structure );
    opt_method = 1;

    // add the last layer to rotate all of the qubits into the |0> state
    //add_finalyzing_layer( gate_structure );

    // store the decomposing gate structure    
    combine( gate_structure );

    // finalizing the decompostition
    //finalize_decomposition();

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
        if ( gates_num.un>0 ) std::cout << gates_num.un << " UN opeartions," << std::endl;
        std::cout << std::endl;
        tbb::tick_count current_time = tbb::tick_count::now();
        std::cout << "--- In total " << (current_time - start_time).seconds() << " seconds elapsed during the decomposition ---" << std::endl;
    }

delete(gate_structure);
*/

iteration_loops.insert({4, 2});
iteration_loops.insert({5, 2});
decomposition_iterations = 4;
    opt_method = 1;
    
    Gates_block* gate_structure = new Gates_block(qbit_num);

    // The first layer that does not undergo optimization
    for (int idx=0; idx<level_limit_min-1; idx++) {

        // create the new decomposing layer
        Gates_block* layer = construct_gate_layer(0,0);
        gate_structure->combine( layer );

    }
std::cout << "level lmit min: " << level_limit_min << std::endl;
    int level = level_limit_min;
    optimization_tolerance = optimization_tolerance*100;
    while ( current_minimum > optimization_tolerance && level < level_limit) {

        // reset optimized parameters
        optimized_parameters_mtx = Matrix_real(1,1);

        // create the new decomposing layer
        Gates_block* layer = construct_gate_layer(0,0);
        gate_structure->combine( layer );

        // store the decomposing gate structure    
        //combine( gate_structure );


        // final tuning of the decomposition parameters
        //final_optimization();
      
        N_Qubit_Decomposition_custom cDecomp_custom;
        for (int iter=0; iter < decomposition_iterations; iter++) {



//Matrix_real parameters_initial = create_random_paramaters(gate_structure);

std::cout << optimization_block << std::endl;
            // solve the optimization problem in isolated optimization process
            cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, initial_guess);
            cDecomp_custom.set_custom_gate_structure( gate_structure );
            //cDecomp_custom.set_optimized_parameters( parameters_initial.get_data(), parameters_initial.size() );
            cDecomp_custom.set_optimization_blocks( optimization_block );
            //cDecomp_custom.set_verbose(false);
            cDecomp_custom.opt_method = 1;
            cDecomp_custom.set_iteration_loops( iteration_loops );
            cDecomp_custom.set_optimization_tolerance( optimization_tolerance );  
            cDecomp_custom.start_decomposition(true);
            //cDecomp_custom.list_gates(0);

            if ( cDecomp_custom.get_current_minimum() < optimization_tolerance ) {
                std::cout << "Optimization problem solved with " << level+1 << " decomposing levels" << std::endl;
                cDecomp_custom.list_gates(0);
                break;
            }

        }

        // exctract the optimized parameters
        optimized_parameters_mtx = cDecomp_custom.get_optimized_parameters();

        // get the current minimum of the optimization
        current_minimum = cDecomp_custom.get_current_minimum();

        std::cout << "Optimization problem converged to " << current_minimum << " with " << level+1 << " decomposing levels" << std::endl;

        level++;
    }

//////////////////////////////////////

        //Gates_block* gate_structure_adaptive = exchange_CRZ_to_Adaptive_gates( gate_structure, optimized_parameters_mtx );
        //delete( gate_structure );
        //gate_structure = gate_structure_adaptive;
        //gate_structure_adaptive = NULL;

        for (int iter=0; iter < decomposition_iterations; iter++) {
  
            Matrix_real optimized_parameters_tmp = optimized_parameters_mtx.copy();

            for (int idx=0; idx<optimized_parameters_mtx.size(); idx+=5) {  
                double param = optimized_parameters_mtx[idx];

                param = param + M_PI;
                param = (1.0-std::cos(param/2))*M_PI;
                param = param - M_PI;

                //param = 0.5*(1.0-std::cos(param))*M_PI;
                optimized_parameters_tmp[idx] = param;
            }

            N_Qubit_Decomposition_custom cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, initial_guess);
            cDecomp_custom.set_custom_gate_structure( gate_structure );
            cDecomp_custom.set_optimized_parameters( optimized_parameters_tmp.get_data(), optimized_parameters_tmp.size() );
            cDecomp_custom.set_verbose(false);
            cDecomp_custom.opt_method = 1;
            cDecomp_custom.set_iteration_loops( iteration_loops );
            cDecomp_custom.set_optimization_blocks(optimization_block);
            cDecomp_custom.set_optimization_tolerance( optimization_tolerance );
            cDecomp_custom.start_decomposition(true);
            //current_minimum = cDecomp_custom.get_current_minimum();

            if ( cDecomp_custom.get_current_minimum() < optimization_tolerance ) {
                current_minimum = cDecomp_custom.get_current_minimum();
                cDecomp_custom.list_gates(0);
                optimized_parameters_mtx = cDecomp_custom.get_optimized_parameters();
            }

        }

/////////////////////////////////////////////////////////



    if (current_minimum > optimization_tolerance) {
        std::cout << "Decomposition needs more laxers. Exiting" << std::endl;
        return;
    }

    std::cout << "**************************************************************" << std::endl;
    std::cout << "***************** Compressing Gate structure *****************" << std::endl;
    std::cout << "**************************************************************" << std::endl;

    for ( int iter=0; iter<25; iter++ ) {
        std::cout << "iteration: " << iter+1 << std::endl;
        Gates_block* gate_structure_compressed = compress_gate_structure( gate_structure );

        if ( gate_structure_compressed != gate_structure ) {
            delete( gate_structure );
            gate_structure = gate_structure_compressed;
            gate_structure_compressed = NULL;
        }

    }

    optimization_tolerance = optimization_tolerance/100;

    opt_method = 1;
    // store the decomposing gate structure    
    combine( gate_structure );

    for (int iter=0; iter < decomposition_iterations; iter++) {
  

        for (int idx=0; idx<optimized_parameters_mtx.size(); idx+=5) {  
            double param = optimized_parameters_mtx[idx];

            param = param + M_PI;
            param = (1.0-std::cos(param/2))*M_PI;
            param = param - M_PI;

            //param = 0.5*(1.0-std::cos(param))*M_PI;
            optimized_parameters_mtx[idx] = param;
        }

        current_minimum = optimization_problem(optimized_parameters_mtx.get_data());

        // final tuning of the decomposition parameters
        final_optimization();

    }

std::cout << optimization_problem(optimized_parameters_mtx.get_data()) << std::endl;

    for (int idx=0; idx<optimized_parameters_mtx.size(); idx+=5) {  
        double param = optimized_parameters_mtx[idx];
/*
        param = param + M_PI;
        param = (1.0-std::cos(param/2))*M_PI;
        param = param - M_PI;
*/
        param = activation_function(param);
        param = activation_function(param);
        //param = 0.5*(1.0-std::cos(param))*M_PI;
        optimized_parameters_mtx[idx] = param;
    }


    Gates_block* gate_structure_tmp = replace_trivial_CRY_gates( gate_structure, optimized_parameters_mtx );
    Matrix_real optimized_parameters_save = optimized_parameters_mtx;

    release_gates();
    optimized_parameters_mtx = optimized_parameters_save;

    combine( gate_structure_tmp );
    delete( gate_structure_tmp );
    delete( gate_structure );
std::cout << optimization_problem(optimized_parameters_mtx.get_data()) << std::endl;
    // final tuning of the decomposition parameters
    opt_method = 1;
    final_optimization();

    opt_method = 0;

    // finalizing the decompostition
    finalize_decomposition();

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
        if ( gates_num.un>0 ) std::cout << gates_num.un << " UN opeartions," << std::endl;
        if ( gates_num.cry>0 ) std::cout << gates_num.cry << " CRY opeartions," << std::endl;
        if ( gates_num.adap>0 ) std::cout << gates_num.adap << " Adaptive opeartions," << std::endl;
        std::cout << std::endl;
        tbb::tick_count current_time = tbb::tick_count::now();
        std::cout << "--- In total " << (current_time - start_time).seconds() << " seconds elapsed during the decomposition ---" << std::endl;
    }


/*

opt_method = 0;
    release_gates();

    optimized_parameters_mtx = minimal_node->optimized_parameters;

    Gates_block* gate_structure = new Gates_block(qbit_num);

    // add statically optimized gate structure
    add_static_gate_layers_1( gate_structure );

    // constructing the decomposing gate structure from decomposition tree
    create_layers_from_decomposition_tree( minimal_root_node, current_level, gate_structure );

    // add statically optimized gate structure
    add_static_gate_layers_2( gate_structure );
    
    // add the last layer to rotate all of the qubits into the |0> state
    add_finalyzing_layer( gate_structure );

    // store the decomposing gate structure    
    combine( gate_structure );

    // finalizing the decompostition
    //finalize_decomposition();

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
        if ( gates_num.un>0 ) std::cout << gates_num.un << " UN opeartions," << std::endl;
        std::cout << std::endl;
        tbb::tick_count current_time = tbb::tick_count::now();
        std::cout << "--- In total " << (current_time - start_time).seconds() << " seconds elapsed during the decomposition ---" << std::endl;
    }
*/
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
N_Qubit_Decomposition_adaptive::compress_gate_structure( Gates_block* gate_structure ) {

    int layer_num_max = 20;
    int layer_num_orig = gate_structure->get_gate_num();

    // create a list of layers to be tested for removal.
    std::vector<int> layers_to_remove;
    layers_to_remove.reserve(layer_num_orig);
    for (int idx=0; idx<layer_num_orig; idx++ ) {
        layers_to_remove.push_back(idx);
    }   

    while ( layers_to_remove.size() > layer_num_max ) {
        int remove_idx = rand() % layers_to_remove.size();
        layers_to_remove.erase( layers_to_remove.begin() + remove_idx );
    }


    int panelties_num = layer_num_max < layer_num_orig ? layer_num_max : layer_num_orig;

    // preallocate panelties associated with the number of remaining two-qubit controlled gates
    matrix_base<int> panelties(1,panelties_num);
    std::vector<Gates_block*> gate_structures_vec(panelties_num);
    std::vector<Matrix_real> optimized_parameters_vec(panelties_num);
    for (int idx=0; idx<layer_num_max; idx++ ) {
        gate_structures_vec.push_back(NULL);
        optimized_parameters_vec.push_back(Matrix_real(0,0));
    }



tbb::parallel_for( 0, panelties_num, 1, [&](int idx_to_remove) {
//for (int idx_to_remove=0; idx_to_remove<layer_num_orig; idx_to_remove++) {

    // check whether the gate to be removed conatins adaptive controlled gate
    //if (!gate_structure->contains_adaptive_gate( idx_to_remove ) ) continue;//return gate_structure;

        Matrix_real optimized_parameters_loc = optimized_parameters_mtx.copy();
        Gates_block* gate_structure_reduced = compress_gate_structure( gate_structure, optimized_parameters_loc, layers_to_remove[idx_to_remove] );
 
        // remove further adaptive gates if possible
        Gates_block* gate_structure_tmp = gate_structure_reduced->clone();//remove_trivial_gates( gate_structure_reduced, optimized_parameters_loc );
        panelties[idx_to_remove] = get_panelty(gate_structure_tmp, optimized_parameters_loc);
        gate_structures_vec[idx_to_remove] = gate_structure_tmp;
        optimized_parameters_vec[idx_to_remove] = optimized_parameters_loc;

        

        delete(gate_structure_reduced);


//}
});    

panelties.print_matrix();

    // determine the reduction with the lowest penalty
    int panelty_min = panelties[0];
    int idx_min = 0;
    for (int idx=0; idx<panelties.size(); idx++) {
        if ( panelty_min > panelties[idx] ) {
            panelty_min = panelties[idx];
            idx_min = idx;
        }
        else if ( panelty_min == panelties[idx] ) {

            // randomly choose the solution between identical penalties
            if ( (rand() % 2) == 1 ) {
                panelty_min = panelties[idx];
                idx_min = idx;
            }

        }
    }

    for (int idx=0; idx<panelties.size(); idx++) {
        if (idx==idx_min) continue;
        if ( gate_structures_vec[idx] == gate_structure) continue;
        delete( gate_structures_vec[idx] );
        gate_structures_vec[idx] = NULL;
    }


    // release the reduced gate structure, keep only the most efficient one
    gate_structure = gate_structures_vec[idx_min];
    optimized_parameters_mtx = optimized_parameters_vec[idx_min];

    int layer_num = gate_structure->get_gate_num();

    if ( layer_num < layer_num_orig ) {
        std::cout << "gate structure reduced to " << layer_num << " layers" << std::endl;
    }
    else {
        std::cout << "gate structure kept at " << layer_num << " layers" << std::endl;
    }


    return gate_structure;



}

/**
@brief ???????????????
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::compress_gate_structure( Gates_block* gate_structure, Matrix_real& optimized_parameters, int layer_idx ) {

    // create reduced gate structure without layer indexed by layer_idx
    Gates_block* gate_structure_reduced = gate_structure->clone();
    gate_structure_reduced->release_gate( layer_idx );
        
    Matrix_real&& parameters_reduced = create_reduced_parameters( gate_structure_reduced, optimized_parameters, layer_idx );

    N_Qubit_Decomposition_custom cDecomp_custom;
       
    // solve the optimization problem in isolated optimization process
    cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, initial_guess);
    cDecomp_custom.set_custom_gate_structure( gate_structure_reduced );
    cDecomp_custom.set_optimized_parameters( parameters_reduced.get_data(), parameters_reduced.size() );
    cDecomp_custom.set_verbose(false);
    cDecomp_custom.opt_method = 1;
    cDecomp_custom.set_iteration_loops( iteration_loops );
    cDecomp_custom.set_optimization_blocks(optimization_block);
    cDecomp_custom.set_optimization_tolerance( optimization_tolerance );
    cDecomp_custom.start_decomposition(true);
    current_minimum = cDecomp_custom.get_current_minimum();

    if ( current_minimum < optimization_tolerance ) {
        //cDecomp_custom.list_gates(0);
        optimized_parameters = cDecomp_custom.get_optimized_parameters();

        for (int iter=0; iter < decomposition_iterations; iter++) {
  
            Matrix_real optimized_parameters_tmp(1, optimized_parameters.size() );

            for (int idx=0; idx<optimized_parameters.size(); idx+=5) {  
                double param = optimized_parameters[idx];

                param = param + M_PI;
                param = (1.0-std::cos(param/2))*M_PI;
                param = param - M_PI;

                //param = activation_function(param);

                //param = 0.5*(1.0-std::cos(param))*M_PI;
                optimized_parameters_tmp[idx] = param;
            }

            cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, initial_guess);
            cDecomp_custom.set_custom_gate_structure( gate_structure_reduced );
            cDecomp_custom.set_optimized_parameters( optimized_parameters_tmp.get_data(), parameters_reduced.size() );
            cDecomp_custom.set_verbose(false);
            cDecomp_custom.opt_method = 1;
            cDecomp_custom.set_iteration_loops( iteration_loops );
            cDecomp_custom.set_optimization_blocks(optimization_block);
            cDecomp_custom.set_optimization_tolerance( optimization_tolerance*100 );
            cDecomp_custom.start_decomposition(true);
            current_minimum = cDecomp_custom.get_current_minimum();

            if ( current_minimum < optimization_tolerance ) {

                //cDecomp_custom.list_gates(0);
                optimized_parameters = cDecomp_custom.get_optimized_parameters();
            }

        }



        return gate_structure_reduced;
    }


    return gate_structure->clone();

}


/**
@brief ???????????????
*/
int 
N_Qubit_Decomposition_adaptive::get_panelty( Gates_block* gate_structure, Matrix_real& optimized_parameters ) {


    int panelty = 0;

    // iterate over the elements of tha parameter array
    int parameter_idx = 0;
    for ( int idx=0; idx<gate_structure->get_gate_num(); idx++) {

        double parameter = optimized_parameters[parameter_idx];
 
        
        if ( std::sin(parameter) < 1e-2 && (std::cos(parameter) + 1) < 1e-2 ) {
            // Condition of pure CNOT gate
            panelty += 2;
        }
        else if ( std::sin(parameter) < 1e-2 && (1-std::cos(parameter) ) < 1e-2 ) {
            // Condition of pure Identity gate
            panelty++;
        }
        else {
            // Condition of controlled rotation gate
            panelty += 4;
        }

        Gate* gate = gate_structure->get_gate( idx );
        parameter_idx += gate->get_parameter_num();

    }

    return panelty;


}


/**
@brief ???????????????
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::replace_trivial_CRY_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters ) {


    Gates_block* gate_structure_ret = new Gates_block(qbit_num);

    int layer_num = gate_structure->get_gate_num();

    Matrix_real optimized_parameters_new(1, (int)(1.4*optimized_parameters.size()));
    memset( optimized_parameters_new.get_data(), 0.0, optimized_parameters_new.size()*sizeof(double) );

    int parameter_idx_orig = 0;
    int parameter_idx_new  = 0;
    for (int idx=0; idx<layer_num; idx++ ) {

        Gate* gate = gate_structure->get_gate(idx);

        if ( gate->get_type() != BLOCK_OPERATION ) {
            std::cout << "N_Qubit_Decomposition_adaptive::replace_trivial_adaptive_gates: Only block gates are accepted in this conversion." << std::endl;
            exit(-1);
        }

        Gates_block* block_op = static_cast<Gates_block*>(gate);
        int param_num = gate->get_parameter_num();


        if (  gate_structure->contains_adaptive_gate(idx) ) {

                Gates_block* layer = block_op->clone();

                for ( int idx=0; idx<layer->get_gate_num(); idx++ ) {

                    Gate* gate_tmp = layer->get_gate(idx);
                    param_num = gate_tmp->get_parameter_num();


                    double* param = optimized_parameters.get_data() + parameter_idx_orig;
//std::cout << param[0] << " " << (gate_tmp->get_type() == ADAPTIVE_OPERATION) << " "  << std::abs(std::sin(param[0])) << " "  << 1+std::cos(param[0]) << std::endl;

                    if ( gate_tmp->get_type() == ADAPTIVE_OPERATION &&  std::abs(std::sin(param[0])) < 1e-3 && std::abs(1+std::cos(param[0])) < 1e-3 ) {
                        // convert to CZ gate
  
                        int target_qbit = gate_tmp->get_target_qbit();
                        int control_qbit = gate_tmp->get_control_qbit();
                        layer->release_gate( idx );

                        CZ* cz_gate = new CZ(qbit_num, target_qbit, control_qbit);
                        layer->insert_gate( (Gate*)cz_gate, idx);

                    }
                    else if ( gate_tmp->get_type() == ADAPTIVE_OPERATION &&  std::abs(std::sin(param[0])) < 1e-4 && std::abs(1-std::cos(param[0])) < 1e-4  ) {
                        // release trivial gate  
                        layer->release_gate( idx );


                    }
                    else if ( gate_tmp->get_type() == ADAPTIVE_OPERATION ) {
                        // controlled Z rotation decomposed into 2 CNOT gates

                        int target_qbit = gate_tmp->get_target_qbit();
                        int control_qbit = gate_tmp->get_control_qbit();
                        layer->release_gate( idx );

                        RZ*   rz_gate_1   = new RZ(qbit_num, target_qbit);
                        CNOT* cnot_gate_1 = new CNOT(qbit_num, target_qbit, control_qbit);
                        RZ*   rz_gate_2   = new RZ(qbit_num, target_qbit);
                        CNOT* cnot_gate_2 = new CNOT(qbit_num, target_qbit, control_qbit);
                        RZ*   rz_gate_3   = new RZ(qbit_num, control_qbit);

                        Gates_block* czr_gate = new Gates_block(qbit_num);
                        czr_gate->add_gate(rz_gate_1);
                        czr_gate->add_gate(cnot_gate_1);
                        czr_gate->add_gate(rz_gate_2);
                        czr_gate->add_gate(cnot_gate_2);
                        czr_gate->add_gate(rz_gate_3);

                        layer->insert_gate( (Gate*)czr_gate, idx);

                        double param_tmp = param[0]; 

                        optimized_parameters_new[parameter_idx_new++] = param_tmp/2; // rz_3 parameter
                        optimized_parameters_new[parameter_idx_new++] = -param_tmp/2; // rz_2 parameter
                        optimized_parameters_new[parameter_idx_new++] = param_tmp/2; // rz_1 parameter


                    }
                    else {
                        memcpy( optimized_parameters_new.get_data()+parameter_idx_new, optimized_parameters.get_data()+parameter_idx_orig, param_num*sizeof(double) );
                        parameter_idx_new  += param_num;
                    }

                    parameter_idx_orig += param_num;


                }

                gate_structure_ret->add_gate_to_end((Gate*)layer);

                
 
                continue;
            

        }



        Gates_block* block_op_cloned = block_op->clone();

        gate_structure_ret->add_gate_to_end((Gate*)block_op_cloned);
        memcpy( optimized_parameters_new.get_data()+parameter_idx_new, optimized_parameters.get_data()+parameter_idx_orig, param_num*sizeof(double) );

        parameter_idx_orig += param_num;
        parameter_idx_new  += param_num;
        
    }

    Matrix_real optimized_parameters_ret(1,parameter_idx_new);
    memcpy( optimized_parameters_ret.get_data(), optimized_parameters_new.get_data(), optimized_parameters_ret.size()*sizeof(double) );

    optimized_parameters = optimized_parameters_ret;


    return gate_structure_ret;


}

/**
@brief ???????????????
*/
Gates_block*
N_Qubit_Decomposition_adaptive::remove_trivial_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters ) {


    int layer_num = gate_structure->get_gate_num();
    int parameter_idx = 0;

    Matrix_real&& optimized_parameters_loc = optimized_parameters.copy();

    Gates_block* gate_structure_loc = gate_structure->clone();

    int idx = 0;
    while (idx<layer_num ) {

        Gate* gate = gate_structure_loc->get_gate(idx);
        int param_num = gate->get_parameter_num();

        if (  gate_structure_loc->contains_adaptive_gate(idx) ) {
 
            double parameter = optimized_parameters_loc[parameter_idx];

            // check whether gate can be removed
            if ( std::abs(std::sin(parameter/2)) < 1e-2 ) {
                
                // remove gate from the structure
                Gates_block* gate_structure_tmp = compress_gate_structure( gate_structure_loc, optimized_parameters_loc, idx );

                optimized_parameters = optimized_parameters_loc;
                delete( gate_structure_loc );
                gate_structure_loc = gate_structure_tmp;
                layer_num = gate_structure_loc->get_gate_num();               

          
            }
            

        }

        parameter_idx += param_num;

        idx++;


    }
std::cout << "N_Qubit_Decomposition_adaptive::remove_trivial_gates :" << gate_structure->get_gate_num() << " reduced to " << gate_structure_loc->get_gate_num() << std::endl;
    return gate_structure_loc;




}


/**
@brief ???????????????
*/
Matrix_real 
N_Qubit_Decomposition_adaptive::create_reduced_parameters( Gates_block* gate_structure, Matrix_real& optimized_parameters, int layer_idx ) {


    // determine the index of the parameter that is about to delete
    int parameter_idx = 0;
    for ( int idx=0; idx<layer_idx; idx++) {
        Gate* gate = gate_structure->get_gate( idx );
        parameter_idx += gate->get_parameter_num();
    }


    Gate* gate = gate_structure->get_gate( layer_idx );
    int param_num_removed = gate->get_parameter_num();

    Matrix_real reduced_parameters(1, optimized_parameters.size() - param_num_removed );
    memcpy( reduced_parameters.get_data(), optimized_parameters.get_data(), (parameter_idx)*sizeof(double));
    memcpy( reduced_parameters.get_data()+parameter_idx, optimized_parameters.get_data()+parameter_idx+param_num_removed, (optimized_parameters.size()-parameter_idx-param_num_removed) *sizeof(double));


    return reduced_parameters;
}



/**
@brief ???????????????
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::exchange_CRZ_to_Adaptive_gates( Gates_block* gate_structure, Matrix_real& parameters ) {
std::cout << "N_Qubit_Decomposition_adaptive::exchange_CRZ_to_Adaptive_gates" << std::endl;
    Gates_block* gate_structure_ret = new Gates_block(qbit_num);

    int layer_num = gate_structure->get_gate_num();

    for (int idx=0; idx<layer_num; idx++ ) {

        Gate* gate = gate_structure->get_gate(idx);

        if ( gate->get_type() != BLOCK_OPERATION ) {
            std::cout << "N_Qubit_Decomposition_adaptive::replace_trivial_adaptive_gates: Only block gates are accepted in this conversion." << std::endl;
            exit(-1);
        }

        Gates_block* block_op = static_cast<Gates_block*>(gate);
        int param_num = gate->get_parameter_num();

        Gates_block* layer = block_op->clone();

        for ( int idx=0; idx<layer->get_gate_num(); idx++ ) {

            Gate* gate_tmp = layer->get_gate(idx);

            if ( gate_tmp->get_type() == CRY_OPERATION ) {
                // convert to CZ gate
  
                int target_qbit = gate_tmp->get_target_qbit();
                int control_qbit = gate_tmp->get_control_qbit();
                layer->release_gate( idx );

                Adaptive* ad_gate = new Adaptive(qbit_num, target_qbit, control_qbit);
                layer->insert_gate( (Gate*)ad_gate, idx);

            }
 

        }

        gate_structure_ret->add_gate_to_end((Gate*)layer);

                
 
        continue;
            

        
    }


    // transform the parameters
    for (int idx=0; idx<parameters.size(); idx+=5 ) {
        double Phi = parameters[idx];

        while (Phi < 0 ) {
            Phi = Phi + 2*M_PI;
        }

        while (Phi > 2*M_PI ) {
            Phi = Phi - 2*M_PI;
        }

        if ( Phi > M_PI ) {
            Phi = 2*M_PI - Phi;
        }

        parameters[idx] = std::acos(1 - 2*Phi/M_PI );

    }


    return gate_structure_ret;
}




/**
@brief ???????????????
*/
void
N_Qubit_Decomposition_adaptive::create_layers_from_decomposition_tree( const Decomposition_Tree_Node* minimal_root_node, int max_level, Gates_block* gate_structure) {

    //Gates_block* gate_structure = new Gates_block(qbit_num);

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
            return;
        }

    }

    return;

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

    std::vector<Gates_block* > layers;

    for (int target_qbit_loc = 0; target_qbit_loc<qbit_num; target_qbit_loc++) {
        for (int control_qbit_loc = target_qbit_loc+1; control_qbit_loc<qbit_num; control_qbit_loc++) {

            Gates_block* layer = new Gates_block( qbit_num );

            bool Theta = true;
            bool Phi = false;
            bool Lambda = true;
            layer->add_u3(target_qbit_loc, Theta, Phi, Lambda);
            layer->add_u3(control_qbit_loc, Theta, Phi, Lambda);  
//            layer->add_ry(control_qbit_loc);  
//            layer->add_ry(target_qbit_loc);  
            layer->add_adaptive(target_qbit_loc, control_qbit_loc);

            layers.push_back(layer);


        }
    }


    while (layers.size()>0) {
 
        int idx = std::rand() % layers.size();

        block->add_gate( layers[idx] );

        layers.erase( layers.begin() + idx );
    }


    return block;

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
    block->add_cz(_target_qbit, _control_qbit );
/*

    block->add_on();
    block->add_ry(_control_qbit);
    block->add_cz(_target_qbit, _control_qbit );
*/

    return block;

}



/**
@brief Call to add static gate components in front of the adaptively optimized gate structure.
*/
void
N_Qubit_Decomposition_adaptive::add_static_gate_layers_1(Gates_block* gate_structure ) {

    Gates_block* block;
    Gate* fixed_gate;
    Matrix_real pi_over_2(1,1);
    Matrix_real minus_pi_over_2(1,1);
    Matrix_real minus_pi(1,1);
    Matrix op_mtx;
    pi_over_2[0] = M_PI/2;
    minus_pi[0] = -M_PI;
    minus_pi_over_2[0] = -M_PI/2;
    RY ry_cons;

            bool Theta = true;
            bool Phi = false;
            bool Lambda = true;
       //     layer->add_u3(target_qbit_loc, Theta, Phi, Lambda);


    block = new Gates_block( qbit_num );
    block->add_cnot(3, 2 );
    block->add_cnot(2, 0 );
    //block->add_cz(2, 3 );
    gate_structure->add_gate( block );

/*
    block = new Gates_block( qbit_num );
    block->add_u3(2, Theta, Phi, Lambda);
    block->add_u3(3, Theta, Phi, Lambda);
    block->add_cry(3, 2 );
    gate_structure->add_gate( block );

    block = new Gates_block( qbit_num );
    block->add_u3(0, Theta, Phi, Lambda);
    block->add_u3(3, Theta, Phi, Lambda);
    block->add_cry(3, 0 );
    gate_structure->add_gate( block );

    block = new Gates_block( qbit_num );
    block->add_u3(2, Theta, Phi, Lambda);
    block->add_u3(0, Theta, Phi, Lambda);
    block->add_cry(0, 2 );
    gate_structure->add_gate( block );
*/
/*
    block = new Gates_block( qbit_num );
    block->add_u3(0, Theta, Phi, Lambda);
    block->add_u3(1, Theta, Phi, Lambda);
    block->add_u3(2, Theta, Phi, Lambda);
    block->add_u3(3, Theta, Phi, Lambda);
    gate_structure->add_gate( block );
*/
///// 
    block = new Gates_block( qbit_num );
    block->add_u3(0, Theta, Phi, Lambda);
    block->add_u3(1, Theta, Phi, Lambda);
    //block->add_ry(0);
    block->add_cnot(1, 0 );
    gate_structure->add_gate( block );
//////



    block = new Gates_block( qbit_num );
    block->add_ry(0);
    block->add_cz(3, 0 );
    gate_structure->add_gate( block );
/*
    block = new Gates_block( qbit_num );
    block->add_u3(2, Theta, Phi, Lambda);
    block->add_u3(3, Theta, Phi, Lambda);
    block->add_cry(3, 2 );
    gate_structure->add_gate( block );

    block = new Gates_block( qbit_num );
    block->add_u3(0, Theta, Phi, Lambda);
    block->add_u3(3, Theta, Phi, Lambda);
    block->add_cry(3, 0 );
    gate_structure->add_gate( block );

    block = new Gates_block( qbit_num );
    block->add_u3(2, Theta, Phi, Lambda);
    block->add_u3(0, Theta, Phi, Lambda);
    block->add_cry(0, 2 );
    gate_structure->add_gate( block );
*/
/*
    block = new Gates_block( qbit_num );
    block->add_u3(0, Theta, Phi, Lambda);
    block->add_u3(1, Theta, Phi, Lambda);
    block->add_u3(2, Theta, Phi, Lambda);
    block->add_u3(3, Theta, Phi, Lambda);
    gate_structure->add_gate( block );
*/
////////////////

    block = new Gates_block( qbit_num );
    block->add_ry(1);
    block->add_cnot(1, 3 );
    gate_structure->add_gate( block );


    block = new Gates_block( qbit_num );
    block->add_ry(1);
    block->add_cnot(1, 2 );
    gate_structure->add_gate( block );

    block = new Gates_block( qbit_num );
    block->add_ry(0);
    block->add_cnot(0, 2 );
    gate_structure->add_gate( block );


    block = new Gates_block( qbit_num );
    block->add_ry(1);
    block->add_cnot(1, 3 );
    gate_structure->add_gate( block );


    block = new Gates_block( qbit_num );
    block->add_ry(0);
    block->add_cz(3, 0 );
    gate_structure->add_gate( block );


    block = new Gates_block( qbit_num );
    block->add_ry(0);
    block->add_ry(1);
    block->add_cnot(1, 0 );
    gate_structure->add_gate( block );



    block = new Gates_block( qbit_num );
    block->add_cnot(0, 2 );
    block->add_ry(0);
    block->add_cnot(2, 0 );
    block->add_cnot(3, 2 );
    gate_structure->add_gate( block );








    return;
}


/**
@brief Call to add static gate components following the the adaptively optimized gate structure.
*/
void
N_Qubit_Decomposition_adaptive::add_static_gate_layers_2( Gates_block* gate_structure ) {

    Gates_block* block;
    Gate* fixed_gate;
    Matrix_real pi_over_2(1,1);
    Matrix_real minus_pi_over_2(1,1);
    Matrix_real minus_pi(1,1);
    Matrix op_mtx;
    pi_over_2[0] = M_PI/2;
    minus_pi[0] = -M_PI;
    minus_pi_over_2[0] = -M_PI/2;
    RY ry_cons;


    return;
}


/**
@brief ??????????????????
*/
void 
N_Qubit_Decomposition_adaptive::add_finalyzing_layer( Gates_block* gate_structure ) {


    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );
/*
    block->add_un();
    block->add_ry(qbit_num-1);
*/
    for (int idx=0; idx<qbit_num; idx++) {
            bool Theta = true;
            bool Phi = false;
            bool Lambda = true;
             block->add_u3(idx, Theta, Phi, Lambda);
//        block->add_ry(idx);
    }

    // adding the opeartion block to the gates
    gate_structure->add_gate( block );

}




/**
@brief ???????????????
*/
void
N_Qubit_Decomposition_adaptive::decompose_UN_gates() {

    // identify UN gates in the decomposition
    for ( std::vector<Gate*>::iterator it=gates.begin(); it != gates.end(); ++it ) {

        Gate* op = *it;

        if (op->get_type() == ON_OPERATION) {
            ON* un_op = static_cast<ON*>( op );

Matrix_real&& UN_parameters = un_op->get_optimized_parameters();
Matrix UNmtx = un_op->get_submatrix( UN_parameters );
//UNmtx.print_matrix();

std::cout << "found UN gate" << std::endl;

N_Qubit_Decomposition_adaptive cDecomp = N_Qubit_Decomposition_adaptive( UNmtx, qbit_num-1, 5, initial_guess );
cDecomp.start_decomposition(true);


        }


    }

exit(-1);

}





