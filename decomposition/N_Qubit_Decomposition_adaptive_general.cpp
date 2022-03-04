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

#include "N_Qubit_Decomposition_adaptive_general.h"
#include "N_Qubit_Decomposition_custom.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "Sub_Matrix_Decomposition_Cost_Function.h"
#include "Random_Orthogonal.h"

#include <time.h>
#include <stdlib.h>



/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive_general::N_Qubit_Decomposition_adaptive_general() : N_Qubit_Decomposition_adaptive() {

}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive_general::N_Qubit_Decomposition_adaptive_general( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, guess_type initial_guess_in ) : N_Qubit_Decomposition_adaptive(Umtx_in, qbit_num_in, level_limit_in, level_limit_min_in, initial_guess_in) {

}




/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive_general::N_Qubit_Decomposition_adaptive_general( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::vector<matrix_base<int>> topology_in, guess_type initial_guess_in ) : N_Qubit_Decomposition_adaptive(Umtx_in, qbit_num_in, level_limit_in, level_limit_min_in, topology_in, initial_guess_in) {

}


/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_adaptive_general::~N_Qubit_Decomposition_adaptive_general() {


    if ( gate_structure != NULL ) {
        // release custom gate structure
        delete gate_structure;
        gate_structure = NULL;
    }

}



/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
@param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
*/
void
N_Qubit_Decomposition_adaptive_general::start_decomposition(bool prepare_export) {


    //The stringstream input to store the output messages.
    std::stringstream sstream;

    //Integer value to set the verbosity level of the output messages.
    int verbose_level;

    verbose_level=1;
    sstream << "***************************************************************" << std::endl;
    sstream << "Starting to disentangle " << qbit_num << "-qubit matrix" << std::endl;
    sstream << "***************************************************************" << std::endl << std::endl << std::endl;
    print(sstream,verbose_level);	    	
   

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

    if (level_limit == 0 ) {
	sstream << "please increase level limit" << std::endl;
	verbose_level=1;
        print(sstream, verbose_level);	
        exit(-1);
    }



//iteration_loops[4] = 2;    
//Gates_block* gate_structure_loc = gate_structure->clone();
//insert_random_layers( gate_structure_loc, optimized_parameters_mtx );


    double optimization_tolerance_orig = optimization_tolerance;
    optimization_tolerance = 1e-4;


    // strages to store the optimized minimums in case of different cirquit depths
    std::vector<double> minimum_vec;
    std::vector<Gates_block*> gate_structure_vec;
    std::vector<Matrix_real> optimized_parameters_vec;

    int level = level_limit_min;
    while ( current_minimum > optimization_tolerance && level <= level_limit) {

        // reset optimized parameters
        optimized_parameters_mtx = Matrix_real(0,0);


        // create gate structure to be optimized
        Gates_block* gate_structure_loc = new Gates_block(qbit_num);
        for (int idx=0; idx<level; idx++) {

            // create the new decomposing layer
            Gates_block* layer = construct_gate_layer(0,0);
            gate_structure_loc->combine( layer );
        }
           
        // add finalyzing layer to the top of the gate structure
        add_finalyzing_layer( gate_structure_loc );

        //measure the time for the decompositin
        tbb::tick_count start_time_loc = tbb::tick_count::now();


        // solve the optimization problem
        N_Qubit_Decomposition_custom cDecomp_custom;
        // solve the optimization problem in isolated optimization process
        cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, initial_guess);
        cDecomp_custom.set_custom_gate_structure( gate_structure_loc );
        //cDecomp_custom.set_optimized_parameters( optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size() );
        cDecomp_custom.set_optimization_blocks( gate_structure_loc->get_gate_num() );
        cDecomp_custom.set_max_iteration( max_iterations );
        cDecomp_custom.set_verbose(false);
        cDecomp_custom.set_iteration_loops( iteration_loops );
        cDecomp_custom.set_optimization_tolerance( optimization_tolerance );  
        cDecomp_custom.start_decomposition(true);
        //cDecomp_custom.list_gates(0);

        tbb::tick_count end_time_loc = tbb::tick_count::now();

        minimum_vec.push_back(cDecomp_custom.get_current_minimum());
        gate_structure_vec.push_back(gate_structure_loc);
        optimized_parameters_vec.push_back(cDecomp_custom.get_optimized_parameters());



        if ( cDecomp_custom.get_current_minimum() < optimization_tolerance ) {
 	   sstream << "Optimization problem solved with " << gate_structure_loc->get_gate_num() << " decomposing layers in " << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
	   verbose_level=1;
           print(sstream, verbose_level);	     
           //cDecomp_custom.list_gates(0);
           break;
        }   
        else {
	   sstream << "Optimization problem converged to " << cDecomp_custom.get_current_minimum() << " with " <<  gate_structure_loc->get_gate_num() << " decomposing layers in "   << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
	   verbose_level=1;
           print(sstream, verbose_level);	         
        }

        level++;
    }



    // find the best decomposition
    int idx_min = 0;
    double current_minimum = minimum_vec[0];
    for (int idx=1; idx<minimum_vec.size(); idx++) {
        if( current_minimum > minimum_vec[idx] ) {
            idx_min = idx;
            current_minimum = minimum_vec[idx];
        }
    }
     
    Gates_block* gate_structure_loc = gate_structure_vec[idx_min];
    optimized_parameters_mtx = optimized_parameters_vec[idx_min];

    // release unnecesarry data
    for (int idx=0; idx<minimum_vec.size(); idx++) {
        if( idx == idx_min ) {
            continue;
        }
        delete( gate_structure_vec[idx] );
    }    
    minimum_vec.clear();
    gate_structure_vec.clear();
    optimized_parameters_vec.clear();
    

    if (current_minimum > optimization_tolerance) {
	sstream << "Decomposition did not reached prescribed high numerical precision." << std::endl;
	verbose_level=1;
        print(sstream, verbose_level);	         
        optimization_tolerance = 1.5*current_minimum < 1e-2 ? 1.5*current_minimum : 1e-2;
    }

    sstream << "Continue with the compression of gate structure consisting of " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
    verbose_level=1;
    print(sstream, verbose_level);	


    if ( current_minimum > 1e-2 ) {
	sstream << "decomposition was unsuccessful. Exiting" << std::endl;
	verbose_level=1;
        print(sstream, verbose_level);	
        return;
    }


    sstream << std::endl;
    sstream << std::endl;
    sstream << "**************************************************************" << std::endl;
    sstream << "***************** Compressing Gate structure *****************" << std::endl;
    sstream << "**************************************************************" << std::endl;
    verbose_level=1;
    print(sstream,verbose_level);	    	
    


    for ( int iter=0; iter<25; iter++ ) {
	sstream << "iteration " << iter+1 << ": ";
	verbose_level=1;
        print(sstream, verbose_level);	
	
        
        Gates_block* gate_structure_compressed = compress_gate_structure( gate_structure_loc );

        if ( gate_structure_compressed != gate_structure_loc ) {
            delete( gate_structure_loc );
            gate_structure_loc = gate_structure_compressed;
            gate_structure_compressed = NULL;
        }

    }


    sstream << "**************************************************************" << std::endl;
    sstream << "************ Final tuning of the Gate structure **************" << std::endl;
    sstream << "**************************************************************" << std::endl;
    verbose_level=1;
    print(sstream,verbose_level);	    	
    

    optimization_tolerance = optimization_tolerance_orig;

    // store the decomposing gate structure    
    combine( gate_structure_loc );
    optimization_block = get_gate_num();



    Gates_block* gate_structure_tmp = gate_structure_loc->clone();
    Matrix_real optimized_parameters_save = optimized_parameters_mtx;

    release_gates();
    optimized_parameters_mtx = optimized_parameters_save;

    combine( gate_structure_tmp );
    delete( gate_structure_tmp );
    delete( gate_structure_loc );


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

    

    verbose_level=1;
    sstream << "In the decomposition with error = " << decomposition_error << " were used " << layer_num << " gates with:" << std::endl;
    print(sstream,verbose_level);	    	

    verbose_level=1;
      
        if ( gates_num.u3>0 ) sstream << gates_num.u3 << " U3 opeartions," << std::endl;
        if ( gates_num.rx>0 ) sstream << gates_num.rx << " RX opeartions," << std::endl;
        if ( gates_num.ry>0 ) sstream << gates_num.ry << " RY opeartions," << std::endl;
        if ( gates_num.rz>0 ) sstream << gates_num.rz << " RZ opeartions," << std::endl;
        if ( gates_num.cnot>0 ) sstream << gates_num.cnot << " CNOT opeartions," << std::endl;
        if ( gates_num.cz>0 ) sstream << gates_num.cz << " CZ opeartions," << std::endl;
        if ( gates_num.ch>0 ) sstream << gates_num.ch << " CH opeartions," << std::endl;
        if ( gates_num.x>0 ) sstream << gates_num.x << " X opeartions," << std::endl;
        if ( gates_num.sx>0 ) sstream << gates_num.sx << " SX opeartions," << std::endl; 
        if ( gates_num.syc>0 ) sstream << gates_num.syc << " Sycamore opeartions," << std::endl;
        if ( gates_num.un>0 ) sstream << gates_num.un << " UN opeartions," << std::endl;
        if ( gates_num.cry>0 ) sstream << gates_num.cry << " CRY opeartions," << std::endl;
        if ( gates_num.adap>0 ) sstream << gates_num.adap << " Adaptive opeartions," << std::endl;

    	print(sstream,verbose_level);	    	
    	 

        std::cout << std::endl;
        tbb::tick_count current_time = tbb::tick_count::now();

	verbose_level=1;
	sstream << "--- In total " << (current_time - start_time).seconds() << " seconds elapsed during the decomposition ---" << std::endl;
    	print(sstream,verbose_level);	    	
    	

#if BLAS==0 // undefined BLAS
    omp_set_num_threads(num_threads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(num_threads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(num_threads);
#endif

}




bool are_two_qubits_independent( Gates_block* two_qubit_gate, Matrix_real& params ) {


    //The stringstream input to store the output messages.
    std::stringstream sstream;

    //Integer value to set the verbosity level of the output messages.
    int verbose_level;

    logging output;

    std::vector<int> involved_qubits = two_qubit_gate->get_involved_qubits();
    if ( involved_qubits.size() > 2 ) {
	sstream << "N_Qubit_Decomposition_adaptive::are_two_qubits_independent: the givel block contains more than 2 qubits" << std::endl;
	verbose_level=1;
        output.print(sstream, verbose_level);	
        exit(-1);
    }

    Matrix mtx_loc = two_qubit_gate->get_matrix( params );
    QGD_Complex16* mtx_data = mtx_loc.get_data();    
    int dim = mtx_loc.rows;

    // reorder the matrix elements
    Matrix mtx_new( dim, dim );
    QGD_Complex16* mtx_new_data = mtx_new.get_data();

    int qbit = involved_qubits[0];
    size_t dim_loc = (size_t)Power_of_2(qbit);
    size_t dim_loc2 = (size_t)dim_loc*2;
    int dim_over_2 = dim/2;

    for ( size_t row_idx=0; row_idx<dim_over_2; row_idx++ ) {

        int row_tmp  = row_idx % dim_loc;
        int row_tmp2 = (row_idx-row_tmp)/dim_loc;

        for ( size_t col_idx=0; col_idx<dim_over_2; col_idx+=dim_loc ) {

            memcpy( mtx_new_data + row_idx*dim + col_idx,                             mtx_data + (row_tmp2*dim_loc2+row_tmp)*dim + (col_idx*2),                   dim_loc*sizeof(QGD_Complex16) );
            memcpy( mtx_new_data + row_idx*dim + col_idx + dim_over_2,                mtx_data + (row_tmp2*dim_loc2+row_tmp)*dim + (col_idx*2 + dim_loc),         dim_loc*sizeof(QGD_Complex16) );
            memcpy( mtx_new_data + (row_idx + dim_over_2)*dim + col_idx,              mtx_data + (row_tmp2*dim_loc2+row_tmp+dim_loc)*dim + (col_idx*2),           dim_loc*sizeof(QGD_Complex16) );
            memcpy( mtx_new_data + (row_idx + dim_over_2)*dim + col_idx + dim_over_2, mtx_data + (row_tmp2*dim_loc2+row_tmp+dim_loc)*dim + (col_idx*2 + dim_loc), dim_loc*sizeof(QGD_Complex16) );


        }
    }    

    double cost_fnc = get_submatrix_cost_function(mtx_new);
//std::cout << cost_fnc << std::endl;
    if (cost_fnc < 1e-2 ) return true;

    return false;

}

/**
@brief ???????????????
*/
int 
N_Qubit_Decomposition_adaptive_general::get_panelty( Gates_block* gate_structure, Matrix_real& optimized_parameters ) {

    //The stringstream input to store the output messages.
    std::stringstream sstream;

    //Integer value to set the verbosity level of the output messages.
    int verbose_level;

    int panelty = 0;

    // iterate over the elements of tha parameter array
    int parameter_idx = 0;
    for ( int idx=0; idx<gate_structure->get_gate_num(); idx++) {

        Gate* gate = gate_structure->get_gate( idx );
        Gates_block* block = static_cast<Gates_block*>(gate);
	gates_num gates_num = block->get_gate_nums();
        if ( gates_num.cnot == 0 ) {
            parameter_idx += gate->get_parameter_num();
            continue;
        }

        Matrix_real param_loc(optimized_parameters.get_data()+parameter_idx, 1, gate->get_parameter_num());

        if ( are_two_qubits_independent(block, param_loc) ) {
            panelty++;
        }
        else{
            panelty += 2;
        }


        parameter_idx += gate->get_parameter_num();
    }

    return panelty;


}


/**
@brief ???????????????
*/
Gates_block*
N_Qubit_Decomposition_adaptive_general::remove_trivial_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters, double& currnt_minimum_loc ) {


    //The stringstream input to store the output messages.
    std::stringstream sstream;

    //Integer value to set the verbosity level of the output messages.
    int verbose_level;

    int layer_num = gate_structure->get_gate_num()-1;
    int parameter_idx = 0;

    Matrix_real&& optimized_parameters_loc = optimized_parameters.copy();

    Gates_block* gate_structure_loc = gate_structure->clone();

    int idx = 0;
    while (idx<layer_num ) {

        Gate* gate = gate_structure_loc->get_gate(idx);
        int param_num = gate->get_parameter_num();

        Matrix_real param_loc(optimized_parameters.get_data()+parameter_idx, 1, param_num);
        Gates_block* block = static_cast<Gates_block*>(gate);
	gates_num gates_num = block->get_gate_nums();
        if ( gates_num.cnot == 0 ) {

            idx++;
            parameter_idx += param_num;
            continue;
        }

        Matrix mtx_loc = block->get_matrix( param_loc );

        if ( are_two_qubits_independent(block, param_loc) ) {
           Gates_block* gate_structure_tmp = compress_gate_structure( gate_structure_loc, idx, optimized_parameters_loc, currnt_minimum_loc );
  	   sstream << "removing trivial layer: " << idx  << " from " << gate_structure_loc->get_gate_num() << " to " << gate_structure_tmp->get_gate_num() << std::endl;
	   verbose_level=1;
	   print(sstream, verbose_level);
	   
       if ( gate_structure_loc->get_gate_num() > gate_structure_tmp->get_gate_num() ) {
                idx--;
            }
       
        optimized_parameters = optimized_parameters_loc;
        delete( gate_structure_loc );
       	gate_structure_loc = gate_structure_tmp;
        layer_num = gate_structure_loc->get_gate_num();  
             
        }


 

        parameter_idx += param_num;

        idx++;


    }
//std::cout << "N_Qubit_Decomposition_adaptive::remove_trivial_gates :" << gate_structure->get_gate_num() << " reduced to " << gate_structure_loc->get_gate_num() << std::endl;
    return gate_structure_loc;




}


/**
@brief ???????????????
*/
Matrix_real 
N_Qubit_Decomposition_adaptive_general::create_reduced_parameters( Gates_block* gate_structure, Matrix_real& optimized_parameters, int layer_idx ) {

    return N_Qubit_Decomposition_adaptive::create_reduced_parameters( gate_structure, optimized_parameters, layer_idx );

}












/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
Gates_block* 
N_Qubit_Decomposition_adaptive_general::construct_gate_layer( const int& _target_qbit, const int& _control_qbit) {

    //The stringstream input to store the output messages.
    std::stringstream sstream;

    //Integer value to set the verbosity level of the output messages.
    int verbose_level;

    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );

    int layer_num = (qbit_num*(qbit_num-1))/2;
    std::vector<Gates_block* > layers;

    if ( topology.size() > 0 ) {
        for ( std::vector<matrix_base<int>>::iterator it=topology.begin(); it!=topology.end(); it++) {

            if ( it->size() != 2 ) {
	       sstream << "The connectivity data should contains two qubits" << std::endl;
	       verbose_level=1;
	       print(sstream, verbose_level);	
               it->print_matrix();
               exit(-1);
            }

            int control_qbit_loc = (*it)[0];
            int target_qbit_loc = (*it)[1];

            if ( control_qbit_loc >= qbit_num || target_qbit_loc >= qbit_num ) {
		sstream << "Label of control/target qubit should be less than the number of qubits in the register." << std::endl;
		verbose_level=1;
        	print(sstream, verbose_level);	
                exit(-1);            
            }

            Gates_block* layer = new Gates_block( qbit_num );

            bool Theta = true;
            bool Phi = true;
            bool Lambda = true;
            layer->add_u3(target_qbit_loc, Theta, Phi, Lambda);
            layer->add_u3(control_qbit_loc, Theta, Phi, Lambda); 
            layer->add_adaptive(target_qbit_loc, control_qbit_loc);

            layers.push_back(layer);


        }
    }
    else {  

        // sequ
        for (int target_qbit_loc = 0; target_qbit_loc<qbit_num; target_qbit_loc++) {
            for (int control_qbit_loc = target_qbit_loc+1; control_qbit_loc<qbit_num; control_qbit_loc++) {

                Gates_block* layer = new Gates_block( qbit_num );

                bool Theta = true;
                bool Phi = true;
                bool Lambda = true;
                layer->add_u3(target_qbit_loc, Theta, Phi, Lambda);
                layer->add_u3(control_qbit_loc, Theta, Phi, Lambda); 
                layer->add_cnot(target_qbit_loc, control_qbit_loc);
                layer->add_u3(target_qbit_loc, Theta, Phi, Lambda);
                layer->add_u3(control_qbit_loc, Theta, Phi, Lambda); 
                layer->add_cnot(target_qbit_loc, control_qbit_loc);

                layers.push_back(layer);

            }
        }
    }

/*
    for (int idx=0; idx<layers.size(); idx++) {
        Gates_block* layer = (Gates_block*)layers[idx];
        block->add_gate( layers[idx] );

    }

*/
    while (layers.size()>0) { 
        int idx = std::rand() % layers.size();
        Gates_block* layer = (Gates_block*)layers[idx];
        block->add_gate( layers[idx] );
        layers.erase( layers.begin() + idx );
    }


    return block;


}




/**
@brief ??????????????????
*/
void 
N_Qubit_Decomposition_adaptive_general::add_finalyzing_layer( Gates_block* gate_structure ) {


    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );


    for (int idx=0; idx<qbit_num; idx++) {
            bool Theta = true;
            bool Phi = true;
            bool Lambda = true;
            block->add_u3(idx, Theta, Phi, Lambda);
    }

    // adding the opeartion block to the gates
    gate_structure->add_gate( block );

}




