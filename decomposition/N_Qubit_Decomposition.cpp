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
/*! \file N_Qubit_Decomposition.cpp
    \brief Base class to determine the decomposition of a unitary into a sequence of two-qubit and one-qubit gate gates.
    This class contains the non-template implementation of the decomposition class
*/

#include "N_Qubit_Decomposition.h"
#include "N_Qubit_Decomposition_Cost_Function.h"



/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition::N_Qubit_Decomposition() : Optimization_Interface() {

    set_optimizer( BFGS );



}





/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition::N_Qubit_Decomposition( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, std::map<std::string, Config_Element>& config_in, guess_type initial_guess_in= CLOSE_TO_ZERO ) : Optimization_Interface(Umtx_in, qbit_num_in, optimize_layer_num_in, config_in, initial_guess_in) {

    set_optimizer( BFGS );

}



/**
@brief Destructor of the class
*/
N_Qubit_Decomposition::~N_Qubit_Decomposition() {


}




/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
@param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
*/
void
N_Qubit_Decomposition::start_decomposition(bool finalize_decomp, bool prepare_export) {



    //The stringstream input to store the output messages.
    std::stringstream sstream;
    sstream << "***************************************************************" << std::endl;
    sstream << "Starting to disentangle " << qbit_num << "-qubit matrix" << std::endl;
    sstream << "***************************************************************" << std::endl << std::endl << std::endl;
    print(sstream, 1);	    	
    

    

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

    // create an instance of class to disentangle the given qubit pair
    Sub_Matrix_Decomposition* cSub_decomposition = new Sub_Matrix_Decomposition(Umtx, qbit_num, optimize_layer_num, config, initial_guess);

    // setting the verbosity
    cSub_decomposition->set_verbose( verbose );

    // setting the debugfile name
    cSub_decomposition->set_debugfile( debugfile_name );

    // setting the maximal number of layers used in the subdecomposition
    cSub_decomposition->set_max_layer_num( max_layer_num );

    // setting the number of successive identical blocks used in the subdecomposition
    cSub_decomposition->set_identical_blocks( identical_blocks );

    // setting the iteration loops in each step of the optimization process
    cSub_decomposition->set_iteration_loops( iteration_loops );

    // set custom gate structure if given
    std::map<int,Gates_block*>::iterator key_it = gate_structure.find( qbit_num );
    if ( key_it != gate_structure.end() ) {
        cSub_decomposition->set_custom_gate_layers( gate_structure[qbit_num] );
    }

    // The maximal error of the optimization problem
    cSub_decomposition->set_optimization_tolerance( optimization_tolerance );

    // setting the maximal number of iterations in the disentangling process
    cSub_decomposition->optimization_block = optimization_block;

    // setting the number of operators in one sub-layer of the disentangling process
    //cSub_decomposition->max_outer_iterations = self.max_outer_iterations

    //start to disentangle the qubit pair
    cSub_decomposition->disentangle_submatrices();
    if ( !cSub_decomposition->subdisentaglement_done) {
        return;
    }
//return;
    // saving the subunitarization gates
    extract_subdecomposition_results( cSub_decomposition );

    delete cSub_decomposition;
    cSub_decomposition = NULL;

    // decompose the qubits in the disentangled submatrices
    decompose_submatrix();



    if (finalize_decomp) {
        // finalizing the decompostition
        finalize_decomposition();

        // simplify layers
        if (qbit_num>2) {
            simplify_layers();
        }

        int optimization_block_orig = optimization_block;
        if ( optimization_block > 0 ) {
            optimization_block = optimization_block*3;
        }
        //max_outer_iterations = 4;


        // final tuning of the decomposition parameters
        final_optimization();

        optimization_block = optimization_block_orig;

        // prepare gates to export
        if (prepare_export) {
            prepare_gates_to_export();
        }

        // calculating the final error of the decomposition
        Matrix matrix_decomposed = get_transformed_matrix(optimized_parameters_mtx, gates.begin(), gates.size(), Umtx );
	calc_decomposition_error( matrix_decomposed );
        

        // get the number of gates used in the decomposition
        gates_num gates_num = get_gate_nums();

        sstream.str("");
    	sstream << "In the decomposition with error = " << decomposition_error << " were used " << layer_num << " gates with:" << std::endl;
    	
        if ( gates_num.u3>0 ) sstream << gates_num.u3 << " U3 gates," << std::endl;
        if ( gates_num.rx>0 ) sstream << gates_num.rx << " RX gates," << std::endl;
        if ( gates_num.ry>0 ) sstream << gates_num.ry << " RY gates," << std::endl;
        if ( gates_num.rz>0 ) sstream << gates_num.rz << " RZ gates," << std::endl;
        if ( gates_num.cnot>0 ) sstream << gates_num.cnot << " CNOT gates," << std::endl;
        if ( gates_num.cz>0 ) sstream << gates_num.cz << " CZ gates," << std::endl;
        if ( gates_num.ch>0 ) sstream << gates_num.ch << " CH gates," << std::endl;
        if ( gates_num.x>0 ) sstream << gates_num.x << " X gates," << std::endl;
        if ( gates_num.sx>0 ) sstream << gates_num.sx << " SX gates," << std::endl; 
        if ( gates_num.syc>0 ) sstream << gates_num.syc << " Sycamore gates," << std::endl;
        if ( gates_num.adap>0 )sstream << gates_num.adap << " Adaptive gates," << std::endl;   	


        sstream << std::endl;
        tbb::tick_count current_time = tbb::tick_count::now();

	sstream << "--- In total " << (current_time - start_time).seconds() << " seconds elapsed during the decomposition ---" << std::endl;
    	print(sstream, 1);	    	
    	

        

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
@brief Start the decompostion process to recursively decompose the submatrices.
*/
void
N_Qubit_Decomposition::decompose_submatrix() {


        if (decomposition_finalized) {
	   std::stringstream sstream;
	   sstream << "Decomposition was already finalized" << std::endl;
	   print(sstream, 1);	    	     
           return;
        }

        if (qbit_num == 2) {
            return;
        }

        // obtaining the subdecomposed submatrices
//Matrix_real optimized_parameters_mtx_tmp(optimized_parameters, 1, parameter_num );
        Matrix subdecomposed_matrix_mtx = get_transformed_matrix( optimized_parameters_mtx, gates.begin(), gates.size(), Umtx );
        QGD_Complex16* subdecomposed_matrix = subdecomposed_matrix_mtx.get_data();

        // get the most unitary submatrix
        // get the number of 2qubit submatrices
        int submatrices_num_row = 2;

        // get the size of the submatrix
        int submatrix_size = int(matrix_size/2);

        // fill up the submatrices and select the most unitary submatrix

        Matrix most_unitary_submatrix_mtx = Matrix(submatrix_size, submatrix_size );
        QGD_Complex16* most_unitary_submatrix = most_unitary_submatrix_mtx.get_data();
        double unitary_error_min = 1e8;

        for (int idx=0; idx<submatrices_num_row; idx++) { // in range(0,submatrices_num_row):
            for (int jdx=0; jdx<submatrices_num_row; jdx++) { // in range(0,submatrices_num_row):

                Matrix submatrix_mtx = Matrix(submatrix_size, submatrix_size);
                QGD_Complex16* submatrix = submatrix_mtx.get_data();

                for ( int row_idx=0; row_idx<submatrix_size; row_idx++ ) {
                    int matrix_offset = idx*(matrix_size*submatrix_size) + jdx*(submatrix_size) + row_idx*matrix_size;
                    int submatrix_offset = row_idx*submatrix_size;
                    memcpy(submatrix+submatrix_offset, subdecomposed_matrix+matrix_offset, submatrix_size*sizeof(QGD_Complex16));
                }

                // calculate the product of submatrix*submatrix'
                Matrix submatrix_mtx_adj = submatrix_mtx;
                submatrix_mtx_adj.transpose();
                submatrix_mtx_adj.conjugate();
                Matrix submatrix_prod = dot( submatrix_mtx, submatrix_mtx_adj);

                // subtract corner element
                QGD_Complex16 corner_element = submatrix_prod[0];
                for (int row_idx=0; row_idx<submatrix_size; row_idx++) {
                    submatrix_prod[row_idx*submatrix_size+row_idx].real = submatrix_prod[row_idx*submatrix_size+row_idx].real - corner_element.real;
                    submatrix_prod[row_idx*submatrix_size+row_idx].imag = submatrix_prod[row_idx*submatrix_size+row_idx].imag - corner_element.imag;
                }

                double unitary_error = cblas_dznrm2( submatrix_size*submatrix_size, submatrix_prod.get_data(), 1 );

                if (unitary_error < unitary_error_min) {
                    unitary_error_min = unitary_error;
                    memcpy(most_unitary_submatrix, submatrix, submatrix_size*submatrix_size*sizeof(QGD_Complex16));
                }

            }
        }

        // if the qubit number in the submatirx is greater than 2 new N-qubit decomposition is started

        // create class tp decompose submatrices
        N_Qubit_Decomposition* cdecomposition = new N_Qubit_Decomposition(most_unitary_submatrix_mtx, qbit_num-1, optimize_layer_num, config, initial_guess);

        // setting the verbosity
        cdecomposition->set_verbose( verbose );

        // setting the debugfile name
        cdecomposition->set_debugfile( debugfile_name );

        // Maximal number of iteartions in the optimization process
        cdecomposition->set_max_iteration(max_outer_iterations);

        // Set the number of identical successive blocks in the sub-decomposition
        cdecomposition->set_identical_blocks(identical_blocks);

        // Set the maximal number of layers for the sub-decompositions
        cdecomposition->set_max_layer_num(max_layer_num);

        // setting the iteration loops in each step of the optimization process
        cdecomposition->set_iteration_loops( iteration_loops );

        // setting gate layer
        cdecomposition->set_optimization_blocks( optimization_block );

        // set custom gate structure if given
        cdecomposition->set_custom_gate_structure( gate_structure );

        // set the toleration of the optimization process
        cdecomposition->set_optimization_tolerance( optimization_tolerance );

        // starting the decomposition of the random unitary
        cdecomposition->start_decomposition(false, false);


        // saving the decomposition gates
        extract_subdecomposition_results( reinterpret_cast<Sub_Matrix_Decomposition*>(cdecomposition) );

        delete cdecomposition;

}



/**
@brief Call to extract and store the calculated parameters and gates of the sub-decomposition processes
@param cSub_decomposition An instance of class Sub_Matrix_Decomposition used to disentangle the n-th qubit from the others.
*/
void
N_Qubit_Decomposition::extract_subdecomposition_results( Sub_Matrix_Decomposition* cSub_decomposition ) {


        // get the unitarization parameters
        int parameter_num_sub_decomp = cSub_decomposition->get_parameter_num();

        // adding the unitarization parameters to the ones stored in the class
        Matrix_real optimized_parameters_tmp(1, parameter_num_sub_decomp+parameter_num);
        cSub_decomposition->get_optimized_parameters(optimized_parameters_tmp.get_data());

        if ( optimized_parameters_mtx.size() > 0 ) {
            memcpy(optimized_parameters_tmp.get_data()+parameter_num_sub_decomp, optimized_parameters_mtx.get_data(), parameter_num*sizeof(double));
            //qgd_free( optimized_parameters );
            //optimized_parameters = NULL;
        }

        optimized_parameters_mtx = optimized_parameters_tmp;

        // cloning the gate list obtained during the subdecomposition
        std::vector<Gate*> sub_decomp_ops = cSub_decomposition->get_gates();
        int gate_num = cSub_decomposition->get_gate_num();

        for ( int idx = gate_num-1; idx >=0; idx--) {
            Gate* op = sub_decomp_ops[idx];

            if (op->get_type() == CNOT_OPERATION) {
                CNOT* cnot_op = static_cast<CNOT*>( op );
                CNOT* cnot_op_cloned = cnot_op->clone();
                cnot_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( cnot_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == CZ_OPERATION) {
                CZ* cz_op = static_cast<CZ*>( op );
                CZ* cz_op_cloned = cz_op->clone();
                cz_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( cz_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == CH_OPERATION) {
                CH* ch_op = static_cast<CH*>( op );
                CH* ch_op_cloned = ch_op->clone();
                ch_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( ch_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == SYC_OPERATION) {
                SYC* syc_op = static_cast<SYC*>( op );
                SYC* syc_op_cloned = syc_op->clone();
                syc_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( syc_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == U3_OPERATION) {
                U3* u3_op = static_cast<U3*>( op );
                U3* u3_op_cloned = u3_op->clone();
                u3_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( u3_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == RX_OPERATION) {
                RX* rx_op = static_cast<RX*>( op );
                RX* rx_op_cloned = rx_op->clone();
                rx_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( rx_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == RY_OPERATION) {
                RY* ry_op = static_cast<RY*>( op );
                RY* ry_op_cloned = ry_op->clone();
                ry_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( ry_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == RZ_OPERATION) {
                RZ* rz_op = static_cast<RZ*>( op );
                RZ* rz_op_cloned = rz_op->clone();
                rz_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( rz_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == RZ_P_OPERATION) {
                RZ_P* rz_op = static_cast<RZ_P*>( op );
                RZ_P* rz_op_cloned = rz_op->clone();
                rz_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( rz_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == X_OPERATION) {
                X* x_op = static_cast<X*>( op );
                X* x_op_cloned = x_op->clone();
                x_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( x_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == Y_OPERATION) {
                Y* y_op = static_cast<Y*>( op );
                Y* y_op_cloned = y_op->clone();
                y_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( y_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == Z_OPERATION) {
                Z* z_op = static_cast<Z*>( op );
                Z* z_op_cloned = z_op->clone();
                z_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( z_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == Y_OPERATION) {
                Y* y_op = static_cast<Y*>( op );
                Y* y_op_cloned = y_op->clone();
                y_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( y_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == SX_OPERATION) {
                SX* sx_op = static_cast<SX*>( op );
                SX* sx_op_cloned = sx_op->clone();
                sx_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( sx_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == CRY_OPERATION) {
                CRY* cry_op = static_cast<CRY*>( op );
                CRY* cry_op_cloned = cry_op->clone();
                cry_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( cry_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == ADAPTIVE_OPERATION) {
                Adaptive* ad_op = static_cast<Adaptive*>( op );
                Adaptive* ad_op_cloned = ad_op->clone();
                ad_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( ad_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == BLOCK_OPERATION) {
                Gates_block* block_op = static_cast<Gates_block*>( op );
                Gates_block* block_op_cloned = block_op->clone();
                block_op_cloned->set_qbit_num( qbit_num );
                Gate* op_cloned = static_cast<Gate*>( block_op_cloned );
                add_gate( op_cloned );
            }
            else if (op->get_type() == GENERAL_OPERATION) {
                Gate* op_cloned = op->clone();
                op_cloned->set_qbit_num( qbit_num );
                add_gate( op_cloned );
            }
        }

}



/**
@brief Call to simplify the gate structure in the layers if possible (i.e. tries to reduce the number of two-qubit gates)
*/
void
N_Qubit_Decomposition::simplify_layers() {

    	//The stringstream input to store the output messages.
    	std::stringstream sstream;
	sstream << "***************************************************************" << std::endl;
	sstream << "Try to simplify layers" << std::endl;
	sstream << "***************************************************************" << std::endl;
	print(sstream, 1);	    	
	         
        

        // current starting index of the optimized parameters
        int parameter_idx = 0;

        Gates_block* gates_loc = new Gates_block( qbit_num );
        Matrix_real optimized_parameters_loc_mtx(1, parameter_num);
        double* optimized_parameters_loc = optimized_parameters_loc_mtx.get_data();
        int parameter_num_loc = 0;

        int layer_idx = 0;

        while (layer_idx < (int)gates.size()) {

            // generate a block of gates to be simplified
            // (containg only successive two-qubit gates)
            Gates_block* block_to_simplify = new Gates_block( qbit_num );

            std::vector<int> involved_qbits;
            // layers in the block to be simplified
            Gates_block* blocks_to_save = new Gates_block( qbit_num );

            // get the successive gates involving the same qubits
            while (true) {

                if (layer_idx >=(int)gates.size() ) {
                    break;
                }

                // get the current layer of gates
                Gate* gate = gates[layer_idx];
                layer_idx = layer_idx + 1;

                Gates_block* block_gate;
                if (gate->get_type() == BLOCK_OPERATION) {
                    block_gate = static_cast<Gates_block*>(gate);
                }
                else {
                    layer_idx = layer_idx + 1;
                    continue;
                }


                //get the involved qubits
                std::vector<int> involved_qbits_op = block_gate->get_involved_qubits();

                // add involved qubits of the gate to the list of involved qubits in the layer
                for (std::vector<int>::iterator it=involved_qbits_op.begin(); it!=involved_qbits_op.end(); it++) {
                    add_unique_elelement( involved_qbits, *it );
                }

                if ( (involved_qbits.size())> 2 && blocks_to_save->get_gate_num() > 0 ) {
                    layer_idx = layer_idx -1;
                    break;
                }

                blocks_to_save->combine(block_gate);

                // adding the gates to teh block if they act on the same qubits
                block_to_simplify->combine(block_gate);


            }

            //number of perations in the block
            int parameter_num_block = block_to_simplify->get_parameter_num();

            // get the number of two-qubit gates and store the block gates if the number of CNOT gates cannot be reduced
            gates_num gate_nums = block_to_simplify->get_gate_nums();

            if (gate_nums.cnot + gate_nums.cz + gate_nums.ch < 2 || involved_qbits.size()> 2) {
                gates_loc->combine(blocks_to_save);
                memcpy(optimized_parameters_loc+parameter_num_loc, optimized_parameters_mtx.get_data()+parameter_idx, parameter_num_block*sizeof(double) );
                parameter_idx = parameter_idx + parameter_num_block;
                parameter_num_loc = parameter_num_loc + parameter_num_block;

                if ( block_to_simplify != NULL ) {
                    delete block_to_simplify;
                    block_to_simplify = NULL;
                }

                if ( blocks_to_save != NULL ) {
                    delete blocks_to_save;
                    blocks_to_save = NULL;
                }

                involved_qbits.clear();

                continue;
            }


            involved_qbits.clear();

            // simplify the given layer
            std::map<int,int> max_layer_num_loc;
            max_layer_num_loc.insert( std::pair<int, int>(2,  gate_nums.cnot+gate_nums.cz+gate_nums.ch-1 ) );
            Gates_block* simplified_layer = NULL;
            double* simplified_parameters = NULL;
            int simplified_parameter_num=0;

            // Try to simplify the sequence of 2-qubit gates
            int simplification_status = simplify_layer( block_to_simplify, optimized_parameters_mtx.get_data()+parameter_idx, parameter_num_block, max_layer_num_loc, simplified_layer, simplified_parameters, simplified_parameter_num );



            // adding the simplified gates (or the non-simplified if the simplification was not successfull)
            if (simplification_status == 0) {
                gates_loc->combine( simplified_layer );

                if (parameter_num < parameter_num_loc + simplified_parameter_num ) {
                    //optimized_parameters_loc = (double*)qgd_realloc( optimized_parameters_loc, parameter_num_loc + simplified_parameter_num, sizeof(double), 64 );
                    optimized_parameters_loc_mtx = Matrix_real(1, parameter_num_loc + simplified_parameter_num);
                    optimized_parameters_loc = optimized_parameters_loc_mtx.get_data();
                }
                memcpy(optimized_parameters_loc+parameter_num_loc, simplified_parameters, simplified_parameter_num*sizeof(double) );
                parameter_num_loc = parameter_num_loc + simplified_parameter_num;
            }
            else {
                // addign the stacked gate to the list, sice the simplification was unsuccessful
                gates_loc->combine( blocks_to_save );

                if (parameter_num < parameter_num_loc + parameter_num_block ) {
                    //optimized_parameters_loc = (double*)qgd_realloc( optimized_parameters_loc, parameter_num_loc + parameter_num_block, sizeof(double), 64 );
                    optimized_parameters_loc_mtx = Matrix_real(1, parameter_num_loc + parameter_num_block);
                    optimized_parameters_loc = optimized_parameters_loc_mtx.get_data();
                }
                memcpy(optimized_parameters_loc+parameter_num_loc, optimized_parameters_mtx.get_data()+parameter_idx, parameter_num_block*sizeof(double) );
                parameter_num_loc = parameter_num_loc + parameter_num_block;
            }

            parameter_idx = parameter_idx + parameter_num_block;

            if ( simplified_layer != NULL ) {
                delete simplified_layer;
                simplified_layer = NULL;
            }

            if ( simplified_parameters != NULL ) {
                qgd_free( simplified_parameters );
                simplified_parameters = NULL;
            }


            if ( blocks_to_save != NULL ) {
                delete blocks_to_save;
                blocks_to_save = NULL;
            }


            if ( block_to_simplify != NULL ) {
                delete block_to_simplify;
                block_to_simplify = NULL;
            }
        }

        // get the number of CNOT gates in the initial structure
        gates_num gate_num_initial = get_gate_nums();
        int two_qbit_num_initial = gate_num_initial.cnot + gate_num_initial.cz + gate_num_initial.ch;

        // clearing the original list of gates and parameters
        release_gates();
        optimized_parameters_mtx = Matrix_real(0,0);

        // store the modified list of gates and parameters
        combine( gates_loc );
        delete gates_loc;
        gates_loc = NULL;
        layer_num = gates.size();


        optimized_parameters_mtx = optimized_parameters_loc_mtx;

        parameter_num = parameter_num_loc;

        gates_num gate_num_simplified = get_gate_nums();
        int two_qbit_num_simplified = gate_num_simplified.cnot + gate_num_simplified.cz + gate_num_simplified.ch;

        
        sstream.str("");
	sstream << std::endl << std::endl << "************************************" << std::endl;
	sstream << "After some additional 2-qubit decompositions the initial gate structure with " <<  two_qbit_num_initial << " two-qubit gates simplified to a structure containing " << two_qbit_num_simplified << " two-qubit gates" << std::endl;
	sstream << "************************************" << std::endl << std::endl;
	print(sstream, 1);	    	
	
        



}

/**
@brief Call to simplify the gate structure in a block of gates (i.e. tries to reduce the number of two-qubit gates)
@param layer An instance of class Gates_block containing the 2-qubit gate structure to be simplified
@param parameters An array of parameters to calculate the matrix representation of the gates in the block of gates.
@param parameter_num_block NUmber of parameters in the block of gates to be simplified.
@param max_layer_num_loc A map of <int n: int num> indicating the maximal number of two-qubit gates allowed in the simplification.
@param simplified_layer An instance of Gates_block containing the simplified structure of gates.
@param simplified_parameters An array of parameters containing the parameters of the simplified block structure.
@param simplified_parameter_num The number of parameters in the simplified block structure.
@return Returns with 0 if the simplification wa ssuccessful.
*/
int
N_Qubit_Decomposition::simplify_layer( Gates_block* layer, double* parameters, int parameter_num_block, std::map<int,int> max_layer_num_loc, Gates_block* &simplified_layer, double* &simplified_parameters, int &simplified_parameter_num) {

        //The stringstream input to store the output messages.
    	std::stringstream sstream;
	sstream << "Try to simplify sub-structure " << std::endl;
	print(sstream, 1);	    	
	          
        

        // get the target bit
        int target_qbit = -1;
        int control_qbit = -1;

        std::vector<Gate*> layer_gates = layer->get_gates();
        for (std::vector<Gate*>::iterator it = layer_gates.begin(); it!=layer_gates.end(); it++) {
            Gate* op = *it;
            if (op->get_type() == CNOT_OPERATION || op->get_type() == CZ_OPERATION || op->get_type() == CH_OPERATION) {
                target_qbit = op->get_target_qbit();
                control_qbit = op->get_control_qbit();
                break;
            }
        }

        // if there are no target or control qubits, return the initial values
        if ( (target_qbit == -1) || ( control_qbit == -1 ) ) {
            return 1;
        }

        // get the matrix of the two qubit space

        // reorder the control and target qubits to the end of the list
        std::vector<int> qbits_reordered;
        for (int qbit_idx=qbit_num-1; qbit_idx>-1; qbit_idx-- ) { // in range(self.qbit_num-1,-1,-1):
            if (  (qbit_idx != target_qbit) && (qbit_idx != control_qbit) )  {
                qbits_reordered.push_back(qbit_idx);
            }
        }

        qbits_reordered.push_back(control_qbit);
        qbits_reordered.push_back(target_qbit);


        // construct abstarct gates correspond to the reordeerd qubits
        Gates_block* reordered_layer = layer->clone();
        reordered_layer->reorder_qubits( qbits_reordered );

        //  get the reordered N-qubit matrix of the layer
        Matrix_real parameters_mtx( parameters, 1, parameter_num );
        Matrix full_matrix_reordered = reordered_layer->get_matrix( parameters_mtx );
        delete reordered_layer;






        // construct the Two-qubit submatrix from the reordered matrix
        Matrix submatrix = Matrix(4,4);
        for ( int element_idx=0; element_idx<16; element_idx++) {
            int col_idx = element_idx % 4;
            int row_idx = int((element_idx-col_idx)/4);
            submatrix[element_idx].real = full_matrix_reordered[col_idx*matrix_size+row_idx].real;
            submatrix[element_idx].imag = -full_matrix_reordered[col_idx*matrix_size+row_idx].imag;
        }

        // decompose the chosen 2-qubit unitary
        N_Qubit_Decomposition* cdecomposition = new N_Qubit_Decomposition(submatrix, 2, true, config, initial_guess);

        // set the maximal number of layers
        cdecomposition->set_max_layer_num( max_layer_num_loc );

        // suppress output messages
        cdecomposition->set_verbose( false );

        // setting the debugfile name
        cdecomposition->set_debugfile( debugfile_name );

        // starting the decomposition
        cdecomposition->start_decomposition(true, false);



        // check whether simplification was succesfull
        if (!cdecomposition->check_optimization_solution()) {
            // return with the original layer, if the simplification wa snot successfull            
	    std::stringstream sstream;
	    sstream << "The simplification of the sub-structure was not possible" << std::endl;
	    print(sstream, 1);	    	   
            delete cdecomposition;
            return 1;
        }



        // contruct the layer containing the simplified gates in the N-qubit space
        // but first get the inverse reordered qubit list
        std::vector<int> qbits_inverse_reordered;
        for (int idx=0; idx<qbit_num; idx++) {
            qbits_inverse_reordered.push_back(-1);
        }

        for (int qbit_idx=qbit_num-1; qbit_idx>-1; qbit_idx-- ) { // in range(self.qbit_num-1,-1,-1):
            qbits_inverse_reordered[qbit_num-1-qbits_reordered[qbit_num-1-qbit_idx]] =  qbit_idx;
        }

        simplified_layer = new Gates_block(qbit_num);

        std::vector<Gate*> simplifying_gates = cdecomposition->get_gates();
        for ( std::vector<Gate*>::iterator it=simplifying_gates.begin(); it!=simplifying_gates.end(); it++) { //int gate_block_idx=0 in range(0,len(cdecomposition.gates)):
            Gate* op = *it;

            Gates_block* block_op = static_cast<Gates_block*>( op );
            block_op->set_qbit_num( qbit_num );
            block_op->reorder_qubits( qbits_inverse_reordered );
            simplified_layer->combine( block_op );
        }



        simplified_parameter_num = cdecomposition->get_parameter_num();
        simplified_parameters = (double*)qgd_calloc(simplified_parameter_num, sizeof(double), 64);
        cdecomposition->get_optimized_parameters( simplified_parameters );


        gates_num gate_nums_layer = layer->get_gate_nums();
        gates_num gate_nums_simplified = simplified_layer->get_gate_nums();
        
        sstream.str("");
	sstream << gate_nums_layer.cnot+gate_nums_layer.cz+gate_nums_layer.ch << " two-qubit gates successfully simplified to " << gate_nums_simplified.cnot << " CNOT gates" << std::endl;
	print(sstream,1);	  	
	
        
        //release allocated memory
        delete cdecomposition;


        return 0;


}







/**
@brief Call to set custom layers to the gate structure that are intended to be used in the subdecomposition.
@param gate_structure An <int, Gates_block*> map containing the gate structure used in the individual subdecomposition (default is used, if a gate structure for specific subdecomposition is missing).
*/
void N_Qubit_Decomposition::set_custom_gate_structure( std::map<int, Gates_block*> gate_structure_in ) {


    for ( std::map<int,Gates_block*>::iterator it=gate_structure_in.begin(); it!= gate_structure_in.end(); it++ ) {
        int key = it->first;

        std::map<int,Gates_block*>::iterator key_it = gate_structure.find( key );

        if ( key_it != gate_structure.end() ) {
            gate_structure.erase( key_it );
        }

        gate_structure.insert( std::pair<int,Gates_block*>(key, it->second->clone()));

    }

}




/**
@brief Set the number of identical successive blocks during the subdecomposition of the n-th qubit.
@param n The number of qubits for which the maximal number of layers should be used in the subdecomposition.
@param identical_blocks_in The number of successive identical layers used in the subdecomposition.
@return Returns with zero in case of success.
*/
int N_Qubit_Decomposition::set_identical_blocks( int n, int identical_blocks_in )  {

    std::map<int,int>::iterator key_it = identical_blocks.find( n );

    if ( key_it != identical_blocks.end() ) {
        identical_blocks.erase( key_it );
    }

    identical_blocks.insert( std::pair<int, int>(n,  identical_blocks_in) );

    return 0;

}



/**
@brief Set the number of identical successive blocks during the subdecomposition of the n-th qubit.
@param identical_blocks_in An <int,int> map containing the number of successive identical layers used in the subdecompositions.
@return Returns with zero in case of success.
*/
int N_Qubit_Decomposition::set_identical_blocks( std::map<int, int> identical_blocks_in )  {

    for ( std::map<int,int>::iterator it=identical_blocks_in.begin(); it!= identical_blocks_in.end(); it++ ) {
        set_identical_blocks( it->first, it->second );
    }

    return 0;

}




