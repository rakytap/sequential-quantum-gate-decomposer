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
@brief After the main optimization problem is solved, the indepent qubits can be rotated into state |0> by this def. The constructed gates are added to the array of gates needed to the decomposition of the input unitary.
*/
void N_Qubit_Decomposition::finalize_decomposition() {

        // store the gates, initial unitary and optimized parameters
        Matrix Umtx_save = Umtx.copy();
        Gates_block* gates_save = static_cast<Gates_block*>(this)->clone();
        Matrix_real optimized_parameters_mtx_save = optimized_parameters_mtx;
        

        // get the transformed matrix resulted by the gates in the list
        Umtx = get_transformed_matrix( optimized_parameters_mtx, gates.begin(), gates.size(), Umtx );



        // preallocate the storage for the finalizing parameters
        finalizing_parameter_num = 3*qbit_num;
        
        release_gates();
        optimized_parameters_mtx = Matrix_real(0,0);

        for( int target_qbit=0; target_qbit<qbit_num; target_qbit++) {
            add_u3(target_qbit, true, true, true );
        }

        std::cout << parameter_num << std::endl;

        Matrix_real solution_guess_tmp = Matrix_real(1, parameter_num);
        memset( solution_guess_tmp.get_data(), 0, solution_guess_tmp.size()*sizeof(double) );

        solve_layer_optimization_problem( parameter_num, solution_guess_tmp );

        std::cout << "current_minimum: " << current_minimum << std::endl;

        // combine results
        Gates_block* gates_save2 = static_cast<Gates_block*>(this)->clone();
        Matrix_real optimized_parameters_mtx_save2 = optimized_parameters_mtx;

        release_gates();
        this->combine( gates_save2 );
        this->combine( gates_save );

        Matrix_real parameters_joined( 1, optimized_parameters_mtx_save.size()+optimized_parameters_mtx_save2.size() );
        memcpy( parameters_joined.get_data(), optimized_parameters_mtx_save.get_data(), optimized_parameters_mtx_save.size()*sizeof(double) );

        memcpy( parameters_joined.get_data()+optimized_parameters_mtx_save.size(), 
                 optimized_parameters_mtx_save2.get_data(), 
                 optimized_parameters_mtx_save2.size()*sizeof(double) );

        optimized_parameters_mtx = parameters_joined;
        
        Umtx = Umtx_save;

        Matrix final_matrix = get_transformed_matrix( optimized_parameters_mtx, gates.begin(), gates.size() );

        // indicate that the decomposition was finalized
        decomposition_finalized = true;

        // calculating the final error of the decomposition
        subtract_diag( final_matrix, final_matrix[0] );
        decomposition_error = cblas_dznrm2( matrix_size*matrix_size, (void*)final_matrix.get_data(), 1 );

        // get the number of gates used in the decomposition
        gates_num gates_num = get_gate_nums();


        //The stringstream input to store the output messages.
	std::stringstream sstream;
	sstream << "The error of the decomposition after finalyzing gates is " << decomposition_error << " with " << layer_num << " layers containing " << gates_num.u3 << " U3 gates and " << gates_num.cnot <<  " CNOT gates" << std::endl;
	print(sstream, 1);	    	      

       
        

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
        Matrix_real optimized_parameters_tmp(1, parameter_num+parameter_num_sub_decomp);
        Matrix_real parameters_submatrix = Matrix_real( optimized_parameters_tmp.get_data()+parameter_num, 1, parameter_num_sub_decomp );

        cSub_decomposition->get_optimized_parameters(parameters_submatrix.get_data());

        if ( optimized_parameters_mtx.size() > 0 ) {
            memcpy(optimized_parameters_tmp.get_data(), optimized_parameters_mtx.get_data(), parameter_num*sizeof(double));
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




