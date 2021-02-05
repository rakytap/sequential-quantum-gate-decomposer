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
/*! \file N_Qubit_Decomposition.h
    \brief Header file for a base class to determine the decomposition of an N-qubit unitary into a sequence of CNOT and U3 operations.
*/

#ifndef N_QUBIT_DECOMPOSITION_H
#define N_QUBIT_DECOMPOSITION_H

#include "N_Qubit_Decomposition_Base.h"


/**
@brief A class to determine the decomposition of an Nqubit unitary into a sequence of CNOT and U3 operations.
*/
template <class subdecomp_class>
class N_Qubit_Decomposition : public N_Qubit_Decomposition_Base {


public:


/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition() : N_Qubit_Decomposition_Base() {

    // A string describing the type of the class
    type = N_QUBIT_DECOMPOSITION_CLASS_BASE;

}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, guess_type initial_guess_in= CLOSE_TO_ZERO ) :
    N_Qubit_Decomposition_Base(Umtx_in, qbit_num_in, optimize_layer_num_in, initial_guess_in) {

    // A string describing the type of the class
    type = N_QUBIT_DECOMPOSITION_CLASS_BASE;

}




/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
@param prepare_export Logical parameter. Set true to prepare the list of operations to be exported, or false otherwise.
*/
void start_decomposition(bool finalize_decomp=true, bool prepare_export=true) {



    if (verbose) {
        printf("***************************************************************\n");
        printf("Starting to disentangle %d-qubit matrix\n", qbit_num);
        printf("***************************************************************\n\n\n");
    }

    //initialize TBB task scheduler
    tbb::task_scheduler_init init;

    // temporarily turn off OpenMP parallelism
#if CBLAS==1
    MKL_Set_Num_Threads(1);
#elif CBLAS==2
    openblas_set_num_threads(1);
#endif

    //measure the time for the decompositin
    clock_t start_time = time(NULL);

    // create an instance of class to disentangle the given qubit pair
    subdecomp_class* cSub_decomposition = new subdecomp_class(Umtx, qbit_num, optimize_layer_num, initial_guess);

    // setting the verbosity
    cSub_decomposition->set_verbose( verbose );

    // setting the maximal number of layers used in the subdecomposition
    cSub_decomposition->set_max_layer_num( max_layer_num );

    // setting the number of successive identical blocks used in the subdecomposition
    cSub_decomposition->set_identical_blocks( identical_blocks );

    // setting the iteration loops in each step of the optimization process
    cSub_decomposition->set_iteration_loops( iteration_loops );

    // set custom gate structure if given
    std::map<int,Operation_block*>::iterator key_it = gate_structure.find( qbit_num );
    if ( key_it != gate_structure.end() ) {
        cSub_decomposition->set_custom_operation_layers( gate_structure[qbit_num] );
    }

    // The maximal error of the optimization problem
    cSub_decomposition->set_optimization_tolerance( optimization_tolerance );

    // setting the maximal number of iterations in the disentangling process
    cSub_decomposition->optimization_block = optimization_block;

    // setting the number of operators in one sub-layer of the disentangling process
    //cSub_decomposition->max_iterations = self.max_iterations

    //start to disentangle the qubit pair
    cSub_decomposition->disentangle_submatrices();
    if ( !cSub_decomposition->subdisentaglement_done) {
        return;
    }

    // saving the subunitarization operations
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

        // final tuning of the decomposition parameters
        final_optimization();

        // prepare operations to export
        if (prepare_export) {
            prepare_operations_to_export();
        }

        // calculating the final error of the decomposition
        Matrix matrix_decomposed = get_transformed_matrix(optimized_parameters, operations.begin(), operations.size(), Umtx );

        subtract_diag( matrix_decomposed, matrix_decomposed[0] );

        decomposition_error = cblas_dznrm2( matrix_size*matrix_size, matrix_decomposed.get_data(), 1 );

        // get the number of gates used in the decomposition
        gates_num gates_num = get_gate_nums();

        if (verbose) {
            printf( "In the decomposition with error = %f were used %d layers with %d U3 operations and %d CNOT gates.\n", decomposition_error, layer_num, gates_num.u3, gates_num.cnot );
            printf("--- In total %f seconds elapsed during the decomposition ---\n", float(time(NULL) - start_time));
        }

    }

#if CBLAS==1
    MKL_Set_Num_Threads(num_threads);
#elif CBLAS==2
    openblas_set_num_threads(num_threads);
#endif

}






/**
@brief Start the decompostion process to recursively decompose the submatrices.
*/
void  decompose_submatrix() {

        if (decomposition_finalized) {
            if (verbose) {
                printf("Decomposition was already finalized\n");
            }
            return;
        }

        if (qbit_num == 2) {
            return;
        }

        // obtaining the subdecomposed submatrices
        Matrix subdecomposed_matrix_mtx = get_transformed_matrix( optimized_parameters, operations.begin(), operations.size(), Umtx );
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
        N_Qubit_Decomposition<subdecomp_class>* cdecomposition = new N_Qubit_Decomposition<subdecomp_class>(most_unitary_submatrix_mtx, qbit_num-1, optimize_layer_num, initial_guess);

        // setting the verbosity
        cdecomposition->set_verbose( verbose );


        // Maximal number of iteartions in the optimization process
        cdecomposition->set_max_iteration(max_iterations);

        // Set the number of identical successive blocks in the sub-decomposition
        cdecomposition->set_identical_blocks(identical_blocks);

        // Set the maximal number of layers for the sub-decompositions
        cdecomposition->set_max_layer_num(max_layer_num);

        // setting the iteration loops in each step of the optimization process
        cdecomposition->set_iteration_loops( iteration_loops );

        // setting operation layer
        cdecomposition->set_optimization_blocks( optimization_block );

        // set custom gate structure if given
        cdecomposition->set_custom_gate_structure( gate_structure );

        // set the toleration of the optimization process
        cdecomposition->set_optimization_tolerance( optimization_tolerance );

        // starting the decomposition of the random unitary
        cdecomposition->start_decomposition(false, false);


        // saving the decomposition operations
        extract_subdecomposition_results( reinterpret_cast<subdecomp_class*>(cdecomposition) );

        delete cdecomposition;

}





/**
@brief Call to extract and store the calculated parameters and operations of the sub-decomposition processes
@param cSub_decomposition An instance of class Sub_Matrix_Decomposition used to disentangle the n-th qubit from the others.
*/
void  extract_subdecomposition_results( subdecomp_class* cSub_decomposition ) {

        // get the unitarization parameters
        int parameter_num_sub_decomp = cSub_decomposition->get_parameter_num();

        // adding the unitarization parameters to the ones stored in the class
        double* optimized_parameters_tmp = (double*)qgd_calloc( (parameter_num_sub_decomp+parameter_num),sizeof(double), 64 );
        cSub_decomposition->get_optimized_parameters(optimized_parameters_tmp);

        if ( optimized_parameters != NULL ) {
            memcpy(optimized_parameters_tmp+parameter_num_sub_decomp, optimized_parameters, parameter_num*sizeof(double));
            qgd_free( optimized_parameters );
            optimized_parameters = NULL;
        }

        optimized_parameters = optimized_parameters_tmp;
        optimized_parameters_tmp = NULL;


        // cloning the operation list obtained during the subdecomposition
        std::vector<Operation*> sub_decomp_ops = cSub_decomposition->get_operations();
        int operation_num = cSub_decomposition->get_operation_num();

        for ( int idx = operation_num-1; idx >=0; idx--) {
            Operation* op = sub_decomp_ops[idx];

            if (op->get_type() == CNOT_OPERATION) {
                CNOT* cnot_op = static_cast<CNOT*>( op );
                CNOT* cnot_op_cloned = cnot_op->clone();
                cnot_op_cloned->set_qbit_num( qbit_num );
                Operation* op_cloned = static_cast<Operation*>( cnot_op_cloned );
                add_operation_to_front( op_cloned );
            }
            else if (op->get_type() == U3_OPERATION) {
                U3* u3_op = static_cast<U3*>( op );
                U3* u3_op_cloned = u3_op->clone();
                u3_op_cloned->set_qbit_num( qbit_num );
                Operation* op_cloned = static_cast<Operation*>( u3_op_cloned );
                add_operation_to_front( op_cloned );
            }
            else if (op->get_type() == BLOCK_OPERATION) {
                Operation_block* block_op = static_cast<Operation_block*>( op );
                Operation_block* block_op_cloned = block_op->clone();
                block_op_cloned->set_qbit_num( qbit_num );
                Operation* op_cloned = static_cast<Operation*>( block_op_cloned );
                add_operation_to_front( op_cloned );
            }
            else if (op->get_type() == GENERAL_OPERATION) {
                Operation* op_cloned = op->clone();
                op_cloned->set_qbit_num( qbit_num );
                add_operation_to_front( op_cloned );
            }
        }

}



/**
@brief Call to simplify the gate structure in the layers if possible (i.e. tries to reduce the number of CNOT gates)
*/
void simplify_layers() {

        if (verbose) {
            printf("***************************************************************\n");
            printf("Try to simplify layers\n");
            printf("***************************************************************\n");
        }

        // current starting index of the optimized parameters
        int parameter_idx = 0;

        Operation_block* operations_loc = new Operation_block( qbit_num );
        double* optimized_parameters_loc = (double*)qgd_calloc(parameter_num, sizeof(double), 64);
        unsigned int parameter_num_loc = 0;

        unsigned int layer_idx = 0;

        while (layer_idx < operations.size()) {

            // generate a block of operations to be simplified
            // (containg only successive two-qubit operations)
            Operation_block* block_to_simplify = new Operation_block( qbit_num );

            std::vector<int> involved_qbits;
            // layers in the block to be simplified
            Operation_block* blocks_to_save = new Operation_block( qbit_num );

            // get the successive operations involving the same qubits
            while (true) {

                if (layer_idx >= operations.size() ) {
                    break;
                }

                // get the current layer of operations
                Operation* operation = operations[layer_idx];
                layer_idx = layer_idx + 1;

                Operation_block* block_operation;
                if (operation->get_type() == BLOCK_OPERATION) {
                    block_operation = static_cast<Operation_block*>(operation);
                }
                else {
                    layer_idx = layer_idx + 1;
                    continue;
                }


                //get the involved qubits
                std::vector<int> involved_qbits_op = block_operation->get_involved_qubits();

                // add involved qubits of the operation to the list of involved qubits in the layer
                for (std::vector<int>::iterator it=involved_qbits_op.begin(); it!=involved_qbits_op.end(); it++) {
                    add_unique_elelement( involved_qbits, *it );
                }

                if ( (involved_qbits.size())> 2 && blocks_to_save->get_operation_num() > 0 ) {
                    layer_idx = layer_idx -1;
                    involved_qbits.clear();
                    break;
                }

                blocks_to_save->combine(block_operation);

                // adding the operations to teh block if they act on the same qubits
                block_to_simplify->combine(block_operation);


            }

            //number of perations in the block
            unsigned int parameter_num_block = block_to_simplify->get_parameter_num();

            // get the number of CNOT gates and store the block operations if the number of CNOT gates cannot be reduced
            gates_num gate_nums = block_to_simplify->get_gate_nums();

            if (gate_nums.cnot < 2) {
                operations_loc->combine(blocks_to_save);
                memcpy(optimized_parameters_loc+parameter_num_loc, optimized_parameters+parameter_idx, parameter_num_block*sizeof(double) );
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

                continue;
            }


            // simplify the given layer
            std::map<int,int> max_layer_num_loc;
            max_layer_num_loc.insert( std::pair<int, int>(2,  gate_nums.cnot-1 ) );
            Operation_block* simplified_layer = NULL;
            double* simplified_parameters = NULL;
            unsigned int simplified_parameter_num=0;

            // Try to simplify the sequence of 2-qubit operations
            int simplification_status = simplify_layer( block_to_simplify, optimized_parameters+parameter_idx, parameter_num_block, max_layer_num_loc, simplified_layer, simplified_parameters, simplified_parameter_num );



            // adding the simplified operations (or the non-simplified if the simplification was not successfull)
            if (simplification_status == 0) {
                operations_loc->combine( simplified_layer );

                if (parameter_num < parameter_num_loc + simplified_parameter_num ) {
                    optimized_parameters_loc = (double*)qgd_realloc( optimized_parameters_loc, parameter_num_loc + simplified_parameter_num, sizeof(double), 64 );
                }
                memcpy(optimized_parameters_loc+parameter_num_loc, simplified_parameters, simplified_parameter_num*sizeof(double) );
                parameter_num_loc = parameter_num_loc + simplified_parameter_num;
            }
            else {
                // addign the stacked operation to the list, sice the simplification was unsuccessful
                operations_loc->combine( blocks_to_save );

                if (parameter_num < parameter_num_loc + parameter_num_block ) {
                    optimized_parameters_loc = (double*)qgd_realloc( optimized_parameters_loc, parameter_num_loc + parameter_num_block, sizeof(double), 64 );
                }
                memcpy(optimized_parameters_loc+parameter_num_loc, optimized_parameters+parameter_idx, parameter_num_block*sizeof(double) );
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
        int cnot_num_initial = gate_num_initial.cnot;

        // clearing the original list of operations and parameters
        release_operations();
        qgd_free( optimized_parameters );
        optimized_parameters = NULL;

        // store the modified list of operations and parameters
        combine( operations_loc );
        delete operations_loc;
        operations_loc = NULL;
        layer_num = operations.size();


        optimized_parameters = optimized_parameters_loc;

        parameter_num = parameter_num_loc;

        gates_num gate_num_simplified = get_gate_nums();
        int cnot_num_simplified = gate_num_simplified.cnot;

        if (verbose) {
            printf("\n\n************************************\n");
            printf("After some additional 2-qubit decompositions the initial gate structure with %d CNOT gates simplified to a structure containing %d CNOT gates.\n", cnot_num_initial, cnot_num_simplified);
            printf("************************************\n\n");
        }



}

/**
@brief Call to simplify the gate structure in a block of operations (i.e. tries to reduce the number of CNOT gates)
@param layer An instance of class Operation_block containing the 2-qubit gate structure to be simplified
@param parameters An array of parameters to calculate the matrix representation of the operations in the block of operations.
@param parameter_num_block NUmber of parameters in the block of operations to be simplified.
@param max_layer_num_loc A map of <int n: int num> indicating the maximal number of CNOT operations allowed in the simplification.
@param simplified_layer An instance of Operation_block containing the simplified structure of operations.
@param simplified_parameters An array of parameters containing the parameters of the simplified block structure.
@param simplified_parameter_num The number of parameters in the simplified block structure.
@return Returns with 0 if the simplification wa ssuccessful.
*/
int simplify_layer( Operation_block* layer, double* parameters, unsigned int parameter_num_block, std::map<int,int> max_layer_num_loc, Operation_block* &simplified_layer, double* &simplified_parameters, unsigned int &simplified_parameter_num) {

        if (verbose) {
            printf("Try to simplify sub-structure \n");
        }

        // get the target bit
        int target_qbit = -1;
        int control_qbit = -1;

        std::vector<Operation*> layer_operations = layer->get_operations();
        for (std::vector<Operation*>::iterator it = layer_operations.begin(); it!=layer_operations.end(); it++) {
            Operation* op = *it;
            if (op->get_type() == CNOT_OPERATION) {
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

        // construct abstarct operations correspond to the reordeerd qubits
        Operation_block* reordered_layer = layer->clone();
        reordered_layer->reorder_qubits( qbits_reordered );

        //  get the reordered N-qubit matrix of the layer
        Matrix full_matrix_reordered = reordered_layer->get_matrix( parameters );
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
        N_Qubit_Decomposition<subdecomp_class>* cdecomposition = new N_Qubit_Decomposition<subdecomp_class>(submatrix, 2, true, initial_guess);

        // set the maximal number of layers
        cdecomposition->set_max_layer_num( max_layer_num_loc );

        // suppress output messages
        cdecomposition->set_verbose( false );

        // starting the decomposition
        cdecomposition->start_decomposition(true, false);



        // check whether simplification was succesfull
        if (!cdecomposition->check_optimization_solution()) {
            // return with the original layer, if the simplification wa snot successfull
            if (verbose) {
                printf("The simplification of the sub-structure was not possible\n");
            }
            delete cdecomposition;
            return 1;
        }



        // contruct the layer containing the simplified operations in the N-qubit space
        // but first get the inverse reordered qubit list
        std::vector<int> qbits_inverse_reordered;
        for (int idx=0; idx<qbit_num; idx++) {
            qbits_inverse_reordered.push_back(-1);
        }

        for (int qbit_idx=qbit_num-1; qbit_idx>-1; qbit_idx-- ) { // in range(self.qbit_num-1,-1,-1):
            qbits_inverse_reordered[qbit_num-1-qbits_reordered[qbit_num-1-qbit_idx]] =  qbit_idx;
        }

        simplified_layer = new Operation_block(qbit_num);

        std::vector<Operation*> simplifying_operations = cdecomposition->get_operations();
        for ( std::vector<Operation*>::iterator it=simplifying_operations.begin(); it!=simplifying_operations.end(); it++) { //int operation_block_idx=0 in range(0,len(cdecomposition.operations)):
            Operation* op = *it;

            Operation_block* block_op = static_cast<Operation_block*>( op );
            block_op->set_qbit_num( qbit_num );
            block_op->reorder_qubits( qbits_inverse_reordered );
            simplified_layer->combine( block_op );
        }



        simplified_parameter_num = cdecomposition->get_parameter_num();
        simplified_parameters = (double*)qgd_calloc(simplified_parameter_num, sizeof(double), 64);
        cdecomposition->get_optimized_parameters( simplified_parameters );


        gates_num gate_nums_layer = layer->get_gate_nums();
        gates_num gate_nums_simplified = simplified_layer->get_gate_nums();
        if (verbose) {
            printf("%d CNOT gates successfully simplified to %d CNOT gates\n", gate_nums_layer.cnot, gate_nums_simplified.cnot);
        }


        //release allocated memory
        delete cdecomposition;


        return 0;


}







/**
@brief Create a clone of the present class.
@return Return with a pointer pointing to the cloned object.
*/
N_Qubit_Decomposition<subdecomp_class>* clone() {


    N_Qubit_Decomposition<subdecomp_class>* ret = new N_Qubit_Decomposition<subdecomp_class>(Umtx, qbit_num, optimize_layer_num, initial_guess);

    // setting computational parameters
    ret->set_identical_blocks( identical_blocks );
    ret->set_max_iteration( max_iterations );
    ret->set_optimization_blocks( optimization_block );
    ret->set_max_layer_num( max_layer_num );
    ret->set_iteration_loops( iteration_loops );

    if ( extract_operations(static_cast<Operation_block*>(ret)) != 0 ) {
        printf("N_Qubit_Decomposition::clone(): extracting operations was not succesfull\n");
        exit(-1);
    }

    return ret;

}



};



#endif
