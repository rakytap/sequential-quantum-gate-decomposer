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
N_Qubit_Decomposition::N_Qubit_Decomposition() {

    // logical value. Set true if finding the minimum number of gate layers is required (default), or false when the maximal number of two-qubit gates is used (ideal for general unitaries).
    optimize_layer_num  = false;

    // A string describing the type of the class
    type = N_QUBIT_DECOMPOSITION_CLASS;

    // The global minimum of the optimization problem
    global_target_minimum = 0;

    
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
N_Qubit_Decomposition::N_Qubit_Decomposition( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, guess_type initial_guess_in= CLOSE_TO_ZERO ) : Decomposition_Base(Umtx_in, qbit_num_in, initial_guess_in) {

    // logical value. Set true if finding the minimum number of gate layers is required (default), or false when the maximal number of two-qubit gates is used (ideal for general unitaries).
    optimize_layer_num  = optimize_layer_num_in;

    // A string describing the type of the class
    type = N_QUBIT_DECOMPOSITION_CLASS;

    // The global minimum of the optimization problem
    global_target_minimum = 0;

    // number of iteratrion loops in the optimization
    iteration_loops[2] = 3;

    // filling in numbers that were not given in the input
    for ( std::map<int,int>::iterator it = max_layer_num_def.begin(); it!=max_layer_num_def.end(); it++) {
        if ( max_layer_num.count( it->first ) == 0 ) {
            max_layer_num.insert( std::pair<int, int>(it->first,  it->second) );
        }
    }


    // initialize custom gate structure
    gate_structure = NULL;


}



/**
@brief Destructor of the class
*/
N_Qubit_Decomposition::~N_Qubit_Decomposition() {

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
N_Qubit_Decomposition::start_decomposition(bool prepare_export) {



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


/*
/// @brief ??????????????????????
struct decomposition_tree_node {
  /// The strored decomposition layer
  Gates_block* layer;
  /// the obtained cost function
  double cost_func_val;
  /// the children nodes in the decomposition tree
  std::vector<decomposition_tree_node*> children;
  /// the child node in the decomposition tree with the minimal cost function
  decomposition_tree_node* minimal_child;
  /// The parent node in the decomposition tree
  decomposition_tree_node* parent;
};
*/

    decomposition_tree_node* parent_node = NULL;
    decomposition_tree_node* minimal_root_node = NULL;
    std::vector<decomposition_tree_node*> children_nodes;


while ( current_minimum > optimization_tolerance ) {


    decomposition_tree_node* minimal_node = NULL;
    children_nodes.clear();


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


                if ( parent_node != NULL ) {
                    // contruct the new gate structure to be optimized
                    add_layers_from_decomposition_tree( minimal_root_node );
                }


                // create the new decomposing layer
                Gates_block* layer = construct_gate_layer(target_qbit_loc, control_qbit_loc);
                add_gate( layer );

                // prepare node to be stored in the decomposition tree
                decomposition_tree_node* current_node = new decomposition_tree_node;
                current_node->layer = layer->clone();

                //add_gate_layers(target_qbit_loc, control_qbit_loc);
                //add_gate_layers(target_qbit_loc, control_qbit_loc);
                //add_gate_layers(target_qbit_loc, control_qbit_loc);

                // add the last layer to rotate all of the qubits into the |0> state
                add_finalyzing_layer();

                tbb::task_arena ta(32);
                ta.execute([&]() {
                    // final tuning of the decomposition parameters
                    final_optimization();

                });   

                // save the current minimum to the current node of the decomposition tree
                current_node->cost_func_val = current_minimum;


                if (minimal_node == NULL) {
                    minimal_node = current_node;
                }
                else {
                    if ( current_node->cost_func_val < minimal_node->cost_func_val ) {
                        minimal_node = current_node;
                    }
                }

                // store the decomposition tree node
                if (parent_node == NULL) {
                    current_node->parent = NULL;
                    root_nodes.push_back(current_node);
                    minimal_root_node = minimal_node;
                }
                else {
                    current_node->parent = parent_node;
                    children_nodes.push_back(current_node);
                }


                if (current_minimum < optimization_tolerance) break;


        }

        if (current_minimum < optimization_tolerance) break;


    }

    if (parent_node != NULL) {
        parent_node->minimal_child =  minimal_node;
        parent_node->children = children_nodes;
    }

    parent_node = minimal_node;
    current_minimum = minimal_node->cost_func_val;

}


    // prepare gates to export
    if (prepare_export) {
        prepare_gates_to_export();
    }

    // calculating the final error of the decomposition
    Matrix matrix_decomposed = get_transformed_matrix(optimized_parameters, gates.begin(), gates.size(), Umtx );
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
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
void 
N_Qubit_Decomposition::add_layers_from_decomposition_tree( const decomposition_tree_node* minimal_root_node ) {


    const decomposition_tree_node* current_node = minimal_root_node;
    while ( current_node != NULL ) {

        Gates_block* layer = current_node->layer->clone();

        // adding the opeartion block to the gates
        add_gate( layer );

std::cout << "another level" << std::endl;    

        if ( current_node->children.size() > 0 ) {
            current_node = current_node->minimal_child;
        }
        else {
            return;
        }

    }



}


/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
Gates_block* 
N_Qubit_Decomposition::construct_gate_layer( const int& _target_qbit, const int& _control_qbit) {


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
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
void 
N_Qubit_Decomposition::add_gate_layers( const int& _target_qbit, const int& _control_qbit) {


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

    // adding the opeartion block to the gates
    add_gate( block );

}


/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
void 
N_Qubit_Decomposition::add_finalyzing_layer() {


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
    add_gate( block );

}

/**
@brief Calculate the error of the decomposition according to the spectral norm of \f$ U-U_{approx} \f$, where \f$ U_{approx} \f$ is the unitary produced by the decomposing quantum cirquit.
@param decomposed_matrix The decomposed matrix, i.e. the result of the decomposing gate structure applied on the initial unitary.
@return Returns with the calculated spectral norm.
*/
void
N_Qubit_Decomposition::calc_decomposition_error(Matrix& decomposed_matrix ) {

	// (U-U_{approx}) (U-U_{approx})^\dagger = 2*I - U*U_{approx}^\dagger - U_{approx}*U^\dagger
	// U*U_{approx}^\dagger = decomposed_matrix_copy
	
 	Matrix A(matrix_size, matrix_size);
	QGD_Complex16* A_data = A.get_data();
	QGD_Complex16* decomposed_data = decomposed_matrix.get_data();
	QGD_Complex16 phase;
	phase.real = decomposed_matrix[0].real/(std::sqrt(decomposed_matrix[0].real*decomposed_matrix[0].real + decomposed_matrix[0].imag*decomposed_matrix[0].imag));
	phase.imag = -decomposed_matrix[0].imag/(std::sqrt(decomposed_matrix[0].real*decomposed_matrix[0].real + decomposed_matrix[0].imag*decomposed_matrix[0].imag));

	for (int idx=0; idx<matrix_size; idx++ ) {
		for (int jdx=0; jdx<matrix_size; jdx++ ) {
			
			if (idx==jdx) {
				QGD_Complex16 mtx_val = mult(phase, decomposed_data[idx*matrix_size+jdx]);
				A_data[idx*matrix_size+jdx].real = 2.0 - 2*mtx_val.real;
				A_data[idx*matrix_size+jdx].imag = 0;
			}
			else {
				QGD_Complex16 mtx_val_ij = mult(phase, decomposed_data[idx*matrix_size+jdx]);
				QGD_Complex16 mtx_val_ji = mult(phase, decomposed_data[jdx*matrix_size+idx]);
				A_data[idx*matrix_size+jdx].real = - mtx_val_ij.real - mtx_val_ji.real;
				A_data[idx*matrix_size+jdx].imag = - mtx_val_ij.imag + mtx_val_ji.imag;
			}

		}
	}


	Matrix alpha(matrix_size, 1);
	Matrix beta(matrix_size, 1);
	Matrix B = create_identity(matrix_size);

	// solve the generalized eigenvalue problem of I- 1/2
	LAPACKE_zggev( CblasRowMajor, 'N', 'N',
                          matrix_size, A.get_data(), matrix_size, B.get_data(),
                          matrix_size, alpha.get_data(),
                          beta.get_data(), NULL, matrix_size, NULL,
                          matrix_size );

	// determine the largest eigenvalue
	double eigval_max = 0;
	for (int idx=0; idx<matrix_size; idx++) {
		double eigval_abs = std::sqrt((alpha[idx].real*alpha[idx].real + alpha[idx].imag*alpha[idx].imag) / (beta[idx].real*beta[idx].real + beta[idx].imag*beta[idx].imag));
		if ( eigval_max < eigval_abs ) eigval_max = eigval_abs;		
	}

	// the norm is the square root of the largest einegvalue.
	decomposition_error = std::sqrt(eigval_max);


}



/**
@brief final optimization procedure improving the accuracy of the decompositin when all the qubits were already disentangled.
*/
void  N_Qubit_Decomposition::final_optimization() {

        if (verbose) {
            printf("***************************************************************\n");
            printf("Final fine tuning of the parameters in the %d-qubit decomposition\n", qbit_num);
            printf("***************************************************************\n");
        }


        //# setting the global minimum
        global_target_minimum = 0;

        if ( optimized_parameters == NULL ) {
std::cout << "iiiiiiiiiiiiiiiiiiiI" << std::endl;
            solve_optimization_problem(NULL, 0);
        }
        else {
            solve_optimization_problem(optimized_parameters, parameter_num);
        }
}



/**
// @brief Call to solve layer by layer the optimization problem. The optimalized parameters are stored in attribute optimized_parameters.
// @param num_of_parameters Number of parameters to be optimized
// @param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void N_Qubit_Decomposition::solve_layer_optimization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl) {

        if (gates.size() == 0 ) {
            return;
        }


        if (solution_guess_gsl == NULL) {
            solution_guess_gsl = gsl_vector_alloc(num_of_parameters);
        }

        if (optimized_parameters == NULL) {
            optimized_parameters = (double*)qgd_calloc(num_of_parameters,sizeof(double), 64);
            memcpy(optimized_parameters, solution_guess_gsl->data, num_of_parameters*sizeof(double) );
        }

        // maximal number of iteration loops
        int iteration_loops_max;
        try {
            iteration_loops_max = std::max(iteration_loops[qbit_num], 1);
        }
        catch (...) {
            iteration_loops_max = 1;
        }

        // do the optimization loops
        for (int idx=0; idx<iteration_loops_max; idx++) {

            size_t iter = 0;
            int status;

            const gsl_multimin_fdfminimizer_type *T;
            gsl_multimin_fdfminimizer *s;

            N_Qubit_Decomposition* par = this;


            gsl_multimin_function_fdf my_func;


            my_func.n = num_of_parameters;
            my_func.f = optimization_problem;
            my_func.df = optimization_problem_grad;
            my_func.fdf = optimization_problem_combined;
            my_func.params = par;


            T = gsl_multimin_fdfminimizer_vector_bfgs2;
            s = gsl_multimin_fdfminimizer_alloc (T, num_of_parameters);


            gsl_multimin_fdfminimizer_set (s, &my_func, solution_guess_gsl, 0.1, 0.1);

            do {
                iter++;

                status = gsl_multimin_fdfminimizer_iterate (s);

                if (status) {
                  break;
                }

                status = gsl_multimin_test_gradient (s->gradient, 1e-1);
                /*if (status == GSL_SUCCESS) {
                    printf ("Minimum found\n");
                }*/

            } while (status == GSL_CONTINUE && iter < 100);

            if (current_minimum > s->f) {
                current_minimum = s->f;
                memcpy( optimized_parameters, s->x->data, num_of_parameters*sizeof(double) );
                gsl_multimin_fdfminimizer_free (s);

                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = solution_guess_gsl->data[jdx] + (2*double(rand())/double(RAND_MAX)-1)*2*M_PI/100;
                }
            }
            else {
                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = solution_guess_gsl->data[jdx] + (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
                }
                gsl_multimin_fdfminimizer_free (s);
            }



        }


}



/**
// @brief The optimization problem of the final optimization
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double N_Qubit_Decomposition::optimization_problem( const double* parameters ) {

        // get the transformed matrix with the gates in the list
        Matrix matrix_new = get_transformed_matrix( parameters, gates.begin(), gates.size(), Umtx );

        double cost_function = get_cost_function(matrix_new);

        return cost_function;
}


/**
// @brief The optimization problem of the final optimization
@param parameters A GNU Scientific Library containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double N_Qubit_Decomposition::optimization_problem( const gsl_vector* parameters, void* void_instance ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(void_instance);
    std::vector<Gate*> gates_loc = instance->get_gates();

    // get the transformed matrix with the gates in the list
    Matrix Umtx_loc = instance->get_Umtx();
    Matrix matrix_new = instance->get_transformed_matrix( parameters->data, gates_loc.begin(), gates_loc.size(), Umtx_loc );

    double cost_function = get_cost_function(matrix_new);

    return cost_function;
}









/**
@brief Call to set custom layers to the gate structure that are intended to be used in the subdecomposition.
@param gate_structure An <int, Gates_block*> map containing the gate structure used in the individual subdecomposition (default is used, if a gate structure for specific subdecomposition is missing).
*/
void N_Qubit_Decomposition::set_custom_gate_structure( Gates_block* gate_structure_in ) {

    gate_structure = gate_structure_in->clone();

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


