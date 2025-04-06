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
/*! \file Optimization_Interface.cpp
    \brief Class implementing optimization engines
*/

#include "Optimization_Interface.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "Adam.h"
#include "grad_descend.h"
#include "BFGS_Powell.h"
#include "Bayes_Opt.h"

#include "RL_experience.h"

#include <fstream>


#ifdef __DFE__
#include "common_DFE.h"
#endif




extern "C" int LAPACKE_dgesv( 	int  matrix_layout, int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb); 	


tbb::spin_mutex my_mutex_optimization_interface;


/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
Optimization_Interface::Optimization_Interface() {

    // logical value. Set true if finding the minimum number of gate layers is required (default), or false when the maximal number of two-qubit gates is used (ideal for general unitaries).
    optimize_layer_num  = false;

    // A string describing the type of the class
    type = N_QUBIT_DECOMPOSITION_CLASS_BASE;

    // The global minimum of the optimization problem
    global_target_minimum = 0;

    // logical variable indicating whether adaptive learning reate is used in the ADAM algorithm
    adaptive_eta = true;

    // parameter to contron the radius of parameter randomization around the curren tminimum
    radius = 1.0;
    randomization_rate = 0.3;

    // The chosen variant of the cost function
    cost_fnc = FROBENIUS_NORM;
    
    number_of_iters = 0;
     
 

    // variables to calculate the cost function with first and second corrections    
    prev_cost_fnv_val = 1.0;
    correction1_scale = 1/1.7;
    correction2_scale = 1/2.0;  


    // number of utilized accelerators
    accelerator_num = 0;

    // set the trace offset
    trace_offset = 0;

    // unique id indentifying the instance of the class
    std::uniform_int_distribution<> distrib_int(0, INT_MAX);  
    int id = distrib_int(gen);


    // Time spent on circuit simulation/cost function evaluation
    circuit_simulation_time = 0.0;
    // time spent on optimization
    CPU_time = 0.0;

}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
Optimization_Interface::Optimization_Interface( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, std::map<std::string, Config_Element>& config, guess_type initial_guess_in= CLOSE_TO_ZERO, int accelerator_num_in ) : Decomposition_Base(Umtx_in, qbit_num_in, config, initial_guess_in) {

    // logical value. Set true if finding the minimum number of gate layers is required (default), or false when the maximal number of two-qubit gates is used (ideal for general unitaries).
    optimize_layer_num  = optimize_layer_num_in;

    // A string describing the type of the class
    type = N_QUBIT_DECOMPOSITION_CLASS_BASE;

    // The global minimum of the optimization problem
    global_target_minimum = 0;
    
    number_of_iters = 0;
         

    // number of iteratrion loops in the optimization
    iteration_loops[2] = 3;

    // filling in numbers that were not given in the input
    for ( std::map<int,int>::iterator it = max_layer_num_def.begin(); it!=max_layer_num_def.end(); it++) {
        if ( max_layer_num.count( it->first ) == 0 ) {
            max_layer_num.insert( std::pair<int, int>(it->first,  it->second) );
        }
    }

    // logical variable indicating whether adaptive learning reate is used in the ADAM algorithm
    adaptive_eta = true;

    // parameter to contron the radius of parameter randomization around the curren tminimum
    radius = 1.0;
    randomization_rate = 0.3;

    // The chosen variant of the cost function
    cost_fnc = FROBENIUS_NORM;


    // variables to calculate the cost function with first and second corrections    
    prev_cost_fnv_val = 1.0;
    correction1_scale = 1/1.7;
    correction2_scale = 1/2.0; 


    // set the trace offset
    trace_offset = 0;

    // unique id indentifying the instance of the class
    std::uniform_int_distribution<> distrib_int(0, INT_MAX);  
    id = distrib_int(gen);



    // Time spent on circuit simulation/cost function evaluation
    circuit_simulation_time = 0.0;
    // time spent on optimization
    CPU_time = 0.0;

#ifdef __DFE__

    // number of utilized accelerators
    accelerator_num = accelerator_num_in;
#else
    accelerator_num = 0;
#endif

}



/**
@brief Destructor of the class
*/
Optimization_Interface::~Optimization_Interface() {


#ifdef __DFE__
    if ( Umtx.cols == Umtx.rows && qbit_num >= 2 && get_accelerator_num() > 0 ) {
        unload_dfe_lib();//releive_DFE();
    }
#endif



}


/**
@brief Call to print out into a file the current cost function and the second Rényi entropy on the subsystem made of qubits 0 and 1.
@param current_minimum The current minimum (to avoid calculating it again
*/
void Optimization_Interface::export_current_cost_fnc(double current_minimum ){

    export_current_cost_fnc( current_minimum, optimized_parameters_mtx);
    
}



/**
@brief Call to print out into a file the current cost function and the second Rényi entropy on the subsystem made of qubits 0 and 1.
@param current_minimum The current minimum (to avoid calculating it again
@param parameters Parameters to be used in the calculations (For Rényi entropy)
*/
void Optimization_Interface::export_current_cost_fnc(double current_minimum, Matrix_real& parameters){

    FILE* pFile;
    std::string filename("costfuncs_and_entropy.txt");
    
    if (project_name != ""){filename = project_name + "_" + filename;}

        const char* c_filename = filename.c_str();
	pFile = fopen(c_filename, "a");

        if (pFile==NULL) {
            fputs ("File error",stderr); 
            std::string error("Cannot open file.");
            throw error;
    }

    Matrix input_state(Power_of_2(qbit_num),1);

    std::uniform_int_distribution<> distrib(0, qbit_num-2); 

    memset(input_state.get_data(), 0.0, (input_state.size()*2)*sizeof(double) ); 
    input_state[0].real = 1.0;

    matrix_base<int> qbit_sublist(1,2);
    qbit_sublist[0] = 0;//distrib(gen);
    qbit_sublist[1] = 1;//qbit_sublist[0]+1;

    double renyi_entropy = get_second_Renyi_entropy(parameters, input_state, qbit_sublist);

    fprintf(pFile,"%i\t%f\t%f\n", (int)number_of_iters, current_minimum, renyi_entropy);
    fclose(pFile);

    return;
}


/**
@brief Call to print out into a file the current cost function and the second Rényi entropy on the subsystem made of qubits 0 and 1.
@param current_minimum The current minimum (to avoid calculating it again
@param parameters Parameters to be used in the calculations (For Rényi entropy)
@param instance A pointer pointing ti the current class instance.
*/
void Optimization_Interface::export_current_cost_fnc(double current_minimum, Matrix_real& parameters, void* void_instance) {


    Optimization_Interface* instance = reinterpret_cast<Optimization_Interface*>(void_instance);
    
    instance->export_current_cost_fnc( current_minimum, parameters );

}


/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
void 
Optimization_Interface::add_finalyzing_layer() {


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
Optimization_Interface::calc_decomposition_error(Matrix& decomposed_matrix ) {

	// (U-U_{approx}) (U-U_{approx})^\dagger = 2*I - U*U_{approx}^\dagger - U_{approx}*U^\dagger
	// U*U_{approx}^\dagger = decomposed_matrix_copy
	
 	/*Matrix A(matrix_size, matrix_size);
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
    
	// the norm is the square root of the largest einegvalue.*/
    if ( cost_fnc == FROBENIUS_NORM ) {
        decomposition_error =  get_cost_function(decomposed_matrix);
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
        Matrix_real&& ret = get_cost_function_with_correction(decomposed_matrix, qbit_num);
        decomposition_error = ret[0] - std::sqrt(prev_cost_fnv_val)*ret[1]*correction1_scale;
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
        Matrix_real&& ret = get_cost_function_with_correction2(decomposed_matrix, qbit_num);
        decomposition_error = ret[0] - std::sqrt(prev_cost_fnv_val)*(ret[1]*correction1_scale + ret[2]*correction2_scale);
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST){
       decomposition_error = get_hilbert_schmidt_test(decomposed_matrix);
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION1 ){
        Matrix&& ret = get_trace_with_correction(decomposed_matrix, qbit_num);
        double d = 1.0/decomposed_matrix.cols;
        decomposition_error = 1 - d*d*(ret[0].real*ret[0].real+ret[0].imag*ret[0].imag+std::sqrt(prev_cost_fnv_val)*correction1_scale*(ret[1].real*ret[1].real+ret[1].imag*ret[1].imag));
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION2 ){
        Matrix&& ret = get_trace_with_correction2(decomposed_matrix, qbit_num);
        double d = 1.0/decomposed_matrix.cols;
        decomposition_error = 1 - d*d*(ret[0].real*ret[0].real+ret[0].imag*ret[0].imag+std::sqrt(prev_cost_fnv_val)*(correction1_scale*(ret[1].real*ret[1].real+ret[1].imag*ret[1].imag)+correction2_scale*(ret[2].real*ret[2].real+ret[2].imag*ret[2].imag)));
    }
    else if ( cost_fnc == SUM_OF_SQUARES) {
        decomposition_error = get_cost_function_sum_of_squares(decomposed_matrix);
    }
    else {
        std::string err("Optimization_Interface::optimization_problem: Cost function variant not implmented.");
        throw err;
    }

}



/**
@brief final optimization procedure improving the accuracy of the decompositin when all the qubits were already disentangled.
*/
void  Optimization_Interface::final_optimization() {

	//The stringstream input to store the output messages.
	std::stringstream sstream;
	sstream << "***************************************************************" << std::endl;
	sstream << "Final fine tuning of the parameters in the " << qbit_num << "-qubit decomposition" << std::endl;
	sstream << "***************************************************************" << std::endl;
	print(sstream, 1);	    	

         


        //# setting the global minimum
        global_target_minimum = 0;

        if ( optimized_parameters_mtx.size() == 0 ) {
            solve_optimization_problem(NULL, 0);
        }
        else {
            current_minimum = optimization_problem(optimized_parameters_mtx.get_data());
            if ( check_optimization_solution() ) return;

            solve_optimization_problem(optimized_parameters_mtx.get_data(), parameter_num);
        }
}




/**
@brief Call to solve layer by layer the optimization problem via calling one of the implemented algorithms. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess Array containing the solution guess.
*/
void Optimization_Interface::solve_layer_optimization_problem( int num_of_parameters, Matrix_real solution_guess) {


    switch ( alg ) {
        case ADAM:
            solve_layer_optimization_problem_ADAM( num_of_parameters, solution_guess);
            return;
        case ADAM_BATCHED:
            solve_layer_optimization_problem_ADAM_BATCHED( num_of_parameters, solution_guess);
            return;
        case GRAD_DESCEND:
            solve_layer_optimization_problem_GRAD_DESCEND( num_of_parameters, solution_guess);
            return;
        case AGENTS:
            solve_layer_optimization_problem_AGENTS( num_of_parameters, solution_guess);
            return;
        case COSINE:
            solve_layer_optimization_problem_COSINE( num_of_parameters, solution_guess);
            return;
        case GRAD_DESCEND_PARAMETER_SHIFT_RULE:
            solve_layer_optimization_problem_GRAD_DESCEND_PARAMETER_SHIFT_RULE( num_of_parameters, solution_guess);
            return;           
        case AGENTS_COMBINED:
            solve_layer_optimization_problem_AGENTS_COMBINED( num_of_parameters, solution_guess);
            return;
        case BFGS:
            solve_layer_optimization_problem_BFGS( num_of_parameters, solution_guess);
            return;
        case BAYES_OPT:
            solve_layer_optimization_problem_BAYES_OPT( num_of_parameters, solution_guess);
            return;
        case BAYES_AGENTS:
            solve_layer_optimization_problem_BAYES_AGENTS( num_of_parameters, solution_guess);
            return;
        case BFGS2:
            solve_layer_optimization_problem_BFGS2( num_of_parameters, solution_guess);
            return;
        default:
            std::string error("Optimization_Interface::solve_layer_optimization_problem: unimplemented optimization algorithm");
            throw error;
    }


}

/**
@brief Call to randomize the parameter.
@param input The parameters are randomized around the values stores in this array
@param output The randomized parameters are stored within this array
@param f0 weight in the randomiztaion (output = input + rand()*sqrt(f0) ).
*/
void Optimization_Interface::randomize_parameters( Matrix_real& input, Matrix_real& output, const double& f0  ) {

    // random generator of real numbers   
    std::uniform_real_distribution<> distrib_prob(0.0, 1.0);
    std::uniform_real_distribution<> distrib_real(-2*M_PI, 2*M_PI);


    double radius_loc;
    if ( config.count("Randomized_Radius") > 0 ) {
        config["Randomized_Radius"].get_property( radius_loc );  
    }
    else {
        radius_loc = radius;
    }

    const int num_of_parameters = input.size();

    int changed_parameters = 0;
    for ( int jdx=0; jdx<num_of_parameters; jdx++) {
        if ( distrib_prob(gen) <= randomization_rate ) {
            output[jdx] = (cost_fnc!=VQE) ? input[jdx] + distrib_real(gen)*std::sqrt(f0)*radius_loc : input[jdx] + distrib_real(gen)*radius_loc;
            changed_parameters++;
        }
        else {
            output[jdx] = input[jdx];
        }
    }

#ifdef __MPI__  
        //MPI_Bcast( (void*)output.get_data(), num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif


}


/**
@brief The cost function of the optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double Optimization_Interface::optimization_problem( double* parameters ) {

    // get the transformed matrix with the gates in the list
    Matrix_real parameters_mtx(parameters, 1, parameter_num );
      
    
    return optimization_problem( parameters_mtx );


}


/**
@brief The cost function of the optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double Optimization_Interface::optimization_problem( Matrix_real& parameters ) {

    // get the transformed matrix with the gates in the list
    if ( parameters.size() != parameter_num ) {
        std::stringstream sstream;
	sstream << "Optimization_Interface::optimization_problem: Number of free paramaters should be " << parameter_num << ", but got " << parameters.size() << std::endl;
        print(sstream, 0);
        std::string err("Optimization_Interface::optimization_problem: Wrong number of parameters.");
        throw err;
    }  
    
    Matrix matrix_new = Umtx.copy();
    apply_to( parameters, matrix_new );

    if ( cost_fnc == FROBENIUS_NORM ) {
        return get_cost_function(matrix_new, trace_offset);
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
        Matrix_real&& ret = get_cost_function_with_correction(matrix_new, qbit_num, trace_offset);
        return ret[0] - std::sqrt(prev_cost_fnv_val)*ret[1]*correction1_scale;
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
        Matrix_real&& ret = get_cost_function_with_correction2(matrix_new, qbit_num, trace_offset);
        return ret[0] - std::sqrt(prev_cost_fnv_val)*(ret[1]*correction1_scale + ret[2]*correction2_scale);
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST){
        return get_hilbert_schmidt_test(matrix_new);
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION1 ){
        Matrix&& ret = get_trace_with_correction(matrix_new, qbit_num);
        double d = 1.0/matrix_new.cols;
        return 1 - d*d*(ret[0].real*ret[0].real+ret[0].imag*ret[0].imag+std::sqrt(prev_cost_fnv_val)*correction1_scale*(ret[1].real*ret[1].real+ret[1].imag*ret[1].imag));
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION2 ){
        Matrix&& ret = get_trace_with_correction2(matrix_new, qbit_num);
        double d = 1.0/matrix_new.cols;
        return 1 - d*d*(ret[0].real*ret[0].real+ret[0].imag*ret[0].imag+std::sqrt(prev_cost_fnv_val)*(correction1_scale*(ret[1].real*ret[1].real+ret[1].imag*ret[1].imag)+correction2_scale*(ret[2].real*ret[2].real+ret[2].imag*ret[2].imag)));
    }
    else if ( cost_fnc == SUM_OF_SQUARES) {
        return get_cost_function_sum_of_squares(matrix_new);
    }
    else {
        std::string err("Optimization_Interface::optimization_problem: Cost function variant not implmented.");
        throw err;
    }

}


/**
@brief The cost function of the optimization with batched input (implemented only for the Frobenius norm cost function)
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
Matrix_real 
Optimization_Interface::optimization_problem_batched( std::vector<Matrix_real>& parameters_vec) {

tbb::tick_count t0_DFE = tbb::tick_count::now();    



    Matrix_real cost_fnc_mtx(parameters_vec.size(), 1);

    int parallel = get_parallel_configuration();
             
#ifdef __DFE__
    if ( Umtx.cols == Umtx.rows && get_accelerator_num() > 0 ) {
        int gatesNum, gateSetNum, redundantGateSets;
        DFEgate_kernel_type* DFEgates = convert_to_batched_DFE_gates( parameters_vec, gatesNum, gateSetNum, redundantGateSets );                        
            
        Matrix_real trace_DFE_mtx(gateSetNum, 3);
        
        
        
        number_of_iters = number_of_iters + parameters_vec.size();  
       
        
   
#ifdef __MPI__
        // the number of decomposing layers are divided between the MPI processes

        int mpi_gateSetNum = gateSetNum / world_size;
        int mpi_starting_gateSetIdx = gateSetNum/world_size * current_rank;

        Matrix_real mpi_trace_DFE_mtx(mpi_gateSetNum, 3);
        

        number_of_iters = number_of_iters + mpi_gateSetNum;   
        

        lock_lib();
        calcqgdKernelDFE( Umtx.rows, Umtx.cols, DFEgates+mpi_starting_gateSetIdx*gatesNum, gatesNum, mpi_gateSetNum, trace_offset, mpi_trace_DFE_mtx.get_data() );
        unlock_lib();

        int bytes = mpi_trace_DFE_mtx.size()*sizeof(double);
        MPI_Allgather(mpi_trace_DFE_mtx.get_data(), bytes, MPI_BYTE, trace_DFE_mtx.get_data(), bytes, MPI_BYTE, MPI_COMM_WORLD);

#else
        lock_lib();
        calcqgdKernelDFE( Umtx.rows, Umtx.cols, DFEgates, gatesNum, gateSetNum, trace_offset, trace_DFE_mtx.get_data() );
        unlock_lib();    
                                                                      
#endif  // __MPI__
      

        // calculate the cost function
        if ( cost_fnc == FROBENIUS_NORM ) {
            for ( int idx=0; idx<parameters_vec.size(); idx++ ) {
                cost_fnc_mtx[idx] = 1-trace_DFE_mtx[idx*3]/Umtx.cols;
            }
        }
        else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
            for ( int idx=0; idx<parameters_vec.size(); idx++ ) {
                cost_fnc_mtx[idx] = 1-(trace_DFE_mtx[idx*3] + std::sqrt(prev_cost_fnv_val)*trace_DFE_mtx[idx*3+1]*correction1_scale)/Umtx.cols;
            }
        }
        else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
            for ( int idx=0; idx<parameters_vec.size(); idx++ ) {
                cost_fnc_mtx[idx] = 1-(trace_DFE_mtx[idx*3] + std::sqrt(prev_cost_fnv_val)*(trace_DFE_mtx[idx*3+1]*correction1_scale + trace_DFE_mtx[idx*3+2]*correction2_scale))/Umtx.cols;
            }
        }
        else {
            std::string err("Optimization_Interface::optimization_problem_batched: Cost function variant not implmented for DFE.");
            throw err;
        }


       


        delete[] DFEgates;

    }
    else{

#endif // __DFE__

#ifdef __MPI__


        // the number of decomposing layers are divided between the MPI processes

        int batch_element_num           = parameters_vec.size();
        int mpi_batch_element_num       = batch_element_num / world_size;
        int mpi_batch_element_remainder = batch_element_num % world_size;

        if ( mpi_batch_element_remainder > 0 ) {
            std::string err("Optimization_Interface::optimization_problem_batched: The size of the batch should be divisible with the number of processes.");
            throw err;
        }

        int mpi_starting_batchIdx = mpi_batch_element_num * current_rank;


        Matrix_real cost_fnc_mtx_loc(mpi_batch_element_num, 1);

        int work_batch = 1;
        if( parallel==0) {
            work_batch = mpi_batch_element_num;
        }

        tbb::parallel_for( tbb::blocked_range<int>(0, (int)mpi_batch_element_num, work_batch), [&](tbb::blocked_range<int> r) {
            for (int idx=r.begin(); idx<r.end(); ++idx) { 
                cost_fnc_mtx_loc[idx] = optimization_problem( parameters_vec[idx + mpi_starting_batchIdx] );
            }
        });

        //number_of_iters = number_of_iters + mpi_batch_element_num; 



        int bytes = cost_fnc_mtx_loc.size()*sizeof(double);
        MPI_Allgather(cost_fnc_mtx_loc.get_data(), bytes, MPI_BYTE, cost_fnc_mtx.get_data(), bytes, MPI_BYTE, MPI_COMM_WORLD);


#else

        int work_batch = 1;
        if( parallel==0) {
            work_batch = parameters_vec.size();
        }

        tbb::parallel_for( tbb::blocked_range<int>(0, (int)parameters_vec.size(), work_batch), [&](tbb::blocked_range<int> r) {
            for (int idx=r.begin(); idx<r.end(); ++idx) { 
                cost_fnc_mtx[idx] = optimization_problem( parameters_vec[idx] );
            }
        });


#endif  // __MPI__
       


#ifdef __DFE__  
    }
#endif

circuit_simulation_time += (tbb::tick_count::now() - t0_DFE).seconds();       
    return cost_fnc_mtx;
        
}





/**
@brief The static cost function of the optimization
@param parameters Array containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param ret_temp A matrix to store trace in for gradient for HS test 
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double Optimization_Interface::optimization_problem( Matrix_real parameters, void* void_instance, Matrix ret_temp) {

    Optimization_Interface* instance = reinterpret_cast<Optimization_Interface*>(void_instance);
    std::vector<Gate*> gates_loc = instance->get_gates();

    {
        tbb::spin_mutex::scoped_lock my_lock{my_mutex_optimization_interface};

        number_of_iters++;
        
    }

    // get the transformed matrix with the gates in the list
    Matrix Umtx_loc = instance->get_Umtx();
    Matrix matrix_new = Umtx_loc.copy();
    instance->apply_to( parameters, matrix_new );
   
    cost_function_type cost_fnc = instance->get_cost_function_variant();

    if ( cost_fnc == FROBENIUS_NORM ) {
        return get_cost_function(matrix_new, instance->get_trace_offset());
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
        double correction1_scale    = instance->get_correction1_scale();
        Matrix_real&& ret = get_cost_function_with_correction(matrix_new, instance->get_qbit_num(), instance->get_trace_offset());
        return ret[0] - 0*std::sqrt(instance->get_previous_cost_function_value())*ret[1]*correction1_scale;
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
        double correction1_scale    = instance->get_correction1_scale();
        double correction2_scale    = instance->get_correction2_scale();            
        Matrix_real&& ret = get_cost_function_with_correction2(matrix_new, instance->get_qbit_num(), instance->get_trace_offset());
        return ret[0] - std::sqrt(instance->get_previous_cost_function_value())*(ret[1]*correction1_scale + ret[2]*correction2_scale);
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST){
    	QGD_Complex16 trace_temp = get_trace(matrix_new);
    	ret_temp[0].real = trace_temp.real;
    	ret_temp[0].imag = trace_temp.imag;
    	double d = 1.0/matrix_new.cols;
        return 1.0-d*d*trace_temp.real*trace_temp.real-d*d*trace_temp.imag*trace_temp.imag;
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION1 ){
        double correction1_scale = instance->get_correction1_scale();
        Matrix&& ret = get_trace_with_correction(matrix_new, instance->get_qbit_num());
        double d = 1.0/matrix_new.cols;
        for (int idx=0; idx<3; idx++){ret_temp[idx].real=ret[idx].real;ret_temp[idx].imag=ret[idx].imag;}
        return 1.0 - d*d*(ret[0].real*ret[0].real+ret[0].imag*ret[0].imag+std::sqrt(instance->get_previous_cost_function_value())*correction1_scale*(ret[1].real*ret[1].real+ret[1].imag*ret[1].imag));
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION2 ){
        double correction1_scale    = instance->get_correction1_scale();
        double correction2_scale    = instance->get_correction2_scale();            
        Matrix&& ret = get_trace_with_correction2(matrix_new, instance->get_qbit_num());
        double d = 1.0/matrix_new.cols;
        for (int idx=0; idx<4; idx++){ret_temp[idx].real=ret[idx].real;ret_temp[idx].imag=ret[idx].imag;}
        return 1.0 - d*d*(ret[0].real*ret[0].real+ret[0].imag*ret[0].imag+std::sqrt(instance->get_previous_cost_function_value())*(correction1_scale*(ret[1].real*ret[1].real+ret[1].imag*ret[1].imag)+correction2_scale*(ret[2].real*ret[2].real+ret[2].imag*ret[2].imag)));
    }
    else if ( cost_fnc == SUM_OF_SQUARES) {
        return get_cost_function_sum_of_squares(matrix_new);
    }
    else {
        std::string err("Optimization_Interface::optimization_problem: Cost function variant not implmented.");
        throw err;
    }


}



/**
@brief The cost function of the optimization
@param parameters Array containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double Optimization_Interface::optimization_problem_non_static( Matrix_real parameters, void* void_instance) {

    Optimization_Interface* instance = reinterpret_cast<Optimization_Interface*>(void_instance);
    Matrix ret(1,3);
    double cost_func = instance->optimization_problem(parameters, void_instance, ret);
    return cost_func;
}




/**
@brief The cost function of the optimization
@param parameters Array containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double Optimization_Interface::optimization_problem( Matrix_real parameters, void* void_instance){
    Optimization_Interface* instance = reinterpret_cast<Optimization_Interface*>(void_instance);
    return instance->optimization_problem_non_static(parameters, void_instance);
}







/**
@brief Calculate the derivative of the cost function with respect to the free parameters.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad Array containing the calculated gradient components.
*/
void Optimization_Interface::optimization_problem_grad( Matrix_real parameters, void* void_instance, Matrix_real& grad ) {

    // The function value at x0
    double f0;

    // calculate the approximate gradient
    optimization_problem_combined( parameters, void_instance, &f0, grad);

}





/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
void Optimization_Interface::optimization_problem_combined_non_static( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad ) {

    Optimization_Interface* instance = reinterpret_cast<Optimization_Interface*>(void_instance);

    int parallel = instance->get_parallel_configuration();

    // the number of free parameters
    int parameter_num_loc = instance->get_parameter_num();

    // the variant of the cost function
    cost_function_type cost_fnc = instance->get_cost_function_variant();

    // value of the cost function from the previous iteration to weigth the correction to the trace
    double prev_cost_fnv_val = instance->get_previous_cost_function_value();
    double correction1_scale    = instance->get_correction1_scale();
    double correction2_scale    = instance->get_correction2_scale();    

    int qbit_num = instance->get_qbit_num();
    int trace_offset_loc = instance->get_trace_offset();

#ifdef __DFE__

///////////////////////////////////////
//std::cout << "number of qubits: " << instance->qbit_num << std::endl;
//tbb::tick_count t0_DFE = tbb::tick_count::now();/////////////////////////////////    
if ( Umtx.cols == Umtx.rows && instance->qbit_num >= 5 && instance->get_accelerator_num() > 0 ) {

    int gatesNum, redundantGateSets, gateSetNum;
    DFEgate_kernel_type* DFEgates = instance->convert_to_DFE_gates_with_derivates( parameters, gatesNum, gateSetNum, redundantGateSets );

    Matrix&& Umtx_loc = instance->get_Umtx();   
    Matrix_real trace_DFE_mtx(gateSetNum, 3);


#ifdef __MPI__
    // the number of decomposing layers are divided between the MPI processes

    int mpi_gateSetNum = gateSetNum / instance->world_size;
    int mpi_starting_gateSetIdx = gateSetNum/instance->world_size * instance->current_rank;

    Matrix_real mpi_trace_DFE_mtx(mpi_gateSetNum, 3);
    
    instance->number_of_iters = instance->number_of_iters + mpi_gateSetNum;    

    lock_lib();
    calcqgdKernelDFE( Umtx_loc.rows, Umtx_loc.cols, DFEgates+mpi_starting_gateSetIdx*gatesNum, gatesNum, mpi_gateSetNum, trace_offset_loc, mpi_trace_DFE_mtx.get_data() );
    unlock_lib();

    int bytes = mpi_trace_DFE_mtx.size()*sizeof(double);
    MPI_Allgather(mpi_trace_DFE_mtx.get_data(), bytes, MPI_BYTE, trace_DFE_mtx.get_data(), bytes, MPI_BYTE, MPI_COMM_WORLD);

#else

    instance->number_of_iters = instance->number_of_iters + gateSetNum;    

    lock_lib();
    calcqgdKernelDFE( Umtx_loc.rows, Umtx_loc.cols, DFEgates, gatesNum, gateSetNum, trace_offset_loc, trace_DFE_mtx.get_data() );
    unlock_lib();

#endif  

    std::stringstream sstream;
    sstream << *f0 << " " << 1.0 - trace_DFE_mtx[0]/Umtx_loc.cols << " " << trace_DFE_mtx[1]/Umtx_loc.cols << " " << trace_DFE_mtx[2]/Umtx_loc.cols << std::endl;
    instance->print(sstream, 5);	
    
  
    if ( cost_fnc == FROBENIUS_NORM ) {
        *f0 = 1-trace_DFE_mtx[0]/Umtx_loc.cols;
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
        *f0 = 1 - (trace_DFE_mtx[0] + std::sqrt(prev_cost_fnv_val)*trace_DFE_mtx[1]*correction1_scale)/Umtx_loc.cols;
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
        *f0 = 1 - (trace_DFE_mtx[0] + std::sqrt(prev_cost_fnv_val)*(trace_DFE_mtx[1]*correction1_scale + trace_DFE_mtx[2]*correction2_scale))/Umtx_loc.cols;
    }
    else {
        std::string err("Optimization_Interface::optimization_problem_combined: Cost function variant not implmented.");
        throw err;
    }

    //double f0_DFE = *f0;

    //Matrix_real grad_components_DFE_mtx(1, parameter_num_loc);
    for (int idx=0; idx<parameter_num_loc; idx++) {

        if ( cost_fnc == FROBENIUS_NORM ) {
            grad[idx] = -trace_DFE_mtx[3*(idx+1)]/Umtx_loc.cols;
        }
        else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
            grad[idx] = -(trace_DFE_mtx[3*(idx+1)] + std::sqrt(prev_cost_fnv_val)*trace_DFE_mtx[3*(idx+1)+1]*correction1_scale)/Umtx_loc.cols;
        }
        else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
            grad[idx] = -(trace_DFE_mtx[3*(idx+1)] + std::sqrt(prev_cost_fnv_val)*(trace_DFE_mtx[3*(idx+1)+1]*correction1_scale + trace_DFE_mtx[3*(idx+1)+2]*correction2_scale))/Umtx_loc.cols;
        }
        else {
            std::string err("Optimization_Interface::optimization_problem_combined: Cost function variant not implmented.");
            throw err;
        }

        //grad_components_DFE_mtx[idx] = gsl_vector_get( grad, idx );


    }

    delete[] DFEgates;

//tbb::tick_count t1_DFE = tbb::tick_count::now();/////////////////////////////////
//std::cout << "uploaded data to DFE: " << (int)(gatesNum*gateSetNum*sizeof(DFEgate_kernel_type)) << " bytes" << std::endl;
//std::cout << "time elapsed DFE: " << (t1_DFE-t0_DFE).seconds() << ", expected time: " << (((double)Umtx_loc.rows*(double)Umtx_loc.cols*gatesNum*gateSetNum/get_chained_gates_num()/4 + 4578*3*get_chained_gates_num()))/350000000 + 0.001<< std::endl;

///////////////////////////////////////
}
else {

#endif

#ifdef __DFE__
tbb::tick_count t0_CPU = tbb::tick_count::now();/////////////////////////////////
#endif

    Matrix_real cost_function_terms;

    // vector containing gradients of the transformed matrix
    std::vector<Matrix> Umtx_deriv;
    Matrix trace_tmp(1,3);

    int work_batch = 10;

    if( parallel == 0 ) {

        *f0 = instance->optimization_problem(parameters, reinterpret_cast<void*>(instance), trace_tmp); 
        Matrix&& Umtx_loc = instance->get_Umtx();   
        Umtx_deriv = instance->apply_derivate_to( parameters, Umtx_loc, parallel );
       
        work_batch = parameter_num_loc;
    }
    else {

        tbb::parallel_invoke(
            [&]{
                *f0 = instance->optimization_problem(parameters, reinterpret_cast<void*>(instance), trace_tmp); 
            },
            [&]{
                Matrix&& Umtx_loc = instance->get_Umtx();   
                Umtx_deriv = instance->apply_derivate_to( parameters, Umtx_loc, parallel );
            }
        );

        work_batch = 10;
    }


    tbb::parallel_for( tbb::blocked_range<int>(0,parameter_num_loc,work_batch), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 


        //for( int idx=0; idx<parameter_num_loc; idx++ ) {

            double grad_comp;
            if ( cost_fnc == FROBENIUS_NORM ) {
                grad_comp = (get_cost_function(Umtx_deriv[idx], trace_offset_loc) - 1.0);
            }
            else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
                Matrix_real deriv_tmp = get_cost_function_with_correction( Umtx_deriv[idx], qbit_num, trace_offset_loc );
                grad_comp = (deriv_tmp[0] - std::sqrt(prev_cost_fnv_val)*deriv_tmp[1]*correction1_scale - 1.0);
            }
            else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
                Matrix_real deriv_tmp = get_cost_function_with_correction2( Umtx_deriv[idx], qbit_num, trace_offset_loc );
                grad_comp = (deriv_tmp[0] - std::sqrt(prev_cost_fnv_val)*(deriv_tmp[1]*correction1_scale + deriv_tmp[2]*correction2_scale) - 1.0);
            }
            else if (cost_fnc == HILBERT_SCHMIDT_TEST){
                double d = 1.0/Umtx_deriv[idx].cols;
                QGD_Complex16 deriv_tmp = get_trace(Umtx_deriv[idx]);
                grad_comp = -2.0*d*d*trace_tmp[0].real*deriv_tmp.real-2.0*d*d*trace_tmp[0].imag*deriv_tmp.imag;
            }
            else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION1 ){
                Matrix&&  deriv_tmp = get_trace_with_correction( Umtx_deriv[idx], qbit_num);
                double d = 1.0/Umtx_deriv[idx].cols;
                grad_comp = -2.0*d*d* (trace_tmp[0].real*deriv_tmp[0].real+trace_tmp[0].imag*deriv_tmp[0].imag+std::sqrt(prev_cost_fnv_val)*correction1_scale*(trace_tmp[1].real*deriv_tmp[1].real+trace_tmp[1].imag*deriv_tmp[1].imag));
            }
            else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION2 ){
                Matrix&& deriv_tmp = get_trace_with_correction2( Umtx_deriv[idx], qbit_num);
                double d = 1.0/Umtx_deriv[idx].cols;
                grad_comp = -2.0*d*d* (trace_tmp[0].real*deriv_tmp[0].real+trace_tmp[0].imag*deriv_tmp[0].imag+std::sqrt(prev_cost_fnv_val)*(correction1_scale*(trace_tmp[1].real*deriv_tmp[1].real+trace_tmp[1].imag*deriv_tmp[1].imag) + correction2_scale*(trace_tmp[2].real*deriv_tmp[2].real+trace_tmp[2].imag*deriv_tmp[2].imag)));
            }
            else if ( cost_fnc == SUM_OF_SQUARES) {
                grad_comp = get_cost_function_sum_of_squares(Umtx_deriv[idx]);
            }
            else {
                std::string err("Optimization_Interface::optimization_problem_combined: Cost function variant not implmented.");
                throw err;
            }
            
            grad[idx] = grad_comp;



        }

    });

    instance->number_of_iters = instance->number_of_iters + parameter_num_loc + 1;  

    std::stringstream sstream;
    sstream << *f0 << std::endl;
    instance->print(sstream, 5);	

#ifdef __DFE__
}
#endif


}



/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
void Optimization_Interface::optimization_problem_combined( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad ){
    Optimization_Interface* instance = reinterpret_cast<Optimization_Interface*>(void_instance);
    instance->optimization_problem_combined_non_static(parameters, void_instance, f0, grad );
    return;
}


/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
void Optimization_Interface::optimization_problem_combined( Matrix_real parameters, double* f0, Matrix_real grad ) {

    optimization_problem_combined( parameters, this, f0, grad );
    return;
}



/**
@brief Call to calculate both the effect of the circuit on th eunitary and it's gradient componets.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param Umtx The unitary on which the circuit is applied in place.
@param Umtx_deriv Array containing the calculated gradient components.
*/
void Optimization_Interface::optimization_problem_combined_unitary( Matrix_real parameters, void* void_instance, Matrix& Umtx, std::vector<Matrix>& Umtx_deriv ) {
    // vector containing gradients of the transformed matrix
    Optimization_Interface* instance = reinterpret_cast<Optimization_Interface*>(void_instance);

    int parallel = instance->get_parallel_configuration();

    if ( parallel ) {

        Matrix Umtx_loc = instance->get_Umtx(); 
        Umtx = Umtx_loc.copy();    
        instance->apply_to( parameters, Umtx );


        Umtx_loc = instance->get_Umtx();
        Umtx_deriv = instance->apply_derivate_to( parameters, Umtx_loc, parallel );

    }
    else {

        tbb::parallel_invoke(
            [&]{
                Matrix Umtx_loc = instance->get_Umtx(); 
                Umtx = Umtx_loc.copy();    
                instance->apply_to( parameters, Umtx );
            },
            [&]{
                Matrix Umtx_loc = instance->get_Umtx();
                Umtx_deriv = instance->apply_derivate_to( parameters, Umtx_loc, parallel );
            });

    }



  



}


/**
@brief Call to calculate both the effect of the circuit on th eunitary and it's gradient componets.
@param parameters Array containing the free parameters to be optimized.
@param Umtx The unitary on which the circuit is applied in place.
@param Umtx_deriv Array containing the calculated gradient components.
*/
void Optimization_Interface::optimization_problem_combined_unitary( Matrix_real parameters, Matrix& Umtx, std::vector<Matrix>& Umtx_deriv ) {

    optimization_problem_combined_unitary( parameters, this, Umtx, Umtx_deriv);
    return;
    
}


/**
@brief Call to get the variant of the cost function used in the calculations
*/
cost_function_type 
Optimization_Interface::get_cost_function_variant() {

    return cost_fnc;

}


/**
@brief Call to set the variant of the cost function used in the calculations
@param variant The variant of the cost function from the enumaration cost_function_type
*/
void 
Optimization_Interface::set_cost_function_variant( cost_function_type variant  ) {

    cost_fnc = variant;

    std::stringstream sstream;
    sstream << "Optimization_Interface::set_cost_function_variant: Cost function variant set to " << cost_fnc << std::endl;
    print(sstream, 2);	


}



/**
@brief Call to set the number of iterations for which an optimization engine tries to solve the optimization problem
@param max_inner_iterations_in The number of iterations for which an optimization engine tries to solve the optimization problem 
*/
void Optimization_Interface::set_max_inner_iterations( int max_inner_iterations_in  ) {

    max_inner_iterations = max_inner_iterations_in;
    
}



/**
@brief Call to set the maximal number of parameter randomization tries to escape a local minimum.
@param random_shift_count_max_in The number of maximal number of parameter randomization tries to escape a local minimum.
*/
void Optimization_Interface::set_random_shift_count_max( int random_shift_count_max_in  ) {

    random_shift_count_max = random_shift_count_max_in;

}


/**
@brief Call to set the optimizer engine to be used in solving the optimization problem.
@param alg_in The chosen algorithm
*/
void Optimization_Interface::set_optimizer( optimization_aglorithms alg_in ) {

    alg = alg_in;

    switch ( alg ) {
        case ADAM:
            max_inner_iterations = 1e5; 
            random_shift_count_max = 100;
            max_outer_iterations = 1;
            return;

        case ADAM_BATCHED:
            max_inner_iterations = 2.5e3;
            random_shift_count_max = 3;
            max_outer_iterations = 1;
            return;

        case GRAD_DESCEND:
            max_inner_iterations = 10000;
            random_shift_count_max = 1;  
            max_outer_iterations = 1e8; 
            return;

        case COSINE:
            max_inner_iterations = 2.5e3;
            random_shift_count_max = 3;
            max_outer_iterations = 1;
            return;
            
        case GRAD_DESCEND_PARAMETER_SHIFT_RULE:
            max_inner_iterations = 2.5e3;
            random_shift_count_max = 3;
            max_outer_iterations = 1;
            return;                       

        case AGENTS:
            max_inner_iterations = 2.5e3;
            random_shift_count_max = 3;
            max_outer_iterations = 1;
            return;

        case AGENTS_COMBINED:
            max_inner_iterations = 2.5e3;
            random_shift_count_max = 3;
            max_outer_iterations = 1;
            return;

        case BFGS:
            max_inner_iterations = 10000;
            random_shift_count_max = 1;  
            max_outer_iterations = 1e8; 
            return;

        case BFGS2:
            max_inner_iterations = 1e5;
            random_shift_count_max = 100;
            max_outer_iterations = 1;
            return;
        
        case BAYES_OPT:
            max_inner_iterations = 100;
            random_shift_count_max = 100;
            max_outer_iterations = 1;
            return;
        case BAYES_AGENTS:
            max_inner_iterations = 100;
            random_shift_count_max = 100;
            max_outer_iterations = 1;
            return;

        default:
            std::string error("Optimization_Interface::set_optimizer: unimplemented optimization algorithm");
            throw error;
    }



}




/**
@brief Call to retrieve the previous value of the cost funtion to be used to evaluate bitflip errors in the cost funtion (see Eq. (21) in arXiv:2210.09191)
*/
double 
Optimization_Interface::get_previous_cost_function_value() {

    return prev_cost_fnv_val;

}



/**
@brief Call to get the pre factor of the single-bitflip errors in the cost function. (see Eq. (21) in arXiv:2210.09191)
@return Returns with the prefactor of the single-bitflip errors in the cost function. 
*/
double 
Optimization_Interface::get_correction1_scale() {

    return correction1_scale;

}



/**
@brief Call to get the pre factor of the two-bitflip errors in the cost function. (see Eq. (21) in arXiv:2210.09191)
@return Returns with the prefactor of the two-bitflip errors in the cost function. 
*/
double 
Optimization_Interface::get_correction2_scale() {

    return correction2_scale;

}






/**
@brief Get the number of iterations.
*/
int 
Optimization_Interface::get_num_iters() {

    return number_of_iters;

}


/**
@brief Call to set custom layers to the gate structure that are intended to be used in the subdecomposition.
@param gate_structure An <int, Gates_block*> map containing the gate structure used in the individual subdecomposition (default is used, if a gate structure for specific subdecomposition is missing).
*/
void 
Optimization_Interface::set_custom_gate_structure( Gates_block* gate_structure_in ) {

    release_gates();

    set_qbit_num( gate_structure_in->get_qbit_num() );

    combine( gate_structure_in );

}


/**
@brief Get the trace offset used in the evaluation of the cost function
*/
int 
Optimization_Interface::get_trace_offset() {

    return trace_offset;

}


/**
@brief Set the trace offset used in the evaluation of the cost function
*/
void 
Optimization_Interface::set_trace_offset(int trace_offset_in) {


    if ( (trace_offset_in + Umtx.cols) > Umtx.rows ) {
        std::string error("Optimization_Interface::set_trace_offset: trace offset must be smaller or equal to the difference of the rows and columns in the input unitary.");
        throw error;

    }

    
    trace_offset = trace_offset_in;


    std::stringstream sstream;
    sstream << "Optimization_Interface::set_trace_offset: trace offset set to " << trace_offset << std::endl;
    print(sstream, 2);	

}


#ifdef __DFE__

void 
Optimization_Interface::upload_Umtx_to_DFE() {
    if (Umtx.cols == Umtx.rows) {
        lock_lib();

        if ( get_initialize_id() != id ) {
            // initialize DFE library
            init_dfe_lib( accelerator_num, qbit_num, id );
        }

        uploadMatrix2DFE( Umtx );


        unlock_lib();
    }

}


/**
@brief Get the number of accelerators to be reserved on DFEs on users demand. 
*/
int 
Optimization_Interface::get_accelerator_num() {

    return accelerator_num;

}


#endif

