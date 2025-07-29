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
/*! \file Generative_Quantum_Machine_Learning_Base.cpp
    \brief Class to solve GQML problems
*/
#include "Generative_Quantum_Machine_Learning_Base.h"
#include <iostream>

static tbb::spin_mutex my_mutex;

/**
@brief A base class to solve GQML problems
This class can be used to approximate a given distribution via a quantum circuit
*/
Generative_Quantum_Machine_Learning_Base::Generative_Quantum_Machine_Learning_Base() {

    // logical value describing whether the decomposition was finalized or not
    decomposition_finalized = false;


    // error of the unitarity of the final decomposition
    decomposition_error = DBL_MAX;


    // The current minimum of the optimization problem
    current_minimum = DBL_MAX;
    
    global_target_minimum = -DBL_MAX;

    // logical value describing whether the optimization problem was solved or not
    optimization_problem_solved = false;


    // The maximal allowed error of the optimization problem
    optimization_tolerance = -DBL_MAX;

    // The convergence threshold in the optimization process
    convergence_threshold = -DBL_MAX;
    
    alg = AGENTS;
    

    random_shift_count_max = 100;
    
    adaptive_eta = false;
    
    cost_fnc = GQML;
    
    ansatz = HEA;

    ev_P_star_P_star = -1.0;
    
}




/**
@brief Constructor of the class.
@param x_vectors_in The input data indices
@param x_bitstrings_in The input data bitstrings
@param P_star_in The distribution to approximate
@param sigma_in Parameter of the gaussian kernels
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param config_in A map that can be used to set hyperparameters during the process
@return An instance of the class
*/
Generative_Quantum_Machine_Learning_Base::Generative_Quantum_Machine_Learning_Base(std::vector<int> x_vectors_in, std::vector<std::vector<int>> x_bitstrings_in, Matrix_real P_star_in, double sigma_in, int qbit_num_in, std::map<std::string, Config_Element>& config_in) : Optimization_Interface(Matrix(Power_of_2(qbit_num_in),1), qbit_num_in, false, config_in, RANDOM, accelerator_num) {

	x_vectors = x_vectors_in;

    P_star = P_star_in;
    // config maps
    config   = config_in;
    // logical value describing whether the decomposition was finalized or not
    decomposition_finalized = false;


    // error of the unitarity of the final decomposition
    decomposition_error = DBL_MAX;


    // The current minimum of the optimization problem
    current_minimum = DBL_MAX;


    // logical value describing whether the optimization problem was solved or not
    optimization_problem_solved = false;

    // override optimization parameters governing the convergence used in gate decomposition applications
    global_target_minimum  = -DBL_MAX;
    optimization_tolerance = -DBL_MAX;
    convergence_threshold  = -DBL_MAX;
    
   

    random_shift_count_max = 100;
    
    adaptive_eta = false;
    
    qbit_num = qbit_num_in;
	

	
    std::random_device rd;  
    
    gen = std::mt19937(rd());
    
    alg = BAYES_OPT;
    
    cost_fnc = GQML;
    
    ansatz = HEA;
    
    ev_P_star_P_star = -1.0;

    sigma = sigma_in;

    x_bitstrings = x_bitstrings_in;
}




/**
@brief Destructor of the class
*/
Generative_Quantum_Machine_Learning_Base::~Generative_Quantum_Machine_Learning_Base(){

}




/**
@brief Call to start solving the GQML problem
*/ 
void Generative_Quantum_Machine_Learning_Base::start_optimization(){

    // initialize the initial state if it was not given
    if ( initial_state.size() == 0 ) {
        initialize_zero_state();
    }


    if (gates.size() == 0 ) {
        std::string error("Variational_Quantum_Eigensolver_Base::Get_ground_state: for GQML process the circuit needs to be initialized");
        throw error;
    }

    int num_of_parameters =  optimized_parameters_mtx.size();
    if ( num_of_parameters == 0 ) {
        std::string error("Variational_Quantum_Eigensolver_Base::Get_ground_state: No intial parameters were given");
        throw error;
    }


    if ( num_of_parameters != get_parameter_num() ) {
        std::string error("Variational_Quantum_Eigensolver_Base::Get_ground_state: The number of initial parameters does not match with the number of parameters in the circuit");
        throw error;
    }    

    ev_P_star_P_star = expectation_value_P_star_P_star();

    // start the GQML process
    Matrix_real solution_guess = optimized_parameters_mtx.copy();
    solve_layer_optimization_problem(num_of_parameters, solution_guess);


    return;
}

/**
@brief Call to evaluate the value of one gaussian kernel function
@param x The index of the first input data
@param y The index of the second input data
@param sigma The parameters of the kernel
@return The calculated value of the kernel function
*/
double Generative_Quantum_Machine_Learning_Base::Gaussian_kernel(int x, int y, double sigma) {
    // The norm stores the distance between the two data points (the more qbit they differ in the bigger it is)
    double norm=0;
    for (int idx=0; idx<qbit_num; idx++) {
        norm += (x_bitstrings[x][idx] - x_bitstrings[y][idx])*(x_bitstrings[x][idx]-x_bitstrings[y][idx]);
    }
    double norm_squared = norm*norm;
    double result = (exp(-norm_squared*0.5*4)+ exp(-norm_squared*0.5*0.1)+ exp(-norm_squared*0.5*0.001))/3;
    return result;
}

/**
@brief Call to evaluate the expectation value of the square of distribution we want to approximate
@return The calculated value of the expectation value of the square of the distribution we want to approximate
*/
double Generative_Quantum_Machine_Learning_Base::expectation_value_P_star_P_star() {
    double ev=0.0;
    tbb::combinable<double> priv_partial_ev{[](){return 0.0;}};
    tbb::parallel_for( tbb::blocked_range<int>(0, x_vectors.size(), 1024), [&](tbb::blocked_range<int> r) {
        double& ev_local = priv_partial_ev.local();
        for (int idx1=r.begin(); idx1<r.end(); idx1++) {
            for (int idx2=0; idx2<x_vectors.size(); idx2++) {
                // if (idx1 != idx2) {
                    ev_local += P_star[x_vectors[idx1]]*P_star[x_vectors[idx2]]*Gaussian_kernel(idx1, idx2, sigma);
                // }
            }
        }
    });
    priv_partial_ev.combine_each([&ev](double a) {
        ev += a;
    });
    ev /= x_vectors.size()*x_vectors.size();
    return ev;
}

/**
@brief Call to evaluate the total variational distance of the given distribution and the one created by our circuit
@param State_right The state on the right for which the expectation value is evaluated. It is a column vector.
@return The calculated total variational distance of the distributions
*/
double Generative_Quantum_Machine_Learning_Base::TV_of_the_distributions(Matrix& State_right) {
    std::vector<double> P_theta;

    for (size_t x_idx=0; x_idx<State_right.size(); x_idx++){
        P_theta.push_back(State_right[x_idx].real*State_right[x_idx].real +State_right[x_idx].imag*State_right[x_idx].imag);
    }

    double TV = 0.0;
    for (int i=0; i<P_theta.size(); i++) {
        TV += abs(P_theta[i]-P_star[i]);
    }
    return TV*0.5;
}


/**
@brief Call to evaluate the maximum mean discrepancy of the given distribution and the one created by our circuit
@param State_left The state on the let for which the expectation value is evaluated. It is a column vector. In the sandwich product it is transposed and conjugated inside the function.
@param State_right The state on the right for which the expectation value is evaluated. It is a column vector.
@return The calculated mmd
*/
double Generative_Quantum_Machine_Learning_Base::MMD_of_the_distributions( Matrix& State_left, Matrix& State_right ) {
    if ( State_left.rows != State_right.rows) {
        std::string error("Variational_Quantum_Eigensolver_Base::Expectation_value_of_energy_real: States on the right and left should be of the same dimension as the Hamiltonian");
        throw error;
    }

    // If the ev of the P_star hasnt been evaluated we need to evaluate it
    if (ev_P_star_P_star < 0 ) {
        ev_P_star_P_star = expectation_value_P_star_P_star();
    }

    // We calculate the distribution created by our circuit at the given traing data points "we sample our distribution"
    std::vector<double> P_theta;
    for (size_t x_idx=0; x_idx<x_vectors.size(); x_idx++){
        QGD_Complex16 x_times_state = State_right[x_vectors[x_idx]];

        P_theta.push_back(x_times_state.real*x_times_state.real + x_times_state.imag*x_times_state.imag);
    }


    // Calculate the expectation values 
    double ev_P_theta_P_theta   = 0.0;
    double ev_P_theta_P_star    = 0.0;
    tbb::combinable<double> priv_partial_ev_P_theta_P_theta{[](){return 0.0;}};
    tbb::combinable<double> priv_partial_ev_P_theta_P_star{[](){return 0.0;}};
    tbb::parallel_for( tbb::blocked_range<int>(0, x_vectors.size(), 1024), [&](tbb::blocked_range<int> r) {
        double& ev_P_theta_P_theta_local = priv_partial_ev_P_theta_P_theta.local();
        double& ev_P_theta_P_star_local = priv_partial_ev_P_theta_P_star.local();
        for (int idx1=r.begin(); idx1<r.end(); idx1++) {
            for (int idx2=0; idx2 < x_vectors.size(); idx2++) {
                // if(idx1 != idx2)
                    ev_P_theta_P_theta_local += P_theta[idx1]*P_theta[idx2]*Gaussian_kernel(idx1, idx2, sigma);
                ev_P_theta_P_star_local += P_theta[idx1]*P_star[x_vectors[idx2]]*Gaussian_kernel(idx1, idx2, sigma);
            }
        }
    });
    priv_partial_ev_P_theta_P_theta.combine_each([&ev_P_theta_P_theta](double a) {
        ev_P_theta_P_theta += a;
    });
    priv_partial_ev_P_theta_P_star.combine_each([&ev_P_theta_P_star](double a) {
        ev_P_theta_P_star += a;
    });

    ev_P_theta_P_theta /= x_vectors.size()*x_vectors.size();
    ev_P_theta_P_star /= x_vectors.size()*x_vectors.size();

    {
        tbb::spin_mutex::scoped_lock my_lock{my_mutex};

        number_of_iters++;
        
    }
    double result = ev_P_theta_P_theta + ev_P_star_P_star - 2*ev_P_theta_P_star;
    return result;
}




/**
@brief The optimization problem of the final optimization
@param parameters Array containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double Generative_Quantum_Machine_Learning_Base::optimization_problem_non_static(Matrix_real parameters, void* void_instance){

    double MMD=0.0;

    Generative_Quantum_Machine_Learning_Base* instance = reinterpret_cast<Generative_Quantum_Machine_Learning_Base*>(void_instance);

    Matrix State = instance->initial_state.copy();

    instance->apply_to(parameters, State);
    MMD = instance->MMD_of_the_distributions(State, State);

    return MMD;
}


/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double Generative_Quantum_Machine_Learning_Base::optimization_problem(Matrix_real& parameters)  {

    // initialize the initial state if it was not given
    if ( initial_state.size() == 0 ) {
        initialize_zero_state();
    }
	
    Matrix State = initial_state.copy();
	
    apply_to(parameters, State);
	
	
    double MMD = MMD_of_the_distributions(State, State);
	
    return MMD;
}



/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
void Generative_Quantum_Machine_Learning_Base::optimization_problem_combined_non_static( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad ) {

    Generative_Quantum_Machine_Learning_Base* instance = reinterpret_cast<Generative_Quantum_Machine_Learning_Base*>(void_instance);

    // initialize the initial state if it was not given
    if ( instance->initial_state.size() == 0 ) {
        instance->initialize_zero_state();
    }

    // the number of free parameters
    int parameter_num_loc = parameters.size();

    Matrix_real cost_function_terms;

    // vector containing gradients of the transformed matrix
    std::vector<Matrix> State_deriv;
    Matrix State;

    int parallel = get_parallel_configuration();

    tbb::parallel_invoke(
        [&]{
            State = instance->initial_state.copy();
            instance->apply_to(parameters, State);
            *f0 = instance->MMD_of_the_distributions(State, State);
            
        },
        [&]{
            Matrix State_loc = instance->initial_state.copy();

            State_deriv = instance->apply_derivate_to( parameters, State_loc, parallel );
            State_loc.release_data();
    });

    tbb::parallel_for( tbb::blocked_range<int>(0,parameter_num_loc,2), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 
            grad[idx] = 2*instance->MMD_of_the_distributions(State_deriv[idx], State);
        }
    });
    
    /*
    double delta = 0.0000001;
    
    tbb::parallel_for( tbb::blocked_range<int>(0,parameter_num_loc,1), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 
            Matrix_real param_tmp = parameters.copy();
            param_tmp[idx] += delta;
            
            Matrix State = instance->initial_state.copy();
            instance->apply_to(param_tmp, State);
            double f_loc = instance->Expectation_value_of_energy_real(State, State);
            
            
            grad[idx] = (f_loc-*f0)/delta;
        }
    });    

*/
    return;
}






/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
void Generative_Quantum_Machine_Learning_Base::optimization_problem_combined( Matrix_real parameters, double* f0, Matrix_real grad )  {

    optimization_problem_combined_non_static( parameters, this, f0, grad );
    return;
}




/**
@brief Calculate the derivative of the cost function with respect to the free parameters.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad Array containing the calculated gradient components.
*/
void Generative_Quantum_Machine_Learning_Base::optimization_problem_grad_vqe( Matrix_real parameters, void* void_instance, Matrix_real& grad ) {

    double f0;
    Generative_Quantum_Machine_Learning_Base* instance = reinterpret_cast<Generative_Quantum_Machine_Learning_Base*>(void_instance);
    instance->optimization_problem_combined_non_static(parameters, void_instance, &f0, grad);
    return;

}




/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double Generative_Quantum_Machine_Learning_Base::optimization_problem( double* parameters ) {

    // get the transformed matrix with the gates in the list
    Matrix_real parameters_mtx(parameters, 1, parameter_num );
    
    return optimization_problem( parameters_mtx );


}

/**
@brief Call to print out into a file the current cost function and the second Rényi entropy on the subsystem made of qubits 0 and 1, and the TV distance.
@param current_minimum The current minimum (to avoid calculating it again
@param parameters Parameters to be used in the calculations (For Rényi entropy)
*/
void Generative_Quantum_Machine_Learning_Base::export_current_cost_fnc(double current_minimum, Matrix_real& parameters){

    FILE* pFile;
    std::string filename("costfuncs_entropy_and_tv.txt");
    
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

    // initialize the initial state if it was not given
    if ( initial_state.size() == 0 ) {
        initialize_zero_state();
    }
	
    Matrix State = initial_state.copy();
    apply_to(parameters, State);

    double tv_distance = TV_of_the_distributions(State);

    fprintf(pFile,"%i\t%f\t%f\t%f\n", (int)number_of_iters, current_minimum, renyi_entropy, tv_distance);
    fclose(pFile);

    return;
}



/**
@brief Initialize the state used in the quantun circuit. All qubits are initialized to state 0
*/
void Generative_Quantum_Machine_Learning_Base::initialize_zero_state( ) {

    initial_state = Matrix( 1 << qbit_num , 1);


    initial_state[0].real = 1.0;
    initial_state[0].imag = 0.0;
    memset(initial_state.get_data()+2, 0.0, (initial_state.size()*2-2)*sizeof(double) );      

    return;
}





/**
@brief Call to set the ansatz type. Currently imp
@param ansatz_in The ansatz type . Possible values: "HEA" (hardware efficient ansatz with U3 and CNOT gates).
*/ 
void Generative_Quantum_Machine_Learning_Base::set_ansatz(ansatz_type ansatz_in){

    ansatz = ansatz_in;
    
    return;
}





/**
@brief Call to generate the circuit ansatz
@param layers The number of layers. The depth of the generated circuit is 2*layers+1 (U3-CNOT-U3-CNOT...CNOT)
@param inner_blocks The number of U3-CNOT repetition within a single layer
*/
void Generative_Quantum_Machine_Learning_Base::generate_circuit( int layers, int inner_blocks ) {


    switch (ansatz){
    
        case HEA:
        {

            release_gates();

            if ( qbit_num < 2 ) {
                std::string error("Variational_Quantum_Eigensolver_Base::generate_initial_circuit: number of qubits should be at least 2");
                throw error;
            }

            for (int layer_idx=0; layer_idx<layers ;layer_idx++){

                for( int idx=0; idx<inner_blocks; idx++) {
                    add_u3(1, true, true, true);                          
                    add_u3(0, true, true, true);
                    add_cnot(1,0);
                }


                for (int control_qbit=1; control_qbit<qbit_num-1; control_qbit=control_qbit+2){
                    if (control_qbit+2<qbit_num){

                        for( int idx=0; idx<inner_blocks; idx++) {
                            add_u3(control_qbit+1, true, true, true);
                            add_u3(control_qbit+2, true, true, true); 
                        
                            add_cnot(control_qbit+2,control_qbit+1);
                        }

                    }

                    for( int idx=0; idx<inner_blocks; idx++) {
                        add_u3(control_qbit+1, true, true, true);  
                        add_u3(control_qbit, true, true, true);  

                        add_cnot(control_qbit+1,control_qbit);

                    }

                }
            }


            return;
        }
        case HEA_ZYZ:
        {

            release_gates();

            if ( qbit_num < 2 ) {
                std::string error("Variational_Quantum_Eigensolver_Base::generate_initial_circuit: number of qubits should be at least 2");
                throw error;
            }

            for (int layer_idx=0; layer_idx<layers ;layer_idx++){

                for( int idx=0; idx<inner_blocks; idx++) {
                    Gates_block* block_1 = new Gates_block( qbit_num );
                    Gates_block* block_2 = new Gates_block( qbit_num );
 
                    // single qubit gates in a ablock are then merged into a single gate kernel.
                    block_1->add_rz(1);
                    block_1->add_ry(1);
                    block_1->add_rz(1);     

                    add_gate( block_1 );                           

                    block_2->add_rz(0);
                    block_2->add_ry(0);
                    block_2->add_rz(0);         

                    add_gate( block_2 );           
                
                    add_cnot(1,0);
                }

                for (int control_qbit=1; control_qbit<qbit_num-1; control_qbit=control_qbit+2){
                    if (control_qbit+2<qbit_num){

                        for( int idx=0; idx<inner_blocks; idx++) {

                            Gates_block* block_1 = new Gates_block( qbit_num );
                            Gates_block* block_2 = new Gates_block( qbit_num );

                            block_1->add_rz(control_qbit+1);
                            block_1->add_ry(control_qbit+1);
                            block_1->add_rz(control_qbit+1); 
                            add_gate( block_1 );                                

                            block_2->add_rz(control_qbit+2);
                            block_2->add_ry(control_qbit+2);
                            block_2->add_rz(control_qbit+2);  
                            add_gate( block_2 );                              

                            add_cnot(control_qbit+2,control_qbit+1);
                        }

                    }

                    for( int idx=0; idx<inner_blocks; idx++) {

                        Gates_block* block_1 = new Gates_block( qbit_num );
                        Gates_block* block_2 = new Gates_block( qbit_num );

                        block_1->add_rz(control_qbit+1);
                        block_1->add_ry(control_qbit+1);
                        block_1->add_rz(control_qbit+1); 
                        add_gate( block_1 );                      

                        block_2->add_rz(control_qbit);
                        block_2->add_ry(control_qbit);
                        block_2->add_rz(control_qbit);     
                        add_gate( block_2 );                  

                        add_cnot(control_qbit+1,control_qbit);
                    }

                }
            }

            return;
        }        
        default:
            std::string error("Variational_Quantum_Eigensolver_Base::generate_initial_circuit: ansatz not implemented");
            throw error;
    }

}

void 
Generative_Quantum_Machine_Learning_Base::set_gate_structure( std::string filename ) {

    release_gates();
    Matrix_real optimized_parameters_mtx_tmp;
    Gates_block* gate_structure_tmp = import_gate_list_from_binary(optimized_parameters_mtx_tmp, filename);

    if ( gates.size() > 0 ) {
        gate_structure_tmp->combine( static_cast<Gates_block*>(this) );

        release_gates();
        combine( gate_structure_tmp );


        Matrix_real optimized_parameters_mtx_tmp2( 1, optimized_parameters_mtx_tmp.size() + optimized_parameters_mtx.size() );
        memcpy( optimized_parameters_mtx_tmp2.get_data(), optimized_parameters_mtx_tmp.get_data(), optimized_parameters_mtx_tmp.size()*sizeof(double) );
        memcpy( optimized_parameters_mtx_tmp2.get_data()+optimized_parameters_mtx_tmp.size(), optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size()*sizeof(double) );
        optimized_parameters_mtx = optimized_parameters_mtx_tmp2;
    }
    else {
        combine( gate_structure_tmp );
        optimized_parameters_mtx = optimized_parameters_mtx_tmp;
        gates_num gates_num = get_gate_nums();

        if ( gates_num.u3>0 )  std::cout << gates_num.u3 << " U3 gates," << std::endl;
        if ( gates_num.rx>0 )  std::cout << gates_num.rx << " RX gates," << std::endl;
        if ( gates_num.ry>0 )  std::cout << gates_num.ry << " RY gates," << std::endl;
        if ( gates_num.rz>0 )  std::cout << gates_num.rz << " RZ gates," << std::endl;
        if ( gates_num.cnot>0 )  std::cout << gates_num.cnot << " CNOT gates," << std::endl;
        if ( gates_num.cz>0 )  std::cout << gates_num.cz << " CZ gates," << std::endl;
        if ( gates_num.ch>0 )  std::cout << gates_num.ch << " CH gates," << std::endl;
        if ( gates_num.x>0 )  std::cout << gates_num.x << " X gates," << std::endl;
        if ( gates_num.sx>0 )  std::cout << gates_num.sx << " SX gates," << std::endl; 
        if ( gates_num.syc>0 )  std::cout << gates_num.syc << " Sycamore gates," << std::endl;   
        if ( gates_num.un>0 )  std::cout << gates_num.un << " UN gates," << std::endl;
        if ( gates_num.cry>0 )  std::cout << gates_num.cry << " CRY gates," << std::endl;  
        if ( gates_num.adap>0 )  std::cout << gates_num.adap << " Adaptive gates," << std::endl;    	
    }

}




/**
@brief Call to set the initial quantum state in the GQML iterations
@param initial_state_in A vector containing the amplitudes of the initial state.
*/
void Generative_Quantum_Machine_Learning_Base::set_initial_state( Matrix initial_state_in ) {



    // check the size of the input state
    if ( initial_state_in.size() != 1 << qbit_num ) {
        std::string error("Variational_Quantum_Eigensolver_Base::set_initial_state: teh number of elements in the input state does not match with the number of qubits.");
        throw error;   
    }

    initial_state = Matrix( 1 << qbit_num, 1 );
    memcpy( initial_state.get_data(), initial_state_in.get_data(), initial_state_in.size()*sizeof( QGD_Complex16 ) );

}




