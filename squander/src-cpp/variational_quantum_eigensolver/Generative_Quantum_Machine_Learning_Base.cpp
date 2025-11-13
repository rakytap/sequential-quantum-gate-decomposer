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
#include <algorithm>
#include <random>

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
@param sample_indices_in The input data indices
@param P_star_in The distribution to approximate
@param sigma_in Parameter of the gaussian kernels
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param use_lookup_table_in Use lookup table for the gaussian kernel
@param config_in A map that can be used to set hyperparameters during the process
@return An instance of the class
*/
Generative_Quantum_Machine_Learning_Base::Generative_Quantum_Machine_Learning_Base(std::vector<int> sample_indices_in, Matrix_real P_star_in, Matrix_real sigma_in, int qbit_num_in, bool use_lookup_table_in, std::vector<std::vector<int>> cliques_in, bool use_exact_in, std::map<std::string, Config_Element>& config_in, int accelerator_num) : Optimization_Interface(Matrix(Power_of_2(qbit_num_in),1), qbit_num_in, false, config_in, RANDOM, accelerator_num) {

	sample_indices = sample_indices_in;

    sample_size = sample_indices.size();

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
	
    alg = BAYES_OPT;
    
    cost_fnc = GQML;
    
    ansatz = HEA;
    
    ev_P_star_P_star = -1.0;

    sigma = sigma_in;

    use_lookup = use_lookup_table_in;

    use_exact = use_exact_in;

    if (use_lookup) {
        fill_lookup_table();
    }

    if (use_exact) {
        MMD_of_the_distributions = &Generative_Quantum_Machine_Learning_Base::MMD_of_the_distributions_exact;
        ev_P_star_P_star = expectation_value_P_star_P_star_exact();
    }
    else {
        MMD_of_the_distributions = &Generative_Quantum_Machine_Learning_Base::MMD_of_the_distributions_approx;
        ev_P_star_P_star = expectation_value_P_star_P_star_approx();
    }
    

    cliques = cliques_in;
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


    // start the GQML process
    Matrix_real solution_guess = optimized_parameters_mtx.copy();
    solve_layer_optimization_problem(num_of_parameters, solution_guess);

    if (ev_P_star_P_star == -1) {
        ev_P_star_P_star = expectation_value_P_star_P_star_exact();
    }
    if (use_lookup && gaussian_lookup_table.size() == 0) {
        fill_lookup_table();
    }

    return;
}

/**
@brief Call to evaluate the value of one gaussian kernel function
@param x The first bitstring input
@param y The second bitstring input
@param sigma The parameters of the kernel
@return The calculated value of the kernel function
*/
double Generative_Quantum_Machine_Learning_Base::Gaussian_kernel(int x, int y) {
    // The norm stores the distance between the two data points (the more qbit they differ in the bigger it is)
    double result=0.0;
    double exponent;

    for (int i=0; i<sigma.size(); i++) {
        exponent = -(x - y)*((x - y)/sigma[i])*0.5;
        result += exp(exponent);
    }
    result /= sigma.size();
    return result;
}

/**
@brief Call to calculate and save the values of the gaussian kernel needed for traing
*/
void Generative_Quantum_Machine_Learning_Base::fill_lookup_table() {
    gaussian_lookup_table = std::vector<double>(1<<qbit_num);
    for (int idx1=0; idx1 < 1<<qbit_num; idx1++) {
        gaussian_lookup_table[idx1] = Gaussian_kernel(idx1, 0);
    }
}

/**
@brief Call to evaluate the approximated expectation value of the square of the distribution
@return The approximated value of the expectation value of the square of the distribution
*/
double Generative_Quantum_Machine_Learning_Base::expectation_value_P_star_P_star_approx() {
    double ev=0.0;
    tbb::combinable<double> priv_partial_ev{[](){return 0.0;}};
    tbb::parallel_for( tbb::blocked_range<int>(0, sample_size, 1024), [&](tbb::blocked_range<int> r) {
        double& ev_local = priv_partial_ev.local();
        for (int idx1=r.begin(); idx1<r.end(); idx1++) {
            for (int idx2=0; idx2<sample_size; idx2++) {
                if (idx1 != idx2) {
                    ev_local += Gaussian_kernel(idx1, idx2);
                }
            }
        }
    });
    priv_partial_ev.combine_each([&ev](double a) {
        ev += a;
    });
    ev /= sample_size*(sample_size-1);
    return ev;
}

/**
@brief Call to evaluate the expectation value of the square of the distribution
@return The calculated value of the expectation value of the square of the distribution
*/
double Generative_Quantum_Machine_Learning_Base::expectation_value_P_star_P_star_exact() {
    double ev=0.0;
    tbb::combinable<double> priv_partial_ev{[](){return 0.0;}};
    tbb::parallel_for( tbb::blocked_range<int>(0, 1<<qbit_num, 1024), [&](tbb::blocked_range<int> r) {
        double& ev_local = priv_partial_ev.local();
        for (int idx1=r.begin(); idx1<r.end(); idx1++) {
            for (int idx2=0; idx2<1<<qbit_num; idx2++) {
                ev_local += P_star[idx1]*P_star[idx2]*Gaussian_kernel(idx1, idx2);
            }
        }
    });
    priv_partial_ev.combine_each([&ev](double a) {
        ev += a;
    });
    return ev;
}

/**
@brief Call to evaluate the total variational distance of the given distribution and the one created by our circuit
@param State_right The state on the right for which the expectation value is evaluated. It is a column vector.
@return The calculated total variational distance of the distributions
*/
double Generative_Quantum_Machine_Learning_Base::TV_of_the_distributions(Matrix& State_right) {
    std::vector<double> P_theta(1<<qbit_num);

    for (size_t x_idx=0; x_idx<State_right.size(); x_idx++){
        P_theta[x_idx] = State_right[x_idx].real*State_right[x_idx].real +State_right[x_idx].imag*State_right[x_idx].imag;
    }

    double TV = 0.0;
    for (int i=0; i<P_theta.size(); i++) {
        TV += abs(P_theta[i]-P_star[i]);
    }
    return TV*0.5;
}


/**
@brief Call to evaluate the maximum mean discrepancy of the given distribution and the one created by our circuit
@param State_right The state on the right for which the expectation value is evaluated. It is a column vector.
@return The calculated mmd
*/
double Generative_Quantum_Machine_Learning_Base::MMD_of_the_distributions_exact( Matrix& State_right ) {
    // If the ev of the P_star hasnt been evaluated we need to evaluate it
    if (ev_P_star_P_star < 0 ) {
        ev_P_star_P_star = expectation_value_P_star_P_star_exact();
    }

    // We calculate the distribution created by our circuit at the given traing data points "we sample our distribution"
    std::vector<double> P_theta(1<<qbit_num);
    tbb::parallel_for( tbb::blocked_range<int>(0, 1<<qbit_num, 1024), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); idx++) {
            P_theta[idx] = State_right[idx].real*State_right[idx].real + State_right[idx].imag*State_right[idx].imag;
        }
    });

    // Calculate the expectation values 
    double ev_P_theta_P_theta   = 0.0;
    double ev_P_theta_P_star    = 0.0;
    int N=1<<qbit_num;
    tbb::combinable<double> priv_partial_ev_P_theta_P_theta{[](){return 0.0;}};
    tbb::combinable<double> priv_partial_ev_P_theta_P_star{[](){return 0.0;}};
    tbb::parallel_for( tbb::blocked_range<int>(0, N-1, 1024), [&](tbb::blocked_range<int> r) {
        double& ev_P_theta_P_theta_local = priv_partial_ev_P_theta_P_theta.local();
        double& ev_P_theta_P_star_local = priv_partial_ev_P_theta_P_star.local();
        for (int idx1=r.begin(); idx1<r.end(); idx1++) {
            double tmp_theta_theta = 0.0;
            double tmp_theta_star = 0.0;
            for (int idx2=0; idx2<idx1-1; idx2++) {
                tmp_theta_theta += 2*P_theta[idx2]*P_theta[N-idx1+idx2-1];
                tmp_theta_star += P_theta[idx2]*P_star[N-idx1+idx2-1]+P_theta[N-idx1+idx2-1]*P_star[idx2];
            }
            ev_P_theta_P_theta_local += tmp_theta_theta*gaussian_lookup_table[N-idx1-1];
            ev_P_theta_P_star_local += tmp_theta_star*gaussian_lookup_table[N-idx1-1];
        }
    });
    priv_partial_ev_P_theta_P_theta.combine_each([&ev_P_theta_P_theta](double a) {
        ev_P_theta_P_theta += a;
    });
    priv_partial_ev_P_theta_P_star.combine_each([&ev_P_theta_P_star](double a) {
        ev_P_theta_P_star += a;
    });
    for (int idx=0; idx<N; idx++) {
        ev_P_theta_P_theta += P_theta[idx]*P_theta[idx]*gaussian_lookup_table[0];
        ev_P_theta_P_star += P_theta[idx]*P_star[idx]*gaussian_lookup_table[0];
    }


    {
        tbb::spin_mutex::scoped_lock my_lock{my_mutex};

        number_of_iters++;
        
    }

    double result = ev_P_theta_P_theta + ev_P_star_P_star - 2*ev_P_theta_P_star;
    return result;
}


/**
@brief Call to evaluate the approximated maximum mean discrepancy of the given distribution and the one created by our circuit
@param State_right The state on the right for which the expectation value is evaluated. It is a column vector.
@return The calculated mmd
*/
double Generative_Quantum_Machine_Learning_Base::MMD_of_the_distributions_approx( Matrix& State_right ) {

    // If the ev of the P_star hasnt been evaluated we need to evaluate it
    if (ev_P_star_P_star < 0 ) {
        ev_P_star_P_star = expectation_value_P_star_P_star_approx();
    }

    // We calculate the distribution created by our circuit at the given traing data points "we sample our distribution"
    std::vector<double> P_theta(1<<qbit_num);
    tbb::parallel_for( tbb::blocked_range<int>(0, 1<<qbit_num, 1024), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); idx++) {
            P_theta[idx] = State_right[idx].real*State_right[idx].real + State_right[idx].imag*State_right[idx].imag;
        }
    });

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(P_theta.begin(), P_theta.end());

    std::vector<int> theta_sample_indices(sample_size);
    tbb::parallel_for( tbb::blocked_range<int>(0, sample_size, 1024), [&](tbb::blocked_range<int> r) {
        for (int sample_idx=r.begin(); sample_idx < r.end(); sample_idx++) {
            theta_sample_indices[sample_idx] = dist(gen);
        }
    });


    // Calculate the expectation values 
    double ev_P_theta_P_theta   = 0.0;
    double ev_P_theta_P_star    = 0.0;
    tbb::combinable<double> priv_partial_ev_P_theta_P_theta{[](){return 0.0;}};
    tbb::combinable<double> priv_partial_ev_P_theta_P_star{[](){return 0.0;}};
    tbb::parallel_for( tbb::blocked_range2d<int>(0, sample_size, 0, sample_size), [&](tbb::blocked_range2d<int> r) {
        double& ev_P_theta_P_theta_local = priv_partial_ev_P_theta_P_theta.local();
        double& ev_P_theta_P_star_local = priv_partial_ev_P_theta_P_star.local();
        for (int idx1=r.rows().begin(); idx1<r.rows().end(); idx1++) {
            for (int idx2=r.cols().begin(); idx2<r.cols().end(); idx2++) {
                if (use_lookup) {
                    if (idx1 != idx2) {
                        ev_P_theta_P_theta_local += gaussian_lookup_table[abs(theta_sample_indices[idx1]-theta_sample_indices[idx2])];
                    }
                    ev_P_theta_P_star_local += gaussian_lookup_table[abs(theta_sample_indices[idx1]-sample_indices[idx2])];
                }
                else {
                    if (idx1 != idx2) {
                        ev_P_theta_P_theta_local += Gaussian_kernel(theta_sample_indices[idx1],theta_sample_indices[idx2]);
                    }
                    ev_P_theta_P_star_local += Gaussian_kernel(theta_sample_indices[idx1], sample_indices[idx2]);
                }
            }
        }
    });
    priv_partial_ev_P_theta_P_theta.combine_each([&ev_P_theta_P_theta](double a) {
        ev_P_theta_P_theta += a;
    });
    priv_partial_ev_P_theta_P_star.combine_each([&ev_P_theta_P_star](double a) {
        ev_P_theta_P_star += a;
    });

    ev_P_theta_P_theta /= sample_size*(sample_size-1);
    ev_P_theta_P_star /= sample_size*sample_size;

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
    MMD = (instance->*MMD_of_the_distributions)(State);

    return MMD;
}


#ifdef __GROQ__
/**
@brief The optimization problem of the final optimization implemented to be run on Groq hardware
@param parameters An array of the free parameters to be optimized.
@param chosen_device Indicate the device on which the state vector emulation is performed
@return Returns with the cost function.
*/
double Generative_Quantum_Machine_Learning_Base::optimization_problem_Groq(Matrix_real& parameters, int chosen_device)  {



    Matrix State;


    //tbb::tick_count t0_DFE = tbb::tick_count::now();
    std::vector<int> target_qbits;
    std::vector<int> control_qbits;
    std::vector<Matrix> u3_qbit;
    extract_gate_kernels_target_and_control_qubits(u3_qbit, target_qbits, control_qbits, parameters);
        

    // initialize the initial state on the chip if it was not given
    if ( initial_state.size() == 0 ) {
        
        Matrix State_zero(0,0);  
        apply_to_groq_sv(accelerator_num, chosen_device, qbit_num, u3_qbit, target_qbits, control_qbits, State_zero, id); 
            
        State = State_zero;
            
    }
    else {
	
        State = initial_state.copy();
        // apply state transformation via the Groq chip
        apply_to_groq_sv(accelerator_num, chosen_device, qbit_num, u3_qbit, target_qbits, control_qbits, State, id);

    }
        
        
/*        
    //////////////////////////////
    Matrix State_copy = State.copy();


    Decomposition_Base::apply_to(parameters, State_copy );


    double diff = 0.0;
    for( int64_t idx=0; idx<State_copy.size(); idx++ ) {

        QGD_Complex16 element = State[idx];
        QGD_Complex16 element_copy = State_copy[idx];
        QGD_Complex16 element_diff;
        element_diff.real = element.real - element_copy.real;
        element_diff.imag = element.imag - element_copy.imag;
 
        double diff_increment = element_diff.real*element_diff.real + element_diff.imag*element_diff.imag;
        diff = diff + diff_increment;
    }
       
    std::cout << "Generative_Quantum_Machine_Learning_Base::apply_to checking diff: " << diff << std::endl;


    if ( diff > 1e-4 ) {
        std::string error("Groq and CPU results do not match");
        throw(error);
    }

    //////////////////////////////
*/


	
    double MMD = (this->*MMD_of_the_distributions)(State);
	
    return MMD;

}

#endif


/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double Generative_Quantum_Machine_Learning_Base::optimization_problem(Matrix_real& parameters)  {

    Matrix State;

#ifdef __GROQ__
    if ( accelerator_num > 0 ) {
    
        
        return optimization_problem_Groq( parameters, 0 );
            
    }
    else {
#endif
        // initialize the initial state if it was not given
        if ( initial_state.size() == 0 ) {
            initialize_zero_state();
        }
	
        State = initial_state.copy();
        apply_to(parameters, State);

#ifdef __GROQ__
    }
#endif
	
    double MMD = (this->*MMD_of_the_distributions)(State);
	
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
            *f0 = (instance->*MMD_of_the_distributions)(State);
            
        },
        [&]{
            Matrix State_loc = instance->initial_state.copy();

            State_deriv = instance->apply_derivate_to( parameters, State_loc, parallel );
            State_loc.release_data();
    });

    tbb::parallel_for( tbb::blocked_range<int>(0,parameter_num_loc,2), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 
            grad[idx] = 2*(instance->*MMD_of_the_distributions)(State_deriv[idx]);
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
                    add_u3(1);                          
                    add_u3(0);
                    add_cnot(1,0);
                }


                for (int control_qbit=1; control_qbit<qbit_num-1; control_qbit=control_qbit+2){
                    if (control_qbit+2<qbit_num){

                        for( int idx=0; idx<inner_blocks; idx++) {
                            add_u3(control_qbit+1);
                            add_u3(control_qbit+2); 
                        
                            add_cnot(control_qbit+2,control_qbit+1);
                        }

                    }

                    for( int idx=0; idx<inner_blocks; idx++) {
                        add_u3(control_qbit+1);  
                        add_u3(control_qbit);  

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

        case QCMRF:
        {
            int num_cliques = cliques.size();
            if (cliques.size() == 0) {
                std::string error("Variational_Quantum_Eigensolver_Base::generate_initial_circuit: input the cliques for using QCMRF ansatz");
                throw error;
            }
            release_gates();
            for (int qbit_idx=0; qbit_idx < qbit_num; qbit_idx++) {
                add_h(qbit_idx);
            }

            std::vector<std::vector<int>> all_subsets;
            for (int clique_idx = 0; clique_idx < num_cliques; clique_idx++) {
                int clique_size = cliques[clique_idx].size();
                std::vector<int> subset;
                generate_clique_circuit(0, cliques[clique_idx], all_subsets, subset);
            }
            for (int qbit_idx=0; qbit_idx < qbit_num; qbit_idx++) {
                add_u3(qbit_idx);
            }
            return;
        }
        default:
            std::string error("Generative_Quantum_Machine_Learning_Base::generate_initial_circuit: ansatz not implemented");
            throw error;
    }

}

/**
@brief Call to generate a MultiRZ gate
@param qbits The qbits the gate operates on. The depth of the generated circuit is 2*number of qbits
*/
void Generative_Quantum_Machine_Learning_Base::MultyRZ(std::vector<int>& qbits) {
    for (int idx=0; idx<qbits.size()-1; idx++) {
        add_cnot(qbits[idx+1], qbits[idx]);
    }
    add_rz(qbits[qbits.size()-1]);
    for (int idx=qbits.size()-1; idx>0; idx--) {
        add_cnot(qbits[idx], qbits[idx-1]);
    }
}

/**
@brief Call to generate the circuit ansatz for the given clique
@param qbits The qbits in the clique.
@param res The qbits for previously generated gates to avoid duplication
@param subset Temporary variable for storing subsets.
*/
void Generative_Quantum_Machine_Learning_Base::generate_clique_circuit(int i, std::vector<int>& arr, std::vector<std::vector<int>>& res, std::vector<int>& subset) {
    if (i == arr.size()) {
        if (subset.size() != 0) {
            if (std::find(res.begin(), res.end(), subset) == res.end()) {
                res.push_back(subset);
                MultyRZ(subset);
            }
        }
        return;
    }

    subset.push_back(arr[i]);
    generate_clique_circuit(i+1, arr, res, subset);

    subset.pop_back();
    generate_clique_circuit(i+1, arr, res, subset);
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
            
        std::stringstream sstream;
        // get the number of gates used in the decomposition
        std::map<std::string, int>&& gate_nums = get_gate_nums();
    	
        for( auto it=gate_nums.begin(); it != gate_nums.end(); it++ ) {
            sstream << it->second << " " << it->first << " gates" << std::endl;
        } 
        print(sstream, 1);	
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




