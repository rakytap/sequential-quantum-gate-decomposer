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
/*! \file Gates_block.cpp
    \brief Class responsible for grouping two-qubit (CNOT,CZ,CH) and one-qubit gates into layers
*/
#include "Variational_Quantum_Eigensolver_Base.h"

Variational_Quantum_Eigensolver_Base::Variational_Quantum_Eigensolver_Base() {

    // logical value describing whether the decomposition was finalized or not
    decomposition_finalized = false;


    // error of the unitarity of the final decomposition
    decomposition_error = 10;


    // The current minimum of the optimization problem
    current_minimum = 1e10;
    
    global_target_minimum = -100;

    // logical value describing whether the optimization problem was solved or not
    optimization_problem_solved = false;


    // The maximal allowed error of the optimization problem
    optimization_tolerance = -25;

    // The convergence threshold in the optimization process
    convergence_threshold = -1;
    
    alg = AGENTS;
    
    iteration_threshold_of_randomization = 2500000;

    random_shift_count_max = 100;
    
    adaptive_eta = false;
    
    cost_fnc = VQE;
    
    ansatz = HEA;
}

Variational_Quantum_Eigensolver_Base::Variational_Quantum_Eigensolver_Base( Matrix_sparse Hamiltonian_in, int qbit_num_in, std::map<std::string, Config_Element>& config_in) : N_Qubit_Decomposition_Base(Matrix(Power_of_2(qbit_num_in),1), qbit_num_in, false, config_in, RANDOM, accelerator_num) {

	Hamiltonian = Hamiltonian_in;
    // config maps
    config   = config_in;
    // logical value describing whether the decomposition was finalized or not
    decomposition_finalized = false;


    // error of the unitarity of the final decomposition
    decomposition_error = 10;


    // The current minimum of the optimization problem
    current_minimum = 1e10;


    // logical value describing whether the optimization problem was solved or not
    optimization_problem_solved = false;

    global_target_minimum = -100;
    // The maximal allowed error of the optimization problem
    optimization_tolerance = -25;

    // The convergence threshold in the optimization process
    convergence_threshold = -40;
    
    
    iteration_threshold_of_randomization = 2500000;

    random_shift_count_max = 100;
    
    adaptive_eta = false;
    
	qbit_num = qbit_num_in;
	
	Zero_state = Matrix(Power_of_2(qbit_num),1);
	
    std::random_device rd;  
    
    gen = std::mt19937(rd());
    
    alg = BAYES_OPT;
    
    cost_fnc = VQE;
    
    ansatz = HEA;
    
}

Variational_Quantum_Eigensolver_Base::~Variational_Quantum_Eigensolver_Base(){}

void Variational_Quantum_Eigensolver_Base::Get_ground_state(){

    initialize_zero_state();
    int num_of_parameters =  optimized_parameters_mtx.size();
    if (gates.size() == 0 ) {
            return;
    }
    Matrix_real solution_guess = optimized_parameters_mtx.copy();
    solve_layer_optimization_problem(num_of_parameters,solution_guess);
    double f0=optimization_problem(optimized_parameters_mtx);
    std::cout<<"Ground state found, energy: "<<current_minimum<<std::endl;
    prepare_gates_to_export();
    return;
}

double Variational_Quantum_Eigensolver_Base::Expected_energy(Matrix& State){
	Matrix tmp = mult(Hamiltonian, State);
	double Energy= 0.0;
	for (int idx=0; idx<State.rows; idx++){
	Energy += State[idx].real*tmp[idx].real + State[idx].imag*tmp[idx].imag;
	} 
	tmp.release_data();
	number_of_iters++;
	return Energy;
}


double Variational_Quantum_Eigensolver_Base::optimization_problem_non_static(Matrix_real parameters, void* void_instance){
	double Energy=0.;
    Variational_Quantum_Eigensolver_Base* instance = reinterpret_cast<Variational_Quantum_Eigensolver_Base*>(void_instance);
	instance->initialize_zero_state();
	Matrix State = instance->Zero_state.copy();
	instance->apply_to(parameters, State);
	Energy = instance->Expected_energy(State);

	return Energy;
}

double Variational_Quantum_Eigensolver_Base::optimization_problem(Matrix_real& parameters)  {
	double Energy;
	initialize_zero_state();
	Matrix State = Zero_state.copy();
	apply_to(parameters, State);
	//State.print_matrix();
	Energy = Expected_energy(State);
	number_of_iters++;
	return Energy;
}


void Variational_Quantum_Eigensolver_Base::optimization_problem_combined_non_static( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad ) {
    Variational_Quantum_Eigensolver_Base* instance = reinterpret_cast<Variational_Quantum_Eigensolver_Base*>(void_instance);
    // the number of free parameters
    int parameter_num_loc = parameters.size();

    Matrix_real cost_function_terms;

    // vector containing gradients of the transformed matrix
    //std::vector<Matrix> State_deriv;

    tbb::parallel_invoke(
        [&]{
            *f0 = instance->optimization_problem_non_static(parameters, reinterpret_cast<void*>(instance));
        },
        [&]{
	    instance->initialize_zero_state();
            Matrix State_loc = instance->Zero_state.copy();
            //State_deriv = instance->apply_derivate_to( parameters, State_loc );
            State_loc.release_data();
    });

    tbb::parallel_for( tbb::blocked_range<int>(0,parameter_num_loc,2), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 
            Matrix_real parameters_grad = parameters.copy();
            parameters_grad[idx] = parameters_grad[idx] - M_PI/2;
            double grad_comp_neg =  instance->optimization_problem_non_static(parameters_grad, reinterpret_cast<void*>(instance));
            parameters_grad[idx] = parameters_grad[idx] + M_PI;
            double grad_comp_pos =  instance->optimization_problem_non_static(parameters_grad, reinterpret_cast<void*>(instance));
            grad[idx] = 0.5*(grad_comp_pos-grad_comp_neg);

        }
    });
    return;
}

void Variational_Quantum_Eigensolver_Base::optimization_problem_combined( Matrix_real parameters, double* f0, Matrix_real grad )  {

    optimization_problem_combined_non_static( parameters, this, f0, grad );
    return;
}

void Variational_Quantum_Eigensolver_Base::optimization_problem_grad_vqe( Matrix_real parameters, void* void_instance, Matrix_real& grad ) {
    double f0;
    Variational_Quantum_Eigensolver_Base* instance = reinterpret_cast<Variational_Quantum_Eigensolver_Base*>(void_instance);
    instance->optimization_problem_combined_non_static(parameters, void_instance, &f0, grad);
    return;
}

double Variational_Quantum_Eigensolver_Base::optimization_problem( double* parameters ) {

    // get the transformed matrix with the gates in the list
    Matrix_real parameters_mtx(parameters, 1, parameter_num );
    
    return optimization_problem( parameters_mtx );


}

void Variational_Quantum_Eigensolver_Base::initialize_zero_state( ) {

	Zero_state[0].real = 1.0;
	Zero_state[0].imag = 0.0;
    memset(Zero_state.get_data()+2, 0.0, (Zero_state.size()*2-2)*sizeof(double) );      
	return;
}

void Variational_Quantum_Eigensolver_Base::set_ansatz(ansatz_type ansatz_in){

    ansatz = ansatz_in;
    
    return;
}

void Variational_Quantum_Eigensolver_Base::generate_initial_circuit( int layers, int blocks, int rot_layers ) {


    switch (ansatz){
    
        case HEA:
        {
            Gates_block* gate_structure_tmp = new Gates_block(qbit_num);
            for (int layer_idx=0; layer_idx<layers ;layer_idx++){
                for (int ridx=0; ridx<rot_layers; ridx++){
                        gate_structure_tmp->add_u3(1, true, true, true);
                        gate_structure_tmp->add_u3(0, true, true, true);
                }
                for (int bidx=0; bidx<blocks; bidx++){
                    gate_structure_tmp->add_adaptive(1,0);
                }
                if (layer_idx==layers-1){
                    gate_structure_tmp->add_u3(0, true, true, true);
                }
                for (int control_qbit=1; control_qbit<qbit_num-1; control_qbit=control_qbit+2){
                    if (control_qbit+2<qbit_num){
                        for (int ridx=0; ridx<rot_layers; ridx++){
                            gate_structure_tmp->add_u3(control_qbit+1, true, true, true);
                            gate_structure_tmp->add_u3(control_qbit+2, true, true, true);
                        }
                        for (int bidx=0; bidx<blocks; bidx++){
                            gate_structure_tmp->add_adaptive(control_qbit+2,control_qbit+1);
                        }
                    }
                    for (int ridx=0; ridx<rot_layers; ridx++){
                        gate_structure_tmp->add_u3(control_qbit+1, true, true, true);
                        gate_structure_tmp->add_u3(control_qbit, true, true, true);
                    }
                    for (int bidx=0; bidx<blocks; bidx++){
                        gate_structure_tmp->add_adaptive(control_qbit+1,control_qbit);
                    }
                    if (layer_idx==layers-1){
                        gate_structure_tmp->add_u3(control_qbit, true, true, true);
                        gate_structure_tmp->add_u3(control_qbit+1, true, true, true);
                    }
                }
            }
            if (qbit_num%2==0){
                gate_structure_tmp->add_u3(qbit_num-1, true, true, true);
            }
            gate_structure_tmp->combine( static_cast<Gates_block*>(this) );
            release_gates();
            combine( gate_structure_tmp );
            Matrix_real optimized_parameters_mtx_tmp( 1, get_parameter_num() );
            if (alg == AGENTS){
	        memset( optimized_parameters_mtx_tmp.get_data(), 0.0, optimized_parameters_mtx_tmp.size()*sizeof(double) ); 
            }
            else{
            std::uniform_real_distribution<> distrib_real(0.0, 2*M_PI); 
	        for(int idx = 0; idx < get_parameter_num(); idx++) {
                    optimized_parameters_mtx_tmp[idx] = distrib_real(gen);
                }
            }
            optimized_parameters_mtx = optimized_parameters_mtx_tmp;
            //max_fusion = (qbit_num>4)? 4:qbit_num-1;
            //fragment_circuit();
            return;
        }
        default:
            std::string error("Variational_Quantum_Eigensolver_Base::generate_initial_circuit: ansatz not implemented");
            throw error;
    }

}

void 
Variational_Quantum_Eigensolver_Base::set_adaptive_gate_structure( std::string filename ) {

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




