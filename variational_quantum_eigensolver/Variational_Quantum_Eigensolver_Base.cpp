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
/*! \file Decomposition_Base.cpp
    \brief Class containing basic methods for the decomposition process.
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
    convergence_threshold = -1;
    
    
    iteration_threshold_of_randomization = 2500000;

    random_shift_count_max = 100;
    
    adaptive_eta = false;
    
	qbit_num = qbit_num_in;
	
	Zero_state = Matrix(Power_of_2(qbit_num),1);
	
    std::random_device rd;  
    
    gen = std::mt19937(rd());
    
    alg = AGENTS;
    
    cost_fnc = VQE;
}

Variational_Quantum_Eigensolver_Base::~Variational_Quantum_Eigensolver_Base(){}

void Variational_Quantum_Eigensolver_Base::Get_ground_state(){

	initialize_zero_state();
	int num_of_parameters =  optimized_parameters_mtx.cols;

    if (gates.size() == 0 ) {
            return;
    }
    Matrix_real parameters = optimized_parameters_mtx.copy();
    solve_layer_optimization_problem(num_of_parameters,parameters);
    double f0=optimization_problem(optimized_parameters_mtx);
    std::cout<<"Ground state found, energy: "<<f0<<std::endl;
    prepare_gates_to_export();
    return;
}

double Variational_Quantum_Eigensolver_Base::Expected_energy(Matrix State){
	Matrix tmp = mult(Hamiltonian, State);
	
	double Energy= 0.;
	for (int idx=0; idx<State.rows; idx++){
		Energy += State[idx].real*tmp[idx].real + State[idx].imag*tmp[idx].imag;
	} 
	tmp.release_data();
	return Energy;
}

double Variational_Quantum_Eigensolver_Base::optimization_problem_vqe(Matrix_real parameters, void* void_instance){
	double Energy=0.;
    Variational_Quantum_Eigensolver_Base* instance = reinterpret_cast<Variational_Quantum_Eigensolver_Base*>(void_instance);
    Gates_block* gate_structure_loc = static_cast<Gates_block*>(instance)->clone();
	Matrix State = instance->Zero_state.copy();
	gate_structure_loc->apply_to(parameters, State);
	Energy = instance->Expected_energy(State);
    State.release_data();
    delete gate_structure_loc;
	return Energy;
}

double Variational_Quantum_Eigensolver_Base::optimization_problem(Matrix_real& parameters)  {
	double Energy=0.;
	Matrix State = Zero_state.copy();
    Gates_block* gate_structure_loc = static_cast<Gates_block*>(this)->clone();
	gate_structure_loc->apply_to(parameters, State);
	Energy = Expected_energy(State);
    State.release_data();
    delete gate_structure_loc;
	return Energy;
}


void Variational_Quantum_Eigensolver_Base::optimization_problem_combined_vqe( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad ) {

    Variational_Quantum_Eigensolver_Base* instance = reinterpret_cast<Variational_Quantum_Eigensolver_Base*>(void_instance);
    // the number of free parameters
    int parameter_num_loc = parameters.size();

    Matrix_real cost_function_terms;

    // vector containing gradients of the transformed matrix
    std::vector<Matrix> State_deriv;

    tbb::parallel_invoke(
        [&]{
            *f0 = instance->optimization_problem_vqe(parameters, reinterpret_cast<void*>(instance)); 
        },
        [&]{
            Matrix State_loc = instance->Zero_state.copy();
            State_deriv = instance->apply_derivate_to( parameters, State_loc );
            State_loc.release_data();
    });

    tbb::parallel_for( tbb::blocked_range<int>(0,parameter_num_loc,2), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 

            double grad_comp;
            grad_comp = instance->Expected_energy(State_deriv[idx]);
            grad[idx] = grad_comp;

        }
    });
    return;
}

void Variational_Quantum_Eigensolver_Base::optimization_problem_combined( Matrix_real parameters, double* f0, Matrix_real grad )  {

    optimization_problem_combined_vqe( parameters, this, f0, grad );
    return;
}

void Variational_Quantum_Eigensolver_Base::optimization_problem_grad_vqe( Matrix_real parameters, void* void_instance, Matrix_real& grad ) {
    double f0;
    optimization_problem_combined_vqe(parameters, void_instance, &f0, grad);
    return;
}

BFGS_Powell Variational_Quantum_Eigensolver_Base::create_bfgs_problem(){
    return BFGS_Powell(optimization_problem_combined_vqe, this);
}

double Variational_Quantum_Eigensolver_Base::optimization_problem( double* parameters ) {

    // get the transformed matrix with the gates in the list
    Matrix_real parameters_mtx(parameters, 1, parameter_num );
    
    return optimization_problem( parameters_mtx );


}

void Variational_Quantum_Eigensolver_Base::initialize_zero_state( ) {
	Matrix tmp = Matrix((int)std::pow(2,qbit_num),1);
	tmp[0].real = 1.;
	tmp[0].imag = 0.;
	for (int idx=1; idx<(int)std::pow(2,qbit_num);idx++){
		tmp[idx].real = 0.;
		tmp[idx].imag = 0.;
	}
	Zero_state = tmp.copy();
	return;
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

void Variational_Quantum_Eigensolver_Base::set_custom_gate_structure( std::map<int, Gates_block*> gate_structure_in ) {


    for ( std::map<int,Gates_block*>::iterator it=gate_structure_in.begin(); it!= gate_structure_in.end(); it++ ) {
        int key = it->first;

        std::map<int,Gates_block*>::iterator key_it = gate_structure.find( key );

        if ( key_it != gate_structure.end() ) {
            gate_structure.erase( key_it );
        }

        gate_structure.insert( std::pair<int,Gates_block*>(key, it->second->clone()));

    }

}


