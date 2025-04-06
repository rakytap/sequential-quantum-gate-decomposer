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
    \brief Class responsible for grouping gates into subcircuits. (Subcircuits can be nested)
*/

#include "CZ.h"
#include "CH.h"
#include "CNOT.h"
#include "U3.h"
#include "RX.h"
#include "RY.h"
#include "CRY.h"
#include "RZ.h"
#include "H.h"
#include "X.h"
#include "Y.h"
#include "Z.h"
#include "SX.h"
#include "SYC.h"
#include "UN.h"
#include "ON.h"
#include "Adaptive.h"
#include "CZ_NU.h"
#include "Composite.h"
#include "Gates_block.h"

#include "custom_kernel_1qubit_gate.h"


#include "apply_large_kernel_to_input.h"

#ifdef USE_AVX 
#include "apply_large_kernel_to_input_AVX.h"
#endif


//static tbb::spin_mutex my_mutex;
/**
@brief Default constructor of the class.
*/
Gates_block::Gates_block() : Gate() {

    // A string describing the type of the operation
    type = BLOCK_OPERATION;
    // number of operation layers
    layer_num = 0;
}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
*/
Gates_block::Gates_block(int qbit_num_in) : Gate(qbit_num_in) {

    // A string describing the type of the operation
    type = BLOCK_OPERATION;
    // number of operation layers
    layer_num = 0;
    
    fragmented = false;
    fragmentation_type = -1;
    max_fusion = -1;
    
}


/**
@brief Destructor of the class.
*/
Gates_block::~Gates_block() {

    release_gates();

}

/**
@brief Call to release the stored gates
*/
void
Gates_block::release_gates() {

    //free the alloctaed memory of the stored gates
    for(std::vector<Gate*>::iterator it = gates.begin(); it != gates.end(); ++it) {

        Gate* gate = *it;
        delete gate;
        
    }
    
    gates.clear();
    layer_num = 0;
    parameter_num = 0;

}


/**
@brief Call to release one gate in the list
*/
void
Gates_block::release_gate( int idx) {

    if ( idx>= (int)gates.size() ) return;

    // fist decrese the number of parameters
    Gate* gate = gates[idx];
    parameter_num -= gate->get_parameter_num();

    gates.erase( gates.begin() + idx );
    
    // TODO: develop a more efficient method for large circuits. Now it is updating the whole circuit
    reset_parameter_start_indices();
    reset_dependency_graph();


}

/**
@brief Call to retrieve the operation matrix (Which is the product of all the operation matrices stored in the operation block)
@param parameters An array pointing to the parameters of the gates
@return Returns with the operation matrix
*/
Matrix
Gates_block::get_matrix( Matrix_real& parameters ) {

    return get_matrix( parameters, false );


}



/**
@brief Call to retrieve the operation matrix (Which is the product of all the operation matrices stored in the operation block)
@param parameters An array pointing to the parameters of the gates
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with the operation matrix
*/
Matrix
Gates_block::get_matrix( Matrix_real& parameters, int parallel ) {

    //The stringstream input to store the output messages.
    std::stringstream sstream;

    // create matrix representation of the gate operations
    Matrix block_mtx = create_identity(matrix_size);

    apply_to(parameters, block_mtx, parallel);

#ifdef DEBUG
    if (block_mtx.isnan()) {
        std::stringstream sstream;
	sstream << "Gates_block::get_matrix: block_mtx contains NaN." << std::endl;
        print(sstream, 0);		       
    }
#endif

    return block_mtx;


}



/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
Gates_block::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& inputs, int parallel ) {

    int work_batch = 1;
    if ( parallel == 0 ) {
        work_batch = inputs.size();
    }
    else {
        work_batch = 1;
    }


    tbb::parallel_for( tbb::blocked_range<int>(0,inputs.size(),work_batch), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 

            Matrix* input = &inputs[idx];

            apply_to( parameters_mtx, *input, parallel );

        }

    });


}


/**
@brief Call to apply the gate on the input array/matrix Gates_block*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gates_block::apply_to( Matrix_real& parameters_mtx_in, Matrix& input, int parallel ) {


    std::vector<int> involved_qubits = get_involved_qubits();
       
    // TODO: GATE fusion has not been adopted to reversed parameter ordering!!!!!!!!!!!!!!!!!!!
    if(max_fusion !=-1 && ((qbit_num>max_fusion && input.cols == 1) && involved_qubits.size()>1)){


        std::string error("Gates_block::apply_to: GATE fusion has not been adopted to reversed parameter ordering!!!!!!!!!!!!!!!!!!!");
        throw error; 
        
        double* parameters = parameters_mtx_in.get_data();     

        if (fragmented==false){
            fragment_circuit();
        };  

        int outer_idx = gates.size()-1;
        for (int block_idx=0; block_idx<involved_qbits.size(); block_idx++){

            if (block_type[block_idx] != 1){

                Gates_block gates_block_mini = Gates_block(block_type[block_idx]);
                std::vector<int> qbits = involved_qbits[block_idx];
#ifdef _WIN32
				int* indices = (int*)_malloca(qbit_num*sizeof(int));
#else
                int indices[qbit_num];
#endif

                for (int jdx=0; jdx<(int)qbits.size(); jdx++){
                    indices[qbits[jdx]]=jdx;
                }

                for (int idx=outer_idx; idx>=block_end[block_idx]; idx--){

                    Gate* gate = gates[idx]->clone();
                    int trgt_qbit = gate->get_target_qbit();
                    int ctrl_qbit = gate->get_control_qbit();
                    int target_qubit_new = indices[trgt_qbit];
                    gate->set_target_qbit(target_qubit_new);

                    int control_qubit_new = (ctrl_qbit==-1) ? -1:indices[ctrl_qbit];
                    gate->set_control_qbit(control_qubit_new);
                    gates_block_mini.add_gate(gate);
                }

                Matrix Umtx_mini = create_identity(Power_of_2(block_type[block_idx]));
                parameters       = parameters - gates_block_mini.get_parameter_num();

                Matrix_real parameters_mtx(parameters, 1, gates_block_mini.get_parameter_num());
                gates_block_mini.apply_to(parameters_mtx, Umtx_mini);

                outer_idx        = block_end[block_idx]-1;

#ifdef USE_AVX
                apply_large_kernel_to_state_vector_input_AVX(Umtx_mini, input, qbits, input.size() );
#else
                apply_large_kernel_to_state_vector_input(Umtx_mini, input, qbits, input.size() );
#endif                
       


            }
            else{

                Gates_block gates_block_mini = Gates_block(qbit_num);
                for (int idx=outer_idx;idx>=0;idx--){
                    Gate* gate = gates[idx]->clone();
                    gates_block_mini.add_gate(gate);
                }
                parameters = parameters - gates_block_mini.get_parameter_num();
                Matrix_real parameters_mtx(parameters, 1, gates_block_mini.get_parameter_num());
                gates_block_mini.apply_to(parameters_mtx, input);
            }
        }
    }
    else if  ( involved_qubits.size() == 1 && gates.size() > 1 && qbit_num > 1 ) {
    // merge successive single qubit gates
    
                Gates_block gates_block_mini = Gates_block(1);
  
                for (int idx=0; idx<gates.size(); idx++){

                    Gate* gate = gates[idx]->clone();
                    gate->set_target_qbit(0);
                    gate->set_qbit_num(1);

                    gates_block_mini.add_gate(gate);
                }
                    // TODO check gates block
                Matrix Umtx_mini = create_identity(2);

                Matrix_real parameters_mtx_loc(parameters_mtx_in.get_data() + gates[0]->get_parameter_start_idx(), 1, gates_block_mini.get_parameter_num());                
                gates_block_mini.apply_to(parameters_mtx_loc, Umtx_mini);

                custom_kernel_1qubit_gate merged_gate( qbit_num, involved_qubits[0], Umtx_mini );
                merged_gate.apply_to( input );
    }
    else {
    
        // No gate fusion
        for( int idx=0; idx<gates.size(); idx++) {

            Gate* operation = gates[idx];
            
            Matrix_real parameters_mtx_loc(parameters_mtx_in.get_data() + operation->get_parameter_start_idx(), 1, operation->get_parameter_num());
            
            if  ( parameters_mtx_loc.size() == 0 && operation->get_type() != BLOCK_OPERATION ) {
                operation->apply_to(input, parallel);            
            }
            else {
                operation->apply_to( parameters_mtx_loc, input, parallel );
            }
#ifdef DEBUG
        if (input.isnan()) {
            std::stringstream sstream;
	    sstream << "Gates_block::apply_to: transformed matrix contains NaN." << std::endl;
            print(sstream, 0);	
        }
#endif


       }

   }

}



//////// experimental attributes to partition the circuits into subsegments. Advantageous in simulation of larger circuits ///////////űű

/**
@brief Call to check whether the given qubit is involved in the sub-circuit or not.
@param involved_qubits A list of qubits
@param new_qbit The qubit to be checked
@param num_of_qbits The number of qubits.
*/
bool is_qbit_present(std::vector<int> involved_qubits, int new_qbit, int num_of_qbits){

    bool contained=false;
    
    for (int idx=0; idx<num_of_qbits; idx++) {
    
        if(involved_qubits[idx] == new_qbit) {
            contained=true;
        }
    }
    
    return contained;
    
}



    
    
void Gates_block::fragment_circuit(){

    std::vector<int> qbits;
    int num_of_qbits=0;
    int max_fusion_temp = (fragmentation_type==-1) ? max_fusion:fragmentation_type;

    for (int idx = gates.size()-1; idx>=0; idx--){      
      
        Gate* gate = gates[idx];
        int target_new = gate -> get_target_qbit();
        int control_new = gate->get_control_qbit();

        if (num_of_qbits == 0) {
            qbits.push_back(target_new);
            num_of_qbits++;

        }

        bool target_contained  = is_qbit_present(qbits,target_new,num_of_qbits);
        bool control_contained = (control_new==-1) ? true : is_qbit_present(qbits, control_new, num_of_qbits);

        if (num_of_qbits == max_fusion_temp && (target_contained == false || control_contained == false)){
            int vidx = 1;

            while(vidx<num_of_qbits){
                int jdx=vidx;

                while(jdx>0 && qbits[jdx-1]>qbits[jdx]){
                    int qbit_temp = qbits[jdx];
                    qbits[jdx]    = qbits[jdx-1];
                    qbits[jdx-1]  =  qbit_temp;
                    jdx--;
                }

                vidx++;
            }

            involved_qbits.push_back(qbits);
            block_end.push_back(idx+1);
            block_type.push_back(num_of_qbits);
            max_fusion_temp = max_fusion;
            idx++;    
            qbits=std::vector<int>{};
            num_of_qbits=0;
            continue;
        }

        if (num_of_qbits<max_fusion_temp && target_contained==false){
            qbits.push_back(target_new);
            num_of_qbits++;
        }

        if (num_of_qbits<max_fusion_temp && control_contained==false){
            qbits.push_back(control_new);
            num_of_qbits++;
        }


    }

    if (num_of_qbits == 1){
        involved_qbits.push_back(qbits);
        block_type.push_back(1);
    }

    else{
        int vidx = 1;

        while(vidx<num_of_qbits){
            int jdx=vidx;

            while(jdx>0 && qbits[jdx-1]>qbits[jdx]){
                int qbit_temp = qbits[jdx];
                qbits[jdx] = qbits[jdx-1];
                qbits[jdx-1] =  qbit_temp;
                jdx--;
            }

            vidx++;
        }

        involved_qbits.push_back(qbits);
        block_type.push_back(num_of_qbits);
        block_end.push_back(0);
    }

    block_end.push_back(0);
    fragmented = true;

}



void Gates_block::get_parameter_max(Matrix_real &range_max) {
    int parameter_idx = 0;
	double *data = range_max.get_data();
        for(int op_idx = 0; op_idx<gates.size(); op_idx++) {

            Gate* gate = gates[op_idx];
            switch (gate->get_type()) {
            case U3_OPERATION: {
                U3* u3_gate = static_cast<U3*>(gate);

                if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_theta_parameter()) {
		            data[parameter_idx] = 4 * M_PI;
                    parameter_idx = parameter_idx + 1;

                }
                else if ((u3_gate->get_parameter_num() == 1) && (u3_gate->is_phi_parameter() || u3_gate->is_lambda_parameter())) {
                    data[parameter_idx] = 2 * M_PI;
                    parameter_idx = parameter_idx + 1;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_theta_parameter() && (u3_gate->is_phi_parameter() || u3_gate->is_lambda_parameter())) {
                    data[parameter_idx] = 4 * M_PI;
                    data[parameter_idx+1] = 2 * M_PI;
                    parameter_idx = parameter_idx + 2;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_phi_parameter() && u3_gate->is_lambda_parameter() ) {
                    data[parameter_idx] = 2 * M_PI;
                    data[parameter_idx+1] = 2 * M_PI;
                    parameter_idx = parameter_idx + 2;
                }
                else if ((u3_gate->get_parameter_num() == 3)) {
                    data[parameter_idx] = 4 * M_PI;
                    data[parameter_idx+1] = 2 * M_PI;
                    data[parameter_idx+2] = 2 * M_PI;
                    parameter_idx = parameter_idx + 3;
                }
                break; }
            case RX_OPERATION:
            case RY_OPERATION:
            case CRY_OPERATION:
            case ADAPTIVE_OPERATION:
                data[parameter_idx] = 4 * M_PI;
                parameter_idx = parameter_idx + 1;
                break;
            case CZ_NU_OPERATION:
                data[parameter_idx] = 2 * M_PI;
                parameter_idx = parameter_idx + 1;
                break;
            case BLOCK_OPERATION: {
                Gates_block* block_gate = static_cast<Gates_block*>(gate);
                Matrix_real parameters_layer(range_max.get_data() + parameter_idx + gate->get_parameter_num(), 1, gate->get_parameter_num() );
                block_gate->get_parameter_max( parameters_layer );
                parameter_idx = parameter_idx + block_gate->get_parameter_num();
                break; }
            default:
                for (int i = 0; i < gate->get_parameter_num(); i++)
                    data[parameter_idx+i] = 2 * M_PI;    
                parameter_idx = parameter_idx + gate->get_parameter_num();
            }
        }
}

//////// experimental attributes to partition the circuits into subsegments. Advantageous in simulation of larger circuits ///////////űű


/**
@brief Call to apply the gate on the input array/matrix by input*Gate_block
@param input The input array on which the gate is applied
*/
void 
Gates_block::apply_from_right( Matrix_real& parameters_mtx, Matrix& input ) {


    //The stringstream input to store the output messages.
    std::stringstream sstream;



    // determine the number of parameters
    int parameters_num_total = 0;  
    for (int idx=0; idx<gates.size(); idx++) {

        // The current gate
        Gate* gate = gates[idx];
        parameters_num_total = parameters_num_total + gate->get_parameter_num();

    }


    double* parameters = parameters_mtx.get_data() + parameters_num_total;



    for( int idx=0; idx<(int)gates.size(); idx++) {

        Gate* operation = gates[idx];
        Matrix_real parameters_mtx(parameters-operation->get_parameter_num(), 1, operation->get_parameter_num());

        switch (operation->get_type()) {
        case CNOT_OPERATION: case CZ_OPERATION:
        case CH_OPERATION: case SYC_OPERATION:
        case X_OPERATION: case Y_OPERATION:
        case Z_OPERATION: case SX_OPERATION:
        case GENERAL_OPERATION: case H_OPERATION:
            operation->apply_from_right(input);
            break;
        case U3_OPERATION: {
            U3* u3_operation = static_cast<U3*>(operation);
            u3_operation->apply_from_right( parameters_mtx, input );
            break; 
        }
        case RX_OPERATION: {
            RX* rx_operation = static_cast<RX*>(operation);
            rx_operation->apply_from_right( parameters_mtx, input );
            break; 
        }
        case RY_OPERATION: {
            RY* ry_operation = static_cast<RY*>(operation);
            ry_operation->apply_from_right( parameters_mtx, input );
            break; 
        }
        case CRY_OPERATION: {
            CRY* cry_operation = static_cast<CRY*>(operation);
            cry_operation->apply_from_right( parameters_mtx, input );
            break; 
        }
        case RZ_OPERATION: {
            RZ* rz_operation = static_cast<RZ*>(operation);
            rz_operation->apply_from_right( parameters_mtx, input );
            break;         
        }
        case UN_OPERATION: {
            UN* un_operation = static_cast<UN*>(operation);
            un_operation->apply_from_right( parameters_mtx, input );
            break; 
        }
        case ON_OPERATION: {
            ON* on_operation = static_cast<ON*>(operation);
            on_operation->apply_from_right( parameters_mtx, input );
            break; 
        }
        case BLOCK_OPERATION: {
            Gates_block* block_operation = static_cast<Gates_block*>(operation);
            block_operation->apply_from_right(parameters_mtx, input);
            break;
        }
        case CZ_NU_OPERATION: {
            CZ_NU* cz_nu_operation = static_cast<CZ_NU*>(operation);
            cz_nu_operation->apply_from_right( parameters_mtx, input );
            break; 
        }        
        case COMPOSITE_OPERATION: {
            Composite* com_operation = static_cast<Composite*>(operation);
            com_operation->apply_from_right( parameters_mtx, input );
            break; 
        }
        case ADAPTIVE_OPERATION: {
            Adaptive* ad_operation = static_cast<Adaptive*>(operation);
            ad_operation->apply_from_right( parameters_mtx, input );
            break; 
        }
        default:
            std::string err("Gates_block::apply_from_right: unimplemented gate"); 
            throw err;
        }

        parameters = parameters - operation->get_parameter_num();

#ifdef DEBUG
        if (input.isnan()) { 
            std::stringstream sstream;
	    sstream << "Gates_block::apply_from_right: transformed matrix contains NaN." << std::endl;
            print(sstream, 0);	            
        }
#endif


    }


}


/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP (NOT IMPLEMENTED YET) and 2 for parallel with TBB (optional)
*/
std::vector<Matrix> 
Gates_block::apply_derivate_to( Matrix_real& parameters_mtx_in, Matrix& input, int parallel ) {

    //The stringstream input to store the output messages.
    std::stringstream sstream;
  
    std::vector<Matrix> grad(parameter_num, Matrix(0,0));

    int work_batch = 1;
    if ( parallel == 0 ) {
        work_batch = gates.size();
    }
    else {
        work_batch = 1;
    }

    // deriv_idx ... the index of the gate block for which the gradient is to be calculated

    tbb::parallel_for( tbb::blocked_range<int>(0,gates.size(),work_batch), [&](tbb::blocked_range<int> r) {
        for (int deriv_idx=r.begin(); deriv_idx<r.end(); ++deriv_idx) { 

        //for (int deriv_idx=0; deriv_idx<gates.size(); ++deriv_idx) { 


            Gate* gate_deriv = gates[deriv_idx];            

            // for constant gate no gardient component is calculated
            if ( gate_deriv->get_parameter_num() == 0 ) {
                continue;
            }
            
            int deriv_parameter_idx = gate_deriv->get_parameter_start_idx();       



            Matrix&& input_loc = input.copy();

            std::vector<Matrix> grad_loc;

            for( int idx=0; idx<gates.size(); idx++) {            

                Gate* operation = gates[idx];
                
                
                Matrix_real parameters_mtx(parameters_mtx_in.get_data() + operation->get_parameter_start_idx(), 1, operation->get_parameter_num());
                
                switch (operation->get_type()) {
                case UN_OPERATION:
                case ON_OPERATION:
                case SYC_OPERATION:
                case COMPOSITE_OPERATION: {
                        int gate_type = (int)operation->get_type();
                        std::string err( "Gates_block::apply_derivate_to: Given operation not supported in gardient calculation");
			throw( err );
	                break;
                }
                default :
                
                    if ( operation->get_parameter_num() == 0 ) {
                
                        if( idx < deriv_idx ) {
                            operation->apply_to( input_loc, parallel );    
                        }
                        else {
                            operation->apply_to_list(grad_loc, parallel );
                        }
                    
                    }
                    else  {
                
                        if( idx < deriv_idx ) {
                            operation->apply_to( parameters_mtx, input_loc, parallel );    
                        }
                        else if ( idx == deriv_idx ) {
                            grad_loc = operation->apply_derivate_to( parameters_mtx, input_loc, parallel );
                        }
                        else {
                            operation->apply_to_list(parameters_mtx, grad_loc, parallel );
                        }                       
                    
                    }
                    
                }
            }


            for ( int idx = 0; idx<(int)grad_loc.size(); idx++ ) {
                grad[deriv_parameter_idx+idx] = grad_loc[idx];
            }


        } // tbb range end
    
    });
   

    return grad;

}

/**
@brief Append a U3 gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param Theta The Theta parameter of the U3 operation
@param Phi The Phi parameter of the U3 operation
@param Lambda The Lambda parameter of the U3 operation
*/
void Gates_block::add_u3(int target_qbit, bool Theta, bool Phi, bool Lambda) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new U3( qbit_num, target_qbit, Theta, Phi, Lambda ));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}

/**
@brief Add a U3 gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param Theta The Theta parameter of the U3 operation
@param Phi The Phi parameter of the U3 operation
@param Lambda The Lambda parameter of the U3 operation
*/
void Gates_block::add_u3_to_front(int target_qbit, bool Theta, bool Phi, bool Lambda) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new U3( qbit_num, target_qbit, Theta, Phi, Lambda ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}



/**
@brief Append a RX gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_rx(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new RX( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}

/**
@brief Add a RX gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_rx_to_front(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new RX( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}



/**
@brief Append a RY gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_ry(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new RY( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}


/**
@brief Add a RY gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_ry_to_front(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new RY( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}




/**
@brief Append a CRY gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cry(int target_qbit, int control_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new CRY( qbit_num, target_qbit, control_qbit));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}



/**
@brief Add a CRY gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cry_to_front(int target_qbit, int control_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new CRY( qbit_num, target_qbit, control_qbit ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}




/**
@brief Append a CZ_NU gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cz_nu(int target_qbit, int control_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new CZ_NU( qbit_num, target_qbit, control_qbit));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}



/**
@brief Add a CZ_NU gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cz_nu_to_front(int target_qbit, int control_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new CZ_NU( qbit_num, target_qbit, control_qbit ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}



/**
@brief Append a RZ gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_rz(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new RZ( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}

/**
@brief Add a RZ gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_rz_to_front(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new RZ( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}




/**
@brief Append a C_NOT gate operation to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cnot(  int target_qbit, int control_qbit) {

        // new cnot operation
        Gate* gate = static_cast<Gate*>(new CNOT(qbit_num, target_qbit, control_qbit ));

        // append the operation to the list
        add_gate(gate);

}



/**
@brief Add a C_NOT gate operation to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cnot_to_front( int target_qbit, int control_qbit) {

        // new cnot operation
        Gate* gate = static_cast<Gate*>(new CNOT(qbit_num, target_qbit, control_qbit ));

        // put the operation to tghe front of the list
        add_gate_to_front(gate);

}




/**
@brief Append a CZ gate operation to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cz(  int target_qbit, int control_qbit) {

        // new cz operation
        Gate* gate = static_cast<Gate*>(new CZ(qbit_num, target_qbit, control_qbit ));

        // append the operation to the list
        add_gate(gate);

}



/**
@brief Add a CZ gate operation to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cz_to_front(  int target_qbit, int control_qbit) {

        // new cz operation
        Gate* gate = static_cast<Gate*>(new CZ(qbit_num, target_qbit, control_qbit ));

        // put the operation to tghe front of the list
        add_gate_to_front(gate);

}

/**
@brief Append a Hadamard gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_h(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new H( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}

/**
@brief Add a Hadamard gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_h_to_front(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new H( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}





/**
@brief Append a X gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_x(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new X( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}

/**
@brief Add a X gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_x_to_front(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new X( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}



/**
@brief Append a Y gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_y(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new Y( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}

/**
@brief Add a Y gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_y_to_front(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new Y( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}



/**
@brief Append a Z gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_z(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new Z( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}

/**
@brief Add a Z gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_z_to_front(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new Z( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}




/**
@brief Append a SX gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_sx(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new SX( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}

/**
@brief Add a SX gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_sx_to_front(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new SX( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}





/**
@brief Append a Sycamore gate operation to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_syc(  int target_qbit, int control_qbit) {

        // new cz operation
        Gate* gate = static_cast<Gate*>(new SYC(qbit_num, target_qbit, control_qbit ));

        // append the operation to the list
        add_gate(gate);

}



/**
@brief Add a Sycamore gate operation to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_syc_to_front(  int target_qbit, int control_qbit) {

        // new cz operation
        Gate* gate = static_cast<Gate*>(new SYC(qbit_num, target_qbit, control_qbit ));

        // put the operation to tghe front of the list
        add_gate_to_front(gate);

}




/**
@brief Append a CH gate (i.e. controlled Hadamard gate) operation to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_ch(  int target_qbit, int control_qbit) {

        // new cz operation
        Gate* gate = static_cast<Gate*>(new CH(qbit_num, target_qbit, control_qbit ));

        // append the operation to the list
        add_gate(gate);

}



/**
@brief Add a CH gate (i.e. controlled Hadamard gate) operation to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_ch_to_front(  int target_qbit, int control_qbit) {

        // new cz operation
        Gate* gate = static_cast<Gate*>(new CH(qbit_num, target_qbit, control_qbit ));

        // put the operation to tghe front of the list
        add_gate_to_front(gate);

}

/**
@brief Append a list of gates to the list of gates
@param gates_in A list of operation class instances.
*/
void Gates_block::add_gates( std::vector<Gate*> gates_in) {

        for(std::vector<Gate*>::iterator it = gates_in.begin(); it != gates_in.end(); ++it) {
            add_gate( *it );
        }

}


/**
@brief Add an array of gates to the front of the list of gates
@param gates_in A list of operation class instances.
*/
void Gates_block::add_gates_to_front( std::vector<Gate*>  gates_in) {

        // adding gates in reversed order!!
        for(std::vector<Gate*>::iterator it = gates_in.end(); it != gates_in.begin(); --it) {
            add_gate_to_front( *it );
        }

}



/**
@brief Append a UN gate to the list of gates
*/
void Gates_block::add_un() {

        // create the operation
        Gate* operation = static_cast<Gate*>(new UN( qbit_num ));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}

/**
@brief Add a UN gate to the front of the list of gates
*/
void Gates_block::add_un_to_front() {

        // create the operation
        Gate* gate = static_cast<Gate*>(new UN( qbit_num ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}


/**
@brief Append a ON gate to the list of gates
*/
void Gates_block::add_on() {

        // create the operation
        Gate* operation = static_cast<Gate*>(new ON( qbit_num ));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}

/**
@brief Add a ON gate to the front of the list of gates
*/
void Gates_block::add_on_to_front() {

        // create the operation
        Gate* gate = static_cast<Gate*>(new ON( qbit_num ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}


/**
@brief Append a Composite gate to the list of gates
*/
void Gates_block::add_composite()  {

        // create the operation
        Gate* operation = static_cast<Gate*>(new Composite( qbit_num ));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}

/**
@brief Add a Composite gate to the front of the list of gates
*/
void Gates_block::add_composite_to_front()  {

        // create the operation
        Gate* gate = static_cast<Gate*>(new Composite( qbit_num ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}



/**
@brief Append a Adaptive gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_adaptive(int target_qbit, int control_qbit)  {

        // create the operation
        Gate* operation = static_cast<Gate*>(new Adaptive( qbit_num, target_qbit, control_qbit));

        // adding the operation to the end of the list of gates
        add_gate( operation );
}


/**
@brief Add a Adaptive gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_adaptive_to_front(int target_qbit, int control_qbit)  {

        // create the operation
        Gate* gate = static_cast<Gate*>(new Adaptive( qbit_num, target_qbit, control_qbit ));

        // adding the operation to the front of the list of gates
        add_gate_to_front( gate );

}



/**
@brief Append a general gate to the list of gates
@param gate A pointer to a class Gate describing a gate operation.
*/
void Gates_block::add_gate( Gate* gate ) {

        //set the number of qubit in the gate
        gate->set_qbit_num( qbit_num );
        
        // determine the parents of the gate
        determine_parents( gate );

        // append the gate to the list
        gates.push_back(gate);
        
        // set the parameter starting index in the parameters array used to execute the circuit.
        gate->set_parameter_start_idx( parameter_num );

        // increase the number of parameters by the number of parameters
        parameter_num = parameter_num + gate->get_parameter_num();

        // increase the number of layers if necessary
        if (gate->get_type() == BLOCK_OPERATION) {
            layer_num = layer_num + 1;
        }

}

/**
@brief Add a gate to the front of the list of gates
@param gate A pointer to a class Gate describing a gate.
*/
 void Gates_block::add_gate_to_front( Gate* gate) {


        // set the number of qubit in the gate
        gate->set_qbit_num( qbit_num );

        // determine the parents of the gate
        determine_children( gate );

        gates.insert( gates.begin(), gate);

        // increase the number of U3 gate parameters by the number of parameters
        parameter_num = parameter_num + gate->get_parameter_num();

        // increase the number of layers if necessary
        if (gate->get_type() == BLOCK_OPERATION) {
            layer_num = layer_num + 1;
        }
        
        // TODO: develop a more efficient method for large circuits. Now it is updating the whole circuit
        reset_parameter_start_indices();
        reset_dependency_graph();

}



/**
@brief Call to insert a gate at a given position 
@param gate A pointer to a class Gate describing a gate.
@param idx The position where to insert the gate.
*/
void 
Gates_block::insert_gate( Gate* gate, int idx ) {


        // set the number of qubit in the gate
        gate->set_qbit_num( qbit_num );

        gates.insert( gates.begin()+idx, gate);

        // increase the number of U3 gate parameters by the number of parameters
        parameter_num = parameter_num + gate->get_parameter_num();

        // increase the number of layers if necessary
        if (gate->get_type() == BLOCK_OPERATION) {
            layer_num = layer_num + 1;
        }
        
        // TODO: develop a more efficient method for large circuits. Now it is updating the whole circuit
        reset_parameter_start_indices();
        reset_dependency_graph();


}



/**
@brief Call to get the number of the individual gate types in the list of gates
@return Returns with an instance gates_num describing the number of the individual gate types
*/
gates_num Gates_block::get_gate_nums() {

        gates_num gate_nums;

        gate_nums.u3      = 0;
        gate_nums.rx      = 0;
        gate_nums.ry      = 0;
        gate_nums.cry      = 0;
        gate_nums.rz      = 0;
        gate_nums.cnot    = 0;
        gate_nums.cz      = 0;
        gate_nums.ch      = 0;
        gate_nums.x       = 0;
        gate_nums.z       = 0;
        gate_nums.y       = 0;
        gate_nums.sx      = 0;
        gate_nums.syc     = 0;
        gate_nums.cz_nu   = 0;
        gate_nums.un     = 0;
        gate_nums.on     = 0;
        gate_nums.com     = 0;
        gate_nums.general = 0;
        gate_nums.adap = 0;
        gate_nums.total = 0;

        for(std::vector<Gate*>::iterator it = gates.begin(); it != gates.end(); ++it) {
            // get the specific gate or block of gates
            Gate* gate = *it;

            if (gate->get_type() == BLOCK_OPERATION) {

                Gates_block* block_gate = static_cast<Gates_block*>(gate);
                gates_num gate_nums_loc = block_gate->get_gate_nums();
                gate_nums.u3   = gate_nums.u3 + gate_nums_loc.u3;
                gate_nums.rx   = gate_nums.rx + gate_nums_loc.rx;
                gate_nums.ry   = gate_nums.ry + gate_nums_loc.ry;
                gate_nums.cry   = gate_nums.cry + gate_nums_loc.cry;
                gate_nums.rz   = gate_nums.rz + gate_nums_loc.rz;
                gate_nums.cnot = gate_nums.cnot + gate_nums_loc.cnot;
                gate_nums.cz = gate_nums.cz + gate_nums_loc.cz;
                gate_nums.ch = gate_nums.ch + gate_nums_loc.ch;
                gate_nums.h  = gate_nums.h + gate_nums_loc.h;
                gate_nums.x  = gate_nums.x + gate_nums_loc.x;
                gate_nums.sx = gate_nums.sx + gate_nums_loc.sx;
                gate_nums.syc   = gate_nums.syc + gate_nums_loc.syc;
                gate_nums.un   = gate_nums.un + gate_nums_loc.un;
                gate_nums.on   = gate_nums.on + gate_nums_loc.on;
                gate_nums.com  = gate_nums.com + gate_nums_loc.com;
                gate_nums.adap = gate_nums.adap + gate_nums_loc.adap;
                gate_nums.total = gate_nums.total + gate_nums_loc.total;

            }
            else if (gate->get_type() == U3_OPERATION) {
                gate_nums.u3   = gate_nums.u3 + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == RX_OPERATION) {
                gate_nums.rx   = gate_nums.rx + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == RY_OPERATION) {
                gate_nums.ry   = gate_nums.ry + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == CRY_OPERATION) {
                gate_nums.cry   = gate_nums.cry + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == RZ_OPERATION) {
                gate_nums.rz   = gate_nums.rz + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == CNOT_OPERATION) {
                gate_nums.cnot   = gate_nums.cnot + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == CZ_OPERATION) {
                gate_nums.cz   = gate_nums.cz + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == CH_OPERATION) {
                gate_nums.ch   = gate_nums.ch + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == H_OPERATION) {
                gate_nums.h   = gate_nums.h + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == X_OPERATION) {
                gate_nums.x   = gate_nums.x + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == Y_OPERATION) {
                gate_nums.y   = gate_nums.y + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == Z_OPERATION) {
                gate_nums.z   = gate_nums.z + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == SX_OPERATION) {
                gate_nums.sx   = gate_nums.sx + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == SYC_OPERATION) {
                gate_nums.syc   = gate_nums.syc + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == GENERAL_OPERATION) {
                gate_nums.general   = gate_nums.general + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == UN_OPERATION) {
                gate_nums.un   = gate_nums.un + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == ON_OPERATION) {
                gate_nums.on   = gate_nums.on + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == COMPOSITE_OPERATION) {
                gate_nums.com   = gate_nums.com + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else if (gate->get_type() == CZ_NU_OPERATION) {
                gate_nums.cz_nu   = gate_nums.cz_nu + 1;
                gate_nums.total = gate_nums.total + 1;
            }            
            else if (gate->get_type() == ADAPTIVE_OPERATION) {
                gate_nums.adap   = gate_nums.adap + 1;
                gate_nums.total = gate_nums.total + 1;
            }
            else {
                std::string err("Gates_block::get_gate_nums: unimplemented gate"); 
                throw err;
            }

        }


        return gate_nums;

}


/**
@brief Call to get the number of free parameters
@return Return with the number of parameters of the gates grouped in the gate block.
*/
int Gates_block::get_parameter_num() {
    return parameter_num;
}




/**
@brief Call to get the number of gates grouped in the class
@return Return with the number of the gates grouped in the gate block.
*/
int Gates_block::get_gate_num() {
    return gates.size();
}


/**
@brief Call to print the list of gates stored in the block of gates for a specific set of parameters
@param parameters The parameters of the gates that should be printed.
@param start_index The ordinal number of the first gate.
*/
void Gates_block::list_gates( const Matrix_real &parameters, int start_index ) {


	//The stringstream input to store the output messages.
	std::stringstream sstream;
	sstream << std::endl << "The gates in the list of gates:" << std::endl;
	print(sstream, 1);	    	
		
        int gate_idx = start_index;
        int parameter_idx = 0;
	double *parameters_data = parameters.get_data();
	//const_cast <Matrix_real&>(parameters);


        for(int op_idx = 0; op_idx<gates.size(); op_idx++) {

            Gate* gate = gates[op_idx];

            if (gate->get_type() == CNOT_OPERATION) {
                CNOT* cnot_gate = static_cast<CNOT*>(gate);
                std::stringstream sstream;
		sstream << gate_idx << "th gate: CNOT with control qubit: " << cnot_gate->get_control_qbit() << " and target qubit: " << cnot_gate->get_target_qbit() << std::endl;
		print(sstream, 1);   		
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == CZ_OPERATION) {
                CZ* cz_gate = static_cast<CZ*>(gate);
		std::stringstream sstream;
		sstream << gate_idx << "th gate: CZ with control qubit: " << cz_gate->get_control_qbit() << " and target qubit: " << cz_gate->get_target_qbit() << std::endl;
		print(sstream, 1);	    	             
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == CH_OPERATION) {
                CH* ch_gate = static_cast<CH*>(gate);
		std::stringstream sstream;
		sstream << gate_idx << "th gate: CH with control qubit: " << ch_gate->get_control_qbit() << " and target qubit: " << ch_gate->get_target_qbit() << std::endl;
		print(sstream, 1);	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == SYC_OPERATION) {
                SYC* syc_gate = static_cast<SYC*>(gate);
		std::stringstream sstream;
		sstream << gate_idx << "th gate: Sycamore gate with control qubit: " << syc_gate->get_control_qbit() << " and target qubit: " << syc_gate->get_target_qbit() << std::endl;
		print(sstream, 1);	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == U3_OPERATION) {

                // definig the U3 parameters
                double vartheta;
                double varphi;
                double varlambda;
		
                // get the inverse parameters of the U3 rotation

                U3* u3_gate = static_cast<U3*>(gate);

                if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_theta_parameter()) {
		   
                    vartheta = std::fmod( 2*parameters_data[parameter_idx], 4*M_PI);
                    varphi = 0;
                    varlambda =0;
                    parameter_idx = parameter_idx + 1;

                }
                else if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_phi_parameter()) {
                    vartheta = 0;
                    varphi = std::fmod( parameters_data[parameter_idx], 2*M_PI);
                    varlambda =0;
                    parameter_idx = parameter_idx + 1;
                }
                else if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_lambda_parameter()) {
                    vartheta = 0;
                    varphi =  0;
                    varlambda = std::fmod( parameters_data[parameter_idx], 2*M_PI);
                    parameter_idx = parameter_idx + 1;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_theta_parameter() && u3_gate->is_phi_parameter() ) {
                    vartheta = std::fmod( 2*parameters_data[parameter_idx], 4*M_PI);
                    varphi = std::fmod( parameters_data[parameter_idx+1], 2*M_PI);
                    varlambda = 0;
                    parameter_idx = parameter_idx + 2;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_theta_parameter() && u3_gate->is_lambda_parameter() ) {
                    vartheta = std::fmod( 2*parameters_data[parameter_idx], 4*M_PI);
                    varphi = 0;
                    varlambda = std::fmod( parameters_data[parameter_idx+1], 2*M_PI);
                    parameter_idx = parameter_idx + 2;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_phi_parameter() && u3_gate->is_lambda_parameter() ) {
                    vartheta = 0;
                    varphi = std::fmod( parameters_data[parameter_idx], 2*M_PI);
                    varlambda = std::fmod( parameters_data[parameter_idx+1], 2*M_PI);
                    parameter_idx = parameter_idx + 2;
                }
                else if ((u3_gate->get_parameter_num() == 3)) {
                    vartheta = std::fmod( 2*parameters_data[parameter_idx], 4*M_PI);
                    varphi = std::fmod( parameters_data[parameter_idx+1], 2*M_PI);
                    varlambda = std::fmod( parameters_data[parameter_idx+2], 2*M_PI);
                    parameter_idx = parameter_idx + 3;
                }

//                message = message + "U3 on target qubit %d with parameters theta = %f, phi = %f and lambda = %f";

		std::stringstream sstream;
		sstream << gate_idx << "th gate: U3 on target qubit: " << u3_gate->get_target_qbit() << " and with parameters theta = " << vartheta << ", phi = " << varphi << " and lambda = " << varlambda << std::endl;
		print(sstream, 1);	    	
		                
                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == RX_OPERATION) {
	        // definig the rotation parameter
                double vartheta;
                // get the inverse parameters of the U3 rotation
                RX* rx_gate = static_cast<RX*>(gate);		
                vartheta = std::fmod( 2*parameters_data[parameter_idx], 4*M_PI);
                parameter_idx = parameter_idx + 1;

		std::stringstream sstream;
		sstream << gate_idx << "th gate: RX on target qubit: " << rx_gate->get_target_qbit() << " and with parameters theta = " << vartheta << std::endl;
		print(sstream, 1);	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == RY_OPERATION) {
                // definig the rotation parameter
                double vartheta;
                // get the inverse parameters of the U3 rotation
                RY* ry_gate = static_cast<RY*>(gate);
                vartheta = std::fmod( 2*parameters_data[parameter_idx], 4*M_PI);
                parameter_idx = parameter_idx + 1;

		std::stringstream sstream;
		sstream << gate_idx << "th gate: RY on target qubit: " << ry_gate->get_target_qbit() << " and with parameters theta = " << vartheta << std::endl; 
		print(sstream, 1);	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == CRY_OPERATION) {
                // definig the rotation parameter
                double vartheta;
                // get the inverse parameters of the U3 rotation
                CRY* cry_gate = static_cast<CRY*>(gate);
                vartheta = std::fmod( 2*parameters_data[parameter_idx], 4*M_PI);
                parameter_idx = parameter_idx + 1;

		std::stringstream sstream;
		sstream << gate_idx << "th gate: CRY on target qubit: " << cry_gate->get_target_qbit() << ", control qubit" << cry_gate->get_control_qbit() << " and with parameters theta = " << vartheta << std::endl;
		print(sstream, 1);	    		                    
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == RZ_OPERATION) {
                // definig the rotation parameter
                double varphi;
                // get the inverse parameters of the U3 rotation
                RZ* rz_gate = static_cast<RZ*>(gate);
                varphi = std::fmod( 2*parameters_data[parameter_idx], 2*M_PI);
                parameter_idx = parameter_idx + 1;

		std::stringstream sstream;
		sstream << gate_idx << "th gate: RZ on target qubit: " << rz_gate->get_target_qbit() << " and with parameters varphi = " << varphi << std::endl;
		print(sstream, 1);	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == H_OPERATION) {
                // get the inverse parameters of the U3 rotation
                H* h_gate = static_cast<H*>(gate);
		std::stringstream sstream;
		sstream << gate_idx << "th gate: Hadamard on target qubit: " << h_gate->get_target_qbit() << std::endl;
		print(sstream, 1);	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == X_OPERATION) {
                // get the inverse parameters of the U3 rotation
                X* x_gate = static_cast<X*>(gate);
		std::stringstream sstream;
		sstream << gate_idx << "th gate: X on target qubit: " << x_gate->get_target_qbit() << std::endl;
		print(sstream, 1);	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == Y_OPERATION) {
                // get the inverse parameters of the U3 rotation
                Y* y_gate = static_cast<Y*>(gate);
		std::stringstream sstream;
		sstream << gate_idx << "th gate: Y on target qubit: " << y_gate->get_target_qbit() << std::endl;
		print(sstream, 1);	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == Z_OPERATION) {
                // get the inverse parameters of the U3 rotation
                Z* z_gate = static_cast<Z*>(gate);
		std::stringstream sstream;
		sstream << gate_idx << "th gate: Z on target qubit: " << z_gate->get_target_qbit() << std::endl;
		print(sstream, 1);	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == SX_OPERATION) {
                // get the inverse parameters of the U3 rotation
                SX* sx_gate = static_cast<SX*>(gate);

		std::stringstream sstream;
		sstream << gate_idx << "th gate: SX on target qubit: " << sx_gate->get_target_qbit() << std::endl;
		print(sstream, 1);	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == BLOCK_OPERATION) {
                Gates_block* block_gate = static_cast<Gates_block*>(gate);
                const Matrix_real parameters_layer(parameters.get_data() + parameter_idx, 1, gate->get_parameter_num() );
                block_gate->list_gates( parameters_layer, gate_idx );
                parameter_idx = parameter_idx + block_gate->get_parameter_num();
                gate_idx = gate_idx + block_gate->get_gate_num();
            }
            else if (gate->get_type() == UN_OPERATION) {
                parameter_idx = parameter_idx + gate->get_parameter_num();

		std::stringstream sstream;
		sstream << gate_idx << "th gate: UN " << gate->get_parameter_num() << " parameters" << std::endl;
		print(sstream, 1);	    	         
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == CZ_NU_OPERATION) {
                // definig the rotation parameter
                double Theta;
                // get the inverse parameters of the U3 rotation
                CZ_NU* cz_nu_gate = static_cast<CZ_NU*>(gate);
                Theta = std::fmod( parameters_data[parameter_idx], 2*M_PI);
                parameter_idx = parameter_idx +1;

		std::stringstream sstream;
		sstream << gate_idx << "th gate: CZ_NU gate on target qubit: " << cz_nu_gate->get_target_qbit() << ", control qubit " << cz_nu_gate->get_control_qbit() << " and with parameters Theta = " << Theta << std::endl;
		print(sstream, 1);	    	
                gate_idx = gate_idx + 1;
            }         
            else if (gate->get_type() == ON_OPERATION) {
                parameter_idx = parameter_idx + gate->get_parameter_num();
		std::stringstream sstream;
		sstream << gate_idx << "th gate: ON " << gate->get_parameter_num() << " parameters" << std::endl;
		print(sstream, 1);	    	 
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == COMPOSITE_OPERATION) {
                parameter_idx = parameter_idx + gate->get_parameter_num();

		std::stringstream sstream;
		sstream << gate_idx << "th gate: Composite " << gate->get_parameter_num() << " parameters" << std::endl;
		print(sstream, 1);	    	               
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == ADAPTIVE_OPERATION) {
                // definig the rotation parameter
                double Theta;
                // get the inverse parameters of the U3 rotation
                Adaptive* ad_gate = static_cast<Adaptive*>(gate);
                Theta = std::fmod( 2*parameters_data[parameter_idx], 4*M_PI);
                parameter_idx = parameter_idx + 1;

		std::stringstream sstream;
		sstream << gate_idx << "th gate: Adaptive gate on target qubit: " << ad_gate->get_target_qbit() << ", control qubit " << ad_gate->get_control_qbit() << " and with parameters Theta = " << Theta << std::endl;
		print(sstream, 1);	    	
                gate_idx = gate_idx + 1;
            }
            else {
                std::string err("Gates_block::list_gates: unimplemented gate"); 
                throw err;
            }

        }


}


/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void Gates_block::reorder_qubits( std::vector<int>  qbit_list) {

    for(std::vector<Gate*>::iterator it = gates.begin(); it != gates.end(); ++it) {

        Gate* gate = *it;

        if (gate->get_type() == CNOT_OPERATION) {
            CNOT* cnot_gate = static_cast<CNOT*>(gate);
            cnot_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == CZ_OPERATION) {
            CZ* cz_gate = static_cast<CZ*>(gate);
            cz_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == CH_OPERATION) {
            CH* ch_gate = static_cast<CH*>(gate);
            ch_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == SYC_OPERATION) {
            SYC* syc_gate = static_cast<SYC*>(gate);
            syc_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == U3_OPERATION) {
             U3* u3_gate = static_cast<U3*>(gate);
             u3_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == RX_OPERATION) {
             RX* rx_gate = static_cast<RX*>(gate);
             rx_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == RY_OPERATION) {
             RY* ry_gate = static_cast<RY*>(gate);
             ry_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == CRY_OPERATION) {
             CRY* cry_gate = static_cast<CRY*>(gate);
             cry_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == RZ_OPERATION) {
             RZ* rz_gate = static_cast<RZ*>(gate);
             rz_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == H_OPERATION) {
             H* h_gate = static_cast<H*>(gate);
             h_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == X_OPERATION) {
             X* x_gate = static_cast<X*>(gate);
             x_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == Y_OPERATION) {
             Y* y_gate = static_cast<Y*>(gate);
             y_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == Z_OPERATION) {
             Z* z_gate = static_cast<Z*>(gate);
             z_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == SX_OPERATION) {
             SX* sx_gate = static_cast<SX*>(gate);
             sx_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == BLOCK_OPERATION) {
             Gates_block* block_gate = static_cast<Gates_block*>(gate);
             block_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == UN_OPERATION) {
             UN* un_gate = static_cast<UN*>(gate);
             un_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == ON_OPERATION) {
             ON* on_gate = static_cast<ON*>(gate);
             on_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == COMPOSITE_OPERATION) {
             Composite* com_gate = static_cast<Composite*>(gate);
             com_gate->reorder_qubits( qbit_list );
         }
         else if (gate->get_type() == ADAPTIVE_OPERATION) {
             Adaptive* ad_gate = static_cast<Adaptive*>(gate);
             ad_gate->reorder_qubits( qbit_list );
         }
         else {
             std::string err("Gates_block::reorder_qubits: unimplemented gate"); 
             throw err;
         }


    }

}



/**
@brief Call to get the qubits involved in the gates stored in the block of gates.
@return Return with a list of the invovled qubits
*/
std::vector<int> Gates_block::get_involved_qubits() {

    std::vector<int> involved_qbits;

    int qbit;


    for(std::vector<Gate*>::iterator it = gates.begin(); it != gates.end(); ++it) {

        Gate* gate = *it;

        qbit = gate->get_target_qbit();
        if (qbit != -1) {
            add_unique_elelement( involved_qbits, qbit );
        }


        qbit = gate->get_control_qbit();
        if (qbit != -1) {
            add_unique_elelement( involved_qbits, qbit );
        }

    }

    return involved_qbits;
}


/**
@brief Call to get the gates stored in the class. (The resulting vector contains borrowed pointers to the gates, so they dont need to be deleted.)
@return Return with a list of the gates.
*/
std::vector<Gate*> Gates_block::get_gates() {
    return gates;
}


/**
@brief Call to get the gates stored in the class.
@return Return with a list of the gates.
*/
Gate* Gates_block::get_gate(int idx) {

    if (idx > (int)gates.size() ) {
        return NULL;
    }

    return gates[idx];
}


/**
@brief Call to append the gates of a gate block to the current circuit.
@param op_block A pointer to an instance of class Gates_block
*/
void Gates_block::combine(Gates_block* op_block) {

    // getting the list of gates
    std::vector<Gate*> gates_in = op_block->get_gates();

    int qbit_num_loc = op_block->get_qbit_num();
    if ( qbit_num_loc != qbit_num ) {
        std::string err("Gates_block::combine: number of qubits in the circuits must be the same"); 
        throw err;
    }

    for(std::vector<Gate*>::iterator it = (gates_in).begin(); it != (gates_in).end(); ++it) {
        Gate* op = *it;
        Gate* op_cloned = op->clone();
        add_gate( op_cloned );
    }

}


/**
@brief Set the number of qubits spanning the matrix of the gates stored in the block of gates.
@param qbit_num_in The number of qubits spanning the matrices.
*/
void Gates_block::set_qbit_num( int qbit_num_in ) {

    if (qbit_num_in > 30) {
        std::string err("Gates_block::set_qbit_num: Number of qubits supported up to 30"); 
        throw err;        
    }

    // setting the number of qubits
    Gate::set_qbit_num(qbit_num_in);

    // setting the number of qubit in the gates
    for(std::vector<Gate*>::iterator it = gates.begin(); it != gates.end(); ++it) {
        Gate* op = *it;
        switch (op->get_type()) {
        case CNOT_OPERATION: case CZ_OPERATION:
        case CH_OPERATION: case SYC_OPERATION:
        case U3_OPERATION: case RY_OPERATION:
        case CRY_OPERATION: case RX_OPERATION:
        case RZ_OPERATION: case X_OPERATION:
        case Y_OPERATION: case Z_OPERATION:
        case SX_OPERATION: case BLOCK_OPERATION:
        case GENERAL_OPERATION: case UN_OPERATION:
        case ON_OPERATION: case COMPOSITE_OPERATION:
        case ADAPTIVE_OPERATION:
        case H_OPERATION:
        case CZ_NU_OPERATION:
            op->set_qbit_num( qbit_num_in );
            break;
        default:
            std::string err("Gates_block::set_qbit_num: unimplemented gate"); 
            throw err;
        }
    }
}


/**
@brief Create a clone of the present class.
@return Return with a pointer pointing to the cloned object.
*/
Gates_block* Gates_block::clone() {

    // creatign new instance of class Gates_block
    Gates_block* ret = new Gates_block( qbit_num );
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    // extracting the gates from the current class
    if (extract_gates( ret ) != 0 ) {
        std::string err("Gates_block::clone(): extracting gates was not succesfull"); 
        throw err;
    };

    return ret;

}


/**
@brief Call to extract the gates stored in the class.
@param op_block An instance of Gates_block class in which the gates will be stored. (The current gates will be erased)
@return Return with 0 on success.
*/
int Gates_block::extract_gates( Gates_block* op_block ) {

    op_block->release_gates();

    for ( std::vector<Gate*>::iterator it=gates.begin(); it != gates.end(); ++it ) {
        Gate* op = *it;
        switch (op->get_type()) {
        case CNOT_OPERATION: case CZ_OPERATION:
        case CH_OPERATION: case SYC_OPERATION:
        case U3_OPERATION: case RY_OPERATION:
        case CRY_OPERATION: case RX_OPERATION:
        case RZ_OPERATION: case X_OPERATION:
        case Y_OPERATION: case Z_OPERATION:
        case SX_OPERATION: case BLOCK_OPERATION:
        case GENERAL_OPERATION: case UN_OPERATION:
        case ON_OPERATION: case COMPOSITE_OPERATION:
        case ADAPTIVE_OPERATION: 
        case H_OPERATION: 
        case CZ_NU_OPERATION:
        {
            Gate* op_cloned = op->clone();
            op_block->add_gate( op_cloned );
            break; }
        default:
            std::string err("Gates_block::extract_gates: unimplemented gate"); 
            throw err;
        }

    }
    
    return 0;

}



/**
@brief Call to determine, whether the circuit contains daptive gate or not.
@return Return with true if the circuit contains an adaptive gate.
*/
bool Gates_block::contains_adaptive_gate() {

    for ( std::vector<Gate*>::iterator it=gates.begin(); it != gates.end(); ++it ) {
        Gate* op = *it;

        if (op->get_type() == ADAPTIVE_OPERATION) {
            return true;
        }
        else if (op->get_type() == BLOCK_OPERATION) {
            Gates_block* block_op = static_cast<Gates_block*>( op );
            bool ret = block_op->contains_adaptive_gate();
            if ( ret ) return true;
        }
    }

    return false;

}




/**
@brief Call to determine, whether the sub-circuit at a given position in the circuit contains daptive gate or not.
@param idx The position of the gate to be checked.
@return Return with true if the circuit contains an adaptive gate.
*/
bool Gates_block::contains_adaptive_gate(int idx) {

    
    Gate* op = gates[idx];

    if (op->get_type() == ADAPTIVE_OPERATION) {
        return true;
    }
    else if (op->get_type() == BLOCK_OPERATION) {
        Gates_block* block_op = static_cast<Gates_block*>( op );
        return block_op->contains_adaptive_gate();
    }

    return false;

}



/**
@brief Call to evaluate the reduced densiy matrix.
@param parameters An array of parameters to calculate the entropy
@param input_state The input state on which the gate structure is applied
@param qbit_list Subset of qubits for which the entropy should be calculated. (Should conatin unique elements)
@Return Returns with the reduced density matrix.
*/
Matrix
Gates_block::get_reduced_density_matrix( Matrix_real& parameters_mtx, Matrix& input_state, matrix_base<int>& qbit_list_subset ) {


    if (input_state.cols != 1) {
        std::string error("Gates_block::get_reduced_density_matrix: The number of columns in input state should be 1");
        throw error;
    }


    if (input_state.rows != matrix_size) {
        std::string error("Gates_block::get_reduced_density_matrix: The number of rows in input state should be 2^qbit_num");
        throw error;
    }

    // determine the transformed state
    Matrix transformed_state = input_state.copy();  
    if ( parameters_mtx.size() > 0 ) {
        bool parallel = true;
        apply_to( parameters_mtx, transformed_state, parallel );
    }


    int subset_qbit_num = qbit_list_subset.size();
    int complementary_qbit_num = qbit_num - subset_qbit_num;


    // list of complementary qubits
    matrix_base<int> qbit_list_complementary( complementary_qbit_num, 1);
    int qbit_idx_count = 0;
    for (int qbit_idx=0; qbit_idx<qbit_num; qbit_idx++) {

        bool qbit_idx_in_subset = false;

        for (int subset_qbit_idx=0; subset_qbit_idx<subset_qbit_num; subset_qbit_idx++) {
            if ( qbit_idx == qbit_list_subset[subset_qbit_idx] ) {
                qbit_idx_in_subset = true;
                break;
            }
        }


        if ( qbit_idx_in_subset ) {
            continue;
        }

        qbit_list_complementary[qbit_idx_count] = qbit_idx;
        qbit_idx_count++;
    }


    // 000010000 one-hot encoded numbers indicating the bit position of the qubits in the register
    matrix_base<int> qbit_masks(qbit_num, 1);
    for (int qbit_idx=0; qbit_idx<qbit_num; qbit_idx++) {
        qbit_masks[ qbit_idx ] = 1 << qbit_idx;
    }



    // retrieve the reduced density matrix
    int rho_matrix_size = 1 << subset_qbit_num;

    Matrix rho(rho_matrix_size, rho_matrix_size);
    memset( rho.get_data(), 0.0, rho.size()*sizeof(QGD_Complex16) );




    int complementary_basis_num = 1 << complementary_qbit_num;
    for ( int row_idx=0; row_idx<rho_matrix_size; row_idx++ ) {

        // index of the amplitude in the state vector
        int idx = 0;
        for (int qbit_idx=0; qbit_idx<subset_qbit_num; qbit_idx++) {
            if ( row_idx & qbit_masks[ qbit_idx ] ) {
                idx = idx | qbit_masks[ qbit_list_subset[qbit_idx] ]; 
            }
        }

        for ( int col_idx=row_idx; col_idx<rho_matrix_size; col_idx++ ) {
        
            // index of the amplitude in the state vector
            int jdx = 0;
            for (int qbit_idx=0; qbit_idx<subset_qbit_num; qbit_idx++) {
                if ( col_idx & qbit_masks[ qbit_idx ] ) {
                    jdx = jdx | qbit_masks[ qbit_list_subset[qbit_idx] ]; 
                }
            }


            // thread local storage for partial permanent
            tbb::combinable<QGD_Complex16> priv_addend {[](){QGD_Complex16 ret; ret.real = 0.0; ret.imag = 0.0; return ret;}};


            tbb::parallel_for( tbb::blocked_range<int>(0, complementary_basis_num, 1024), [&](tbb::blocked_range<int> r) {

                QGD_Complex16& rho_element_priv = priv_addend.local();

                for (int compl_idx=r.begin(); compl_idx<r.end(); compl_idx++) {

 
                    int idx_loc = idx;
                    int jdx_loc = jdx;

                    for (int qbit_idx=0; qbit_idx<complementary_qbit_num; qbit_idx++) {
                        if ( compl_idx & qbit_masks[ qbit_idx ] ) {
                            idx_loc = idx_loc | qbit_masks[ qbit_list_complementary[qbit_idx] ]; 
                            jdx_loc = jdx_loc | qbit_masks[ qbit_list_complementary[qbit_idx] ]; 
                        }
                    }

                    QGD_Complex16  element_idx = transformed_state[ idx_loc ];
                    QGD_Complex16& element_jdx = transformed_state[ jdx_loc ];

                    // conjugate because of the bra-vector
                    element_idx.imag = -element_idx.imag;

                    QGD_Complex16 addend = mult( element_idx, element_jdx );

                    rho_element_priv.real = rho_element_priv.real + addend.real;
                    rho_element_priv.imag = rho_element_priv.imag + addend.imag;


                }
            });

        

            QGD_Complex16 rho_element;
            rho_element.real = 0.0;
            rho_element.imag = 0.0;

            priv_addend.combine_each([&](QGD_Complex16 &a) {
                rho_element.real = rho_element.real + a.real;
                rho_element.imag = rho_element.imag + a.imag;
            });

            rho[ row_idx * rho.stride + col_idx ].real += rho_element.real;
            rho[ row_idx * rho.stride + col_idx ].imag += rho_element.imag;

            if ( row_idx == col_idx ) {
                continue;
            }

            rho[ col_idx * rho.stride + row_idx ].real += rho_element.real;
            rho[ col_idx * rho.stride + row_idx ].imag -= rho_element.imag;

        }

        
    }


    // test the trace of the reduced density matrix
    double trace = 0.0;
    for( int idx=0; idx<rho_matrix_size; idx++) {
        trace = trace + rho[idx*rho.stride+idx].real;
    }

    if ( abs( trace-1.0 ) > 1e-6 ) {
        std::string error("Gates_block::get_reduced_density_matrix: The trace of the reduced density matrix is not unity");
        throw error;
    }

    return rho;


}


/**
@brief Call to evaluate the seconf Rényi entropy. The quantum circuit is applied on an input state input. The entropy is evaluated for the transformed state.
@param parameters An array of parameters to calculate the entropy
@param input_state The input state on which the gate structure is applied
@param qbit_list Subset of qubits for which the entropy should be calculated. (Should conatin unique elements)
@Return Returns with the calculated entropy
*/
double Gates_block::get_second_Renyi_entropy( Matrix_real& parameters_mtx, Matrix& input_state, matrix_base<int>& qbit_list_subset ) {

    // determine the reduced density matrix
    Matrix rho = get_reduced_density_matrix( parameters_mtx, input_state, qbit_list_subset );


    // calculate the second Rényi entropy 
    // Tr( rho @ rho )
    double trace_rho_square = 0.0;

    for (int idx=0; idx<rho.rows; idx++) {

        double trace_tmp = 0.0;

        for (int jdx=0; jdx<rho.rows; jdx++) {
            QGD_Complex16& element = rho[ idx*rho.stride + jdx ];
            double tmp = element.real * element.real + element.imag * element.imag;

            trace_tmp = trace_tmp + tmp;
        }


        trace_rho_square = trace_rho_square + trace_tmp;


    }




    double entropy = -log(trace_rho_square);


    return entropy;

}


/**
@brief Call to remove those integer elements from list1 which are present in list 
@param list1 A list of integers
@param list2 A list of integers
@return Returns with the reduced list determined from list1.
*/
std::vector<int> remove_list_intersection( std::vector<int>& list1, std::vector<int>& list2 ) {

    std::vector<int> ret = list1;

    for( std::vector<int>::iterator it2 = list2.begin(); it2 != list2.end(); it2++ ) {
    
        std::vector<int>::iterator element_found = std::find(ret.begin(), ret.end(), *it2);
        
        if( element_found != ret.end() ) {
            ret.erase( element_found );
        }    
        
    }
    
    return ret;



}


/**
@brief Call to obtain the parent gates in the circuit. A parent gate needs to be applied prior to the given gate. The parent gates are stored via the "parents" attribute of the gate instance
@param gate The gate for which the parents are determined. 
*/
void 
Gates_block::determine_parents( Gate* gate ) {


    std::vector<int>&& involved_qubits = gate->get_involved_qubits();
/*
    std::cout << "involved qubits in the current gate: " << std::endl;
    for( int idx=0; idx<involved_qubits.size(); idx++ ) {
    std::cout << involved_qubits[idx] << ", ";
    }
    std::cout << std::endl;
  */  
    // iterate over gates in the circuit
    for( int idx=gates.size()-1; idx>=0; idx-- ) {
        Gate* gate_loc = gates[idx];
        std::vector<int>&& involved_qubits_loc = gate_loc->get_involved_qubits();
        
        std::vector<int>&& reduced_qbit_list = remove_list_intersection( involved_qubits, involved_qubits_loc );
        
        if( reduced_qbit_list.size() < involved_qubits.size() ) {
            // parent gate found, setting parent-child relation
            
            gate->add_parent( gate_loc );
            gate_loc->add_child( gate );
            
            involved_qubits = std::move(reduced_qbit_list);
            
            
        }
        
        
        // return if no further involved qubits left
        if( involved_qubits.size() == 0 ) {
            break;
        }
    }
    
    
}
    
    
    
/**
@brief Call to obtain the child gates in the circuit. A child gate needs to be applied after the given gate. The children gates are stored via the "children" attribute of the gate instance
@param gate The gate for which the children are determined. 
*/
void 
Gates_block::determine_children( Gate* gate ) {


    std::vector<int>&& involved_qubits = gate->get_involved_qubits();
    
    // iterate over gates in the circuit
    for( int idx=0; idx<gates.size(); idx++ ) {
        Gate* gate_loc = gates[idx];
        std::vector<int>&& involved_qubits_loc = gate_loc->get_involved_qubits();
        
        std::vector<int>&& reduced_qbit_list = remove_list_intersection( involved_qubits, involved_qubits_loc );
        
        if( reduced_qbit_list.size() < involved_qubits.size() ) {
            // child gate found, setting parent-child relation
            
            gate->add_child( gate_loc );
            gate_loc->add_parent( gate );
            
            involved_qubits = std::move(reduced_qbit_list);
            
            
        }
        
        
        // return if no further involved qubits left
        if( involved_qubits.size() == 0 ) {
            break;
        }
    }    

}


/**
@brief Method reset the parameter start indices of gate operations incorporated in the circuit. (When a gate is inserted into the circuit at other position than the end.)
*/
void 
Gates_block::reset_parameter_start_indices() {

    int parameter_idx = 0;
    
    for( std::vector<Gate*>::iterator gate_it = gates.begin(); gate_it != gates.end(); gate_it++ ) {
    
        Gate* gate = *gate_it;
        
        gate->set_parameter_start_idx( parameter_idx );
        
        parameter_idx = parameter_idx + gate->get_parameter_num();
    
    }

}


/**
@brief Method to reset the dependency graph of the gates in the circuit
*/
void  
Gates_block::reset_dependency_graph() {

    // first clear parent/children data from the gates
    for( std::vector<Gate*>::iterator gate_it = gates.begin(); gate_it != gates.end(); gate_it++ ) {    
        Gate* gate = *gate_it;
        gate->clear_children();
        gate->clear_parents();    
    }


    // first clear parent/children data from the gates
    for( std::vector<Gate*>::iterator gate_it = gates.begin(); gate_it != gates.end(); gate_it++ ) {    
        Gate* gate = *gate_it;

        // determine the parents of the gate
        determine_parents( gate );

    }




}


/**
@brief Method to generate a flat circuit. A flat circuit is a circuit does not containing subcircuits: there are no Gates_block instances (containing subcircuits) in the resulting circuit. If the original circuit contains subcircuits, the gates in the subcircuits are directly incorporated in the resulting flat circuit.
*/
Gates_block* 
Gates_block::get_flat_circuit() {

    Gates_block* flat_circuit = new Gates_block( qbit_num );

    for( std::vector<Gate*>::iterator gate_it=gates.begin(); gate_it != gates.end(); gate_it++ ) {

        Gate* gate = *gate_it;

        if( gate->get_type() == BLOCK_OPERATION ) {

            Gates_block* circuit_inner      = static_cast<Gates_block*>( gate );
            Gates_block* flat_circuit_inner = circuit_inner->get_flat_circuit();

            flat_circuit->combine( flat_circuit_inner );

            delete( flat_circuit_inner );
        }
        else {
            flat_circuit->add_gate( gate->clone() );
        }

    }



    return flat_circuit;


}



/**
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is embedded.
@param parameters The parameter array corresponding to the circuit in which the gate is embedded
@return Returns with the array of the extracted parameters.
*/
Matrix_real 
Gates_block::extract_parameters( Matrix_real& parameters ) {

    if ( get_parameter_start_idx() + get_parameter_num() < parameters.size()  ) {
        std::string err("Gates_block::extract_parameters: Cant extract parameters, since th einput arary has not enough elements.");
        throw err;     
    }

    Matrix_real extracted_parameters(1, get_parameter_num());

    memcpy( extracted_parameters.get_data(), parameters.get_data() + get_parameter_start_idx(), get_parameter_num()*sizeof(double) );

    return extracted_parameters;

}


#ifdef __DFE__


/**
@brief Method to create random initial parameters for the optimization
@return 
*/
DFEgate_kernel_type* Gates_block::convert_to_DFE_gates_with_derivates( Matrix_real& parameters_mtx, int& gatesNum, int& gateSetNum, int& redundantGateSets, bool only_derivates ) {

    int parameter_num = get_parameter_num();
    if ( parameter_num != parameters_mtx.size() ) {
        std::string error("Gates_block::convert_to_DFE_gates: wrong number of parameters");
        throw error;
    }

    gates_num gate_nums   = get_gate_nums();
    int gates_total_num   = gate_nums.total; 
    int chained_gates_num = get_chained_gates_num();
    int gate_padding      = gates_total_num % chained_gates_num == 0 ? 0 : chained_gates_num - (gates_total_num % chained_gates_num);
    gatesNum              = gates_total_num+gate_padding;
/*
std::cout << "chained gates num: " << chained_gates_num << std::endl;
std::cout << "number of gates: " << gatesNum << std::endl;
*/


    gateSetNum = only_derivates ? parameter_num : parameter_num+1;

#ifdef __MPI__
    int rem = gateSetNum % (4 * world_size );
    if ( rem == 0 ) {
        redundantGateSets = 0;
    }
    else {
        redundantGateSets = (4 * world_size ) - (gateSetNum % (4 * world_size ));
        gateSetNum = gateSetNum + redundantGateSets;
    }
#else
    int rem = gateSetNum % 4;
    if ( rem == 0 ) {
        redundantGateSets = 0;
    }
    else {
        redundantGateSets = 4 - (gateSetNum % 4);
        gateSetNum = gateSetNum + redundantGateSets;
    }
#endif


    DFEgate_kernel_type* DFEgates = new DFEgate_kernel_type[gatesNum*gateSetNum];
    

    int gate_idx = 0;
    convert_to_DFE_gates( parameters_mtx, DFEgates, gate_idx );


    // padding with identity gates
    for (int idx=gate_idx; idx<gatesNum; idx++ ){

        DFEgate_kernel_type& DFEGate = DFEgates[idx];

        
        DFEGate.target_qbit = 0;
        DFEGate.control_qbit = -1;
        DFEGate.gate_type = U3_OPERATION;
        DFEGate.ThetaOver2 = (int32_t)(0);
        DFEGate.Phi = (int32_t)(0);
        DFEGate.Lambda = (int32_t)(0); 
        DFEGate.metadata = 0;

    }
/*
    for ( int idx=0; idx<gatesNum; idx++ ) {

        std::cout << "target qubit: " << (int)DFEgates[idx].target_qbit << " control qubit: " << (int)DFEgates[idx].control_qbit << " gate type: " << (int)DFEgates[idx].gate_type << std::endl; 
    }
*/

    // adjust parameters for derivation
    if (only_derivates ) {
        for (int idx=1; idx<(gateSetNum-1); idx++) {
           memcpy(DFEgates+idx*gatesNum, DFEgates, gatesNum*sizeof(DFEgate_kernel_type));
        }
    }
    else {
        for (int idx=0; idx<(gateSetNum-1); idx++) {
           memcpy(DFEgates+(idx+1)*gatesNum, DFEgates, gatesNum*sizeof(DFEgate_kernel_type));
        }
    }

    gate_idx = 0;
    int gate_set_index = parameter_num-1;
    if (only_derivates) {
        adjust_parameters_for_derivation( DFEgates, gatesNum, gate_idx, gate_set_index );
    } 
    else {
        adjust_parameters_for_derivation( DFEgates+gatesNum, gatesNum, gate_idx, gate_set_index );
    }

/*
    for ( int idx=0; idx<gatesNum*(parameter_num+1); idx++ ) {

        std::cout << "target qubit: " << (int)DFEgates[idx].target_qbit << " control qubit: " << (int)DFEgates[idx].control_qbit << " theta: " << (int)DFEgates[idx].ThetaOver2 << " lambda: " << (int)DFEgates[idx].Lambda << std::endl; 
    }
*/
    return DFEgates;

}


/**
@brief Method to create random initial parameters for the optimization
@return 
*/
void Gates_block::adjust_parameters_for_derivation( DFEgate_kernel_type* DFEgates, const int gatesNum, int& gate_idx, int& gate_set_index) {

        int parameter_idx = parameter_num;
        //int gate_set_index = parameter_num-1;

        int32_t parameter_shift = (int32_t)(M_PI/2*(1<<25));

        for(int op_idx = gates.size()-1; op_idx>=0; op_idx--) { 

            Gate* gate = gates[op_idx];
//std::cout <<   gate_idx << " " <<   gate_set_index << " " << gate->get_type() << std::endl;        

            if (gate->get_type() == CNOT_OPERATION) {
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == CZ_OPERATION) {            
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == CH_OPERATION) {        	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == SYC_OPERATION) {
                std::string error("Gates_block::convert_to_DFE_gates: SYC_gate not implemented");
                throw error;
            }
            else if (gate->get_type() == U3_OPERATION) {

                // definig the U3 parameters
                double varthetaOver2;
                double varphi;
                double varlambda;
		
                // get the inverse parameters of the U3 rotation

                U3* u3_gate = static_cast<U3*>(gate);

                if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_theta_parameter()) {		   
                    DFEgate_kernel_type& DFEGate = DFEgates[gate_set_index*gatesNum + gate_idx];
                    DFEGate.ThetaOver2           = DFEGate.ThetaOver2 + parameter_shift;
                    DFEGate.metadata             = (1<<7); // the leading bit indicates that derivate is processed
                    gate_set_index               = gate_set_index - 1;

                    parameter_idx                = parameter_idx - 1;

                }
                else if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_phi_parameter()) { // not checked
                    DFEgate_kernel_type& DFEGate = DFEgates[gate_set_index*gatesNum + gate_idx];
                    DFEGate.Phi                  = DFEGate.Phi + parameter_shift;
                    DFEGate.metadata             = 3 + (1<<7); // The 0th and 1st element in kernel matrix should be zero for derivates and 3 = 0011, plus the leading bit indicates that derivate is processed
                    gate_set_index               = gate_set_index - 1;

                    parameter_idx                = parameter_idx - 1;
                }
                else if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_lambda_parameter()) { // not checked
                    DFEgate_kernel_type& DFEGate = DFEgates[gate_set_index*gatesNum + gate_idx];
                    DFEGate.Lambda               = DFEGate.Lambda + parameter_shift;
                    DFEGate.metadata             = 5 + (1<<7); // The 0st and 3nd element in kernel matrix should be zero for derivates and 5 = 0101, plus the leading bit indicates that derivate is processed
                    gate_set_index               = gate_set_index - 1;

                    parameter_idx                = parameter_idx - 1;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_theta_parameter() && u3_gate->is_phi_parameter() ) { //not checked

                    DFEgate_kernel_type& DFEGate2= DFEgates[gate_set_index*gatesNum + gate_idx];
                    DFEGate2.Phi                 = DFEGate2.Phi + parameter_shift;
                    DFEGate2.metadata            = 3 + (1<<7); // The 0th and 1st element in kernel matrix should be zero for derivates and 3 = 0011, plus the leading bit indicates that derivate is processed
                    gate_set_index               = gate_set_index - 1;

                    DFEgate_kernel_type& DFEGate = DFEgates[gate_set_index*gatesNum + gate_idx];
                    DFEGate.ThetaOver2           = DFEGate.ThetaOver2 + parameter_shift;
                    DFEGate.metadata             = (1<<7); // the leading bit indicates that derivate is processed
                    gate_set_index               = gate_set_index - 1;



                    parameter_idx = parameter_idx - 2;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_theta_parameter() && u3_gate->is_lambda_parameter() ) {
//////////////////////////////////////////////////////
                    DFEgate_kernel_type& DFEGate2= DFEgates[gate_set_index*gatesNum + gate_idx];
                    DFEGate2.Lambda              = DFEGate2.Lambda + parameter_shift;
                    DFEGate2.metadata            = 5 + (1<<7); // The 0st and 3nd element in kernel matrix should be zero for derivates and 5 = 0101, plus the leading bit indicates that derivate is processed
                    gate_set_index               = gate_set_index - 1;

                    DFEgate_kernel_type& DFEGate = DFEgates[gate_set_index*gatesNum + gate_idx];
                    DFEGate.ThetaOver2           = DFEGate.ThetaOver2 + parameter_shift;
                    DFEGate.metadata             = (1<<7); // the leading bit indicates that derivate is processed
                    gate_set_index               = gate_set_index - 1;



                    parameter_idx = parameter_idx - 2;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_phi_parameter() && u3_gate->is_lambda_parameter() ) { // not checked

                    DFEgate_kernel_type& DFEGate2= DFEgates[gate_set_index*gatesNum + gate_idx];
                    DFEGate2.Lambda              = DFEGate2.Lambda + parameter_shift;
                    DFEGate2.metadata            = 5 + (1<<7); // The 0st and 3nd element in kernel matrix should be zero for derivates and 5 = 0101, plus the leading bit indicates that derivate is processed
                    gate_set_index               = gate_set_index - 1;

                    DFEgate_kernel_type& DFEGate = DFEgates[gate_set_index*gatesNum + gate_idx];
                    DFEGate.Phi                  = DFEGate.Phi + parameter_shift;
                    DFEGate.metadata             = 3 + (1<<7); // The 0th and 1st element in kernel matrix should be zero for derivates and 3 = 0011, plus the leading bit indicates that derivate is processed
                    gate_set_index               = gate_set_index - 1;


                    parameter_idx = parameter_idx - 2;
                }
                else if ((u3_gate->get_parameter_num() == 3)) {

                    DFEgate_kernel_type& DFEGate3= DFEgates[gate_set_index*gatesNum + gate_idx];
                    DFEGate3.Lambda              = DFEGate3.Lambda + parameter_shift;
                    DFEGate3.metadata            = 5 + (1<<7); // The 0st and 3nd element in kernel matrix should be zero for derivates and 5 = 0101, plus the leading bit indicates that derivate is processed
                    gate_set_index               = gate_set_index - 1;

                    DFEgate_kernel_type& DFEGate2= DFEgates[gate_set_index*gatesNum + gate_idx];
                    DFEGate2.Phi                 = DFEGate2.Phi + parameter_shift;
                    DFEGate2.metadata            = 3 + (1<<7); // The 0th and 1st element in kernel matrix should be zero for derivates and 3 = 0011, plus the leading bit indicates that derivate is processed
                    gate_set_index               = gate_set_index - 1;


                    DFEgate_kernel_type& DFEGate = DFEgates[gate_set_index*gatesNum + gate_idx];
                    DFEGate.ThetaOver2           = DFEGate.ThetaOver2 + parameter_shift;
                    DFEGate.metadata             = (1<<7); // the leading bit indicates that derivate is processed
                    gate_set_index               = gate_set_index - 1;

                    parameter_idx = parameter_idx - 3;
                }

                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == RX_OPERATION) { // Did not cehcked

                DFEgate_kernel_type& DFEGate = DFEgates[gate_set_index*gatesNum + gate_idx];
                DFEGate.ThetaOver2           = DFEGate.ThetaOver2 + parameter_shift;
                DFEGate.metadata             = (1<<7); // the leading bit indicates that derivate is processed
                gate_set_index               = gate_set_index - 1;
	
                parameter_idx = parameter_idx - 1;    
	    	 
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == RY_OPERATION) { // Did not cehcked

                DFEgate_kernel_type& DFEGate = DFEgates[gate_set_index*gatesNum + gate_idx];
                DFEGate.ThetaOver2           = DFEGate.ThetaOver2 + parameter_shift;
                DFEGate.metadata             = (1<<7); // the leading bit indicates that derivate is processed
                gate_set_index               = gate_set_index - 1;

                parameter_idx = parameter_idx - 1;

    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == CRY_OPERATION) {

                DFEgate_kernel_type& DFEGate = DFEgates[gate_set_index*gatesNum + gate_idx];
                DFEGate.ThetaOver2           = DFEGate.ThetaOver2 + parameter_shift;
                DFEGate.metadata             = (1<<7); // the leading bit indicates that derivate is processed
                gate_set_index               = gate_set_index - 1;

                parameter_idx = parameter_idx - 1;
	    		                    
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == RZ_OPERATION) { // Did not cehcked

                std::string error("Gates_block::adjust_parameters_for_derivation: RZ gate not implemented for DFE");
                throw error;	    	
            }
            else if (gate->get_type() == RZ_P_OPERATION) { // Did not cehcked

                DFEgate_kernel_type& DFEGate = DFEgates[gate_set_index*gatesNum + gate_idx];
                DFEGate.Phi                  = DFEGate.Phi + parameter_shift;
                DFEGate.metadata             = 3 + (1<<7); // The 0-th and 1st element in kernel matrix should be zero for derivates and 3 = 0011, plus the leading bit indicates that derivate is processed
                gate_set_index               = gate_set_index - 1;

                parameter_idx = parameter_idx - 1;
	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == H_OPERATION) {

            }
            else if (gate->get_type() == X_OPERATION) {

            }
            else if (gate->get_type() == Y_OPERATION) {

            }
            else if (gate->get_type() == Z_OPERATION) {

            }
            else if (gate->get_type() == SX_OPERATION) {
                std::string error("Gates_block::convert_to_DFE_gates: SX_gate not implemented");
                throw error;	    	
            }
            else if (gate->get_type() == BLOCK_OPERATION) {

                Gates_block* block_gate = static_cast<Gates_block*>(gate);
                block_gate->adjust_parameters_for_derivation( DFEgates, gatesNum, gate_idx, gate_set_index);         
                //gate_set_index               = gate_set_index - block_gate->get_parameter_num();

                parameter_idx = parameter_idx - block_gate->get_parameter_num();
            }
            else if (gate->get_type() == UN_OPERATION) {
                std::string error("Gates_block::convert_to_DFE_gates: UN_gate not implemented");
                throw error;
            }
            else if (gate->get_type() == ON_OPERATION) {

                // THE LAST GATE IS A GENERAL GATE APPENDED IN THE BLOCK-WISE OPTIMISATION ROUTINE OF DECOMPOSITION_BASE
                std::string error("Gates_block::convert_to_DFE_gates: ON_gate not implemented");
                throw error;
            }
            else if (gate->get_type() == COMPOSITE_OPERATION) {
                std::string error("Gates_block::convert_to_DFE_gates: Composite_gate not implemented");
                throw error;
            }
            else if (gate->get_type() == GENERAL_OPERATION) {
                std::string error("Gates_block::convert_to_DFE_gates: general_gate not implemented");
                throw error;
            }
            else if (gate->get_type() == ADAPTIVE_OPERATION) {

                DFEgate_kernel_type& DFEGate = DFEgates[gate_set_index*gatesNum + gate_idx];
                DFEGate.metadata              = (1<<7); // the leading bit indicates that derivate is processed
                DFEGate.ThetaOver2           = DFEGate.ThetaOver2 + parameter_shift;
                gate_set_index               = gate_set_index - 1;

                parameter_idx = parameter_idx - 1;
	    		                    
                gate_idx = gate_idx + 1;
            }
            else {
                std::string err("Gates_block::adjust_parameters_for_derivation: unimplemented gate"); 
                throw err;
            }

        }


}


/**
@brief Method to create random initial parameters for the optimization
@return 
*/
DFEgate_kernel_type* 
Gates_block::convert_to_batched_DFE_gates( std::vector<Matrix_real>& parameters_mtx_vec, int& gatesNum, int& gateSetNum, int& redundantGateSets ) {


    gates_num gate_nums   = get_gate_nums();
    int gates_total_num   = gate_nums.total; 
    int chained_gates_num = get_chained_gates_num();
    int gate_padding      = gates_total_num % chained_gates_num == 0 ? 0 : chained_gates_num - (gates_total_num % chained_gates_num);
    gatesNum              = gates_total_num+gate_padding;
/*
std::cout << "chained gates num: " << chained_gates_num << std::endl;
std::cout << "number of gates: " << gatesNum << std::endl;
*/


    gateSetNum = parameters_mtx_vec.size();

#ifdef __MPI__
    int rem = gateSetNum % (4 * world_size );
    if ( rem == 0 ) {
        redundantGateSets = 0;
    }
    else {
        redundantGateSets = (4 * world_size ) - (gateSetNum % (4 * world_size ));
        gateSetNum = gateSetNum + redundantGateSets;
    }
#else
    int rem = gateSetNum % 4;
    if ( rem == 0 ) {
        redundantGateSets = 0;
    }
    else {
        redundantGateSets = 4 - (gateSetNum % 4);
        gateSetNum = gateSetNum + redundantGateSets;
    }
#endif

    DFEgate_kernel_type* DFEgates = new DFEgate_kernel_type[gatesNum*gateSetNum];


    tbb::parallel_for( 0, gateSetNum, 1, [&](int gateset_idx) {
    
        int gate_idx = gateset_idx * gatesNum;

        if ( gateset_idx < parameters_mtx_vec.size() ) {
            Matrix_real& parameters_mtx = parameters_mtx_vec[gateset_idx];
            convert_to_DFE_gates( parameters_mtx, DFEgates, gate_idx );
        }

        // padding with identity gates
        for (int idx=gate_idx; idx<(gateset_idx+1)*gatesNum; idx++ ){

            DFEgate_kernel_type& DFEGate = DFEgates[idx];

        
            DFEGate.target_qbit = 0;
            DFEGate.control_qbit = -1;
            DFEGate.gate_type = U3_OPERATION;
            DFEGate.ThetaOver2 = (int32_t)(0);
            DFEGate.Phi = (int32_t)(0);
            DFEGate.Lambda = (int32_t)(0); 
            DFEGate.metadata = 0;

            gate_idx++;

        }

    });
    

    return DFEgates;

}

/**
@brief Method to create random initial parameters for the optimization
@return 
*/
DFEgate_kernel_type* Gates_block::convert_to_DFE_gates( Matrix_real& parameters_mtx, int& gatesNum ) {

    int parameter_num = get_parameter_num();
    if ( parameter_num != parameters_mtx.size() ) {
        std::string error("Gates_block::convert_to_DFE_gates: wrong number of parameters");
        throw error;
    }


    gates_num gate_nums   = get_gate_nums();
    int gates_total_num   = gate_nums.total; 
    int chained_gates_num = get_chained_gates_num();
    int gate_padding      = chained_gates_num - (gates_total_num % chained_gates_num);
    gatesNum              = gates_total_num+gate_padding;



    DFEgate_kernel_type* DFEgates = new DFEgate_kernel_type[gates_total_num+gate_padding];
    
    int gate_idx = 0;
    convert_to_DFE_gates( parameters_mtx, DFEgates, gate_idx );


    // padding with identity gates
    for (int idx=gate_idx; idx<gatesNum; idx++ ){

        DFEgate_kernel_type& DFEGate = DFEgates[idx];

        
        DFEGate.target_qbit = 0;
        DFEGate.control_qbit = -1;
        DFEGate.gate_type = U3_OPERATION;
        DFEGate.ThetaOver2 = (int32_t)(0);
        DFEGate.Phi = (int32_t)(0);
        DFEGate.Lambda = (int32_t)(0); 
        DFEGate.metadata = 0;

    }
/*
    for ( int idx=0; idx<gatesNum; idx++ ) {

        std::cout << "target qubit: " << (int)DFEgates[idx].target_qbit << " control qubit: " << (int)DFEgates[idx].control_qbit << " gate type: " << (int)DFEgates[idx].gate_type << std::endl; 
    }
*/


    return DFEgates;

}



/**
@brief Method to create random initial parameters for the optimization
@return 
*/
void Gates_block::convert_to_DFE_gates( const Matrix_real& parameters_mtx, DFEgate_kernel_type* DFEgates, int& start_index ) {

   	
        int& gate_idx = start_index;
        int parameter_idx = parameter_num;
	double *parameters_data = parameters_mtx.get_data();
	//const_cast <Matrix_real&>(parameters);


        for(int op_idx = gates.size()-1; op_idx>=0; op_idx--) { 

            Gate* gate = gates[op_idx];
            DFEgate_kernel_type& DFEGate = DFEgates[gate_idx];

            if (gate->get_type() == CNOT_OPERATION) {
                CNOT* cnot_gate = static_cast<CNOT*>(gate);
                DFEGate.target_qbit = cnot_gate->get_target_qbit();
                DFEGate.control_qbit = cnot_gate->get_control_qbit();
                DFEGate.gate_type = CNOT_OPERATION;
                DFEGate.ThetaOver2 = (int32_t)(M_PI/2*(1<<25));
                DFEGate.Phi = (int32_t)(0);
                DFEGate.Lambda = (int32_t)(M_PI*(1<<25)); 
                DFEGate.metadata = 0;
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == CZ_OPERATION) {
                CZ* cz_gate = static_cast<CZ*>(gate);    
                DFEGate.target_qbit = cz_gate->get_target_qbit();
                DFEGate.control_qbit = cz_gate->get_control_qbit();
                DFEGate.gate_type = CZ_OPERATION;
                DFEGate.ThetaOver2 = (int32_t)(0); // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.Phi = (int32_t)(0); // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.Lambda = (int32_t)(M_PI*(1<<25));  // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!	
                DFEGate.metadata = 0;             
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == CH_OPERATION) {
                CH* ch_gate = static_cast<CH*>(gate);   
                DFEGate.target_qbit = ch_gate->get_target_qbit();
                DFEGate.control_qbit = ch_gate->get_control_qbit();
                DFEGate.gate_type = CH_OPERATION;
                DFEGate.ThetaOver2 = (int32_t)(M_PI/4*(1<<25)); // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.Phi = (int32_t)(0); // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.Lambda = (int32_t)(M_PI*(1<<25));  // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!	 
                DFEGate.metadata = 0;        	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == SYC_OPERATION) {
                std::string error("Gates_block::convert_to_DFE_gates: SYC_gate not implemented");
                throw error;
            }
            else if (gate->get_type() == U3_OPERATION) {

                // definig the U3 parameters
                double varthetaOver2;
                double varphi;
                double varlambda;
		
                // get the inverse parameters of the U3 rotation

                U3* u3_gate = static_cast<U3*>(gate);

                if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_theta_parameter()) {
		   
                    varthetaOver2 = std::fmod( parameters_data[parameter_idx-1], 2*M_PI);
                    varphi = 0;
                    varlambda =0;
                    parameter_idx = parameter_idx - 1;

                }
                else if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_phi_parameter()) {
                    varthetaOver2 = 0;
                    varphi = std::fmod( parameters_data[parameter_idx-1], 2*M_PI);
                    varlambda =0;
                    parameter_idx = parameter_idx - 1;
                }
                else if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_lambda_parameter()) {
                    varthetaOver2 = 0;
                    varphi =  0;
                    varlambda = std::fmod( parameters_data[parameter_idx-1], 2*M_PI);
                    parameter_idx = parameter_idx - 1;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_theta_parameter() && u3_gate->is_phi_parameter() ) {
                    varthetaOver2 = std::fmod( parameters_data[parameter_idx-2], 2*M_PI);
                    varphi = std::fmod( parameters_data[parameter_idx-1], 2*M_PI);
                    varlambda = 0;
                    parameter_idx = parameter_idx - 2;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_theta_parameter() && u3_gate->is_lambda_parameter() ) {
                    varthetaOver2 = std::fmod( parameters_data[parameter_idx-2], 2*M_PI);
                    varphi = 0;
                    varlambda = std::fmod( parameters_data[parameter_idx-1], 2*M_PI);
                    parameter_idx = parameter_idx - 2;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_phi_parameter() && u3_gate->is_lambda_parameter() ) {
                    varthetaOver2 = 0;
                    varphi = std::fmod( parameters_data[parameter_idx-2], 2*M_PI);
                    varlambda = std::fmod( parameters_data[parameter_idx-1], 2*M_PI);
                    parameter_idx = parameter_idx - 2;
                }
                else if ((u3_gate->get_parameter_num() == 3)) {
                    varthetaOver2 = std::fmod( parameters_data[parameter_idx-3], 2*M_PI);
                    varphi = std::fmod( parameters_data[parameter_idx-2], 2*M_PI);
                    varlambda = std::fmod( parameters_data[parameter_idx-1], 2*M_PI);
                    parameter_idx = parameter_idx - 3;
                }

                DFEGate.target_qbit = u3_gate->get_target_qbit();
                DFEGate.control_qbit = -1;
                DFEGate.gate_type = U3_OPERATION;
                DFEGate.ThetaOver2 = (int32_t)(varthetaOver2*(1<<25)); 
                DFEGate.Phi = (int32_t)(varphi*(1<<25)); 
                DFEGate.Lambda = (int32_t)(varlambda*(1<<25));	
                DFEGate.metadata = 0;
                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == RX_OPERATION) {
	        // definig the rotation parameter
                double varthetaOver2;
                // get the inverse parameters of the U3 rotation
                RX* rx_gate = static_cast<RX*>(gate);		
                varthetaOver2 = std::fmod( parameters_data[parameter_idx-1], 2*M_PI);
                parameter_idx = parameter_idx - 1;

                DFEGate.target_qbit = rx_gate->get_target_qbit();
                DFEGate.control_qbit = -1;
                DFEGate.gate_type = RX_OPERATION;
                DFEGate.ThetaOver2 = (int32_t)(varthetaOver2*(1<<25)); // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.Phi = (int32_t)(-M_PI/2*(1<<25));  // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.Lambda = (int32_t)(M_PI/2*(1<<25)); // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!	
                DFEGate.metadata = 0;    
	    	 
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == RY_OPERATION) {
                // definig the rotation parameter
                double varthetaOver2;
                // get the inverse parameters of the U3 rotation
                RY* ry_gate = static_cast<RY*>(gate);
                varthetaOver2 = std::fmod( parameters_data[parameter_idx-1], 2*M_PI);
                parameter_idx = parameter_idx - 1;

                DFEGate.target_qbit = ry_gate->get_target_qbit();
                DFEGate.control_qbit = -1;
                DFEGate.gate_type = RY_OPERATION;
                DFEGate.ThetaOver2 = (int32_t)(varthetaOver2*(1<<25)); // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.Phi = (int32_t)(0);  // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.Lambda = (int32_t)(0); // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!	
                DFEGate.metadata = 0;
    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == CRY_OPERATION) {
                // definig the rotation parameter
                double Phi;
                // get the inverse parameters of the U3 rotation
                CRY* cry_gate = static_cast<CRY*>(gate);
                double varthetaOver2 = std::fmod( parameters_data[parameter_idx-1], 2*M_PI);
                parameter_idx = parameter_idx - 1;
                DFEGate.target_qbit = cry_gate->get_target_qbit();
                DFEGate.control_qbit = cry_gate->get_control_qbit();
                DFEGate.gate_type = CRY_OPERATION;
                DFEGate.ThetaOver2 = (int32_t)(varthetaOver2*(1<<25)); // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.Phi = (int32_t)(0);  // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.Lambda = (int32_t)(0); // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.metadata = 0;
	    		                    
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == RZ_OPERATION) { // Did not cehcked

                std::string error("Gates_block::convert_to_DFE_gates: RZ gate not implemented for DFE. Use RZ_P gate instead that differs from RZ gate by a global phase");
                throw error;	    	
            }
            else if (gate->get_type() == RZ_P_OPERATION) {


                // definig the rotation parameter
                double varphi;
                // get the inverse parameters of the U3 rotation
                RZ* rz_gate = static_cast<RZ*>(gate);
                varphi = std::fmod( parameters_data[parameter_idx-1], 2*M_PI);
                parameter_idx = parameter_idx - 1;

                DFEGate.target_qbit = rz_gate->get_target_qbit();
                DFEGate.control_qbit = -1;
                DFEGate.gate_type = RZ_P_OPERATION;
                DFEGate.ThetaOver2 = (int32_t)(0); // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.Phi = (int32_t)(varphi*(1<<25));  // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.Lambda = (int32_t)(0); // TODO: check !!!!!!!!!!!!!!!!!!!!!!!!!!!
                DFEGate.metadata = 0;
	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == H_OPERATION) {
                // get the inverse parameters of the Hadamard rotation
                H* h_gate = static_cast<H*>(gate);

                std::string error("Gates_block::convert_to_DFE_gates: Hadamard gate not implemented");
                throw error;
	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == X_OPERATION) {
                // get the inverse parameters of the U3 rotation
                X* x_gate = static_cast<X*>(gate);

                DFEGate.target_qbit = x_gate->get_target_qbit();
                DFEGate.control_qbit = -1;
                DFEGate.gate_type = X_OPERATION;
                DFEGate.ThetaOver2 = (int32_t)(M_PI/2*(1<<25));
                DFEGate.Phi = (int32_t)(0);
                DFEGate.Lambda = (int32_t)(M_PI*(1<<25)); 
                DFEGate.metadata = 0;
	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == Y_OPERATION) {
                // get the inverse parameters of the U3 rotation
                Y* y_gate = static_cast<Y*>(gate);

                DFEGate.target_qbit = y_gate->get_target_qbit();
                DFEGate.control_qbit = -1;
                DFEGate.gate_type = Y_OPERATION;
                DFEGate.ThetaOver2 = (int32_t)(M_PI/2*(1<<25));
                DFEGate.Phi = (int32_t)(M_PI/2*(1<<25));
                DFEGate.Lambda = (int32_t)(M_PI/2*(1<<25)); 
                DFEGate.metadata = 0;
	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == Z_OPERATION) {
                // get the inverse parameters of the U3 rotation
                Z* z_gate = static_cast<Z*>(gate);

                DFEGate.target_qbit = z_gate->get_target_qbit();
                DFEGate.control_qbit = -1;
                DFEGate.gate_type = Z_OPERATION;
                DFEGate.ThetaOver2 = (int32_t)(0);
                DFEGate.Phi = (int32_t)(0);
                DFEGate.Lambda = (int32_t)(M_PI*(1<<25)); 
                DFEGate.metadata = 0;
	    	
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == SX_OPERATION) {
                // get the inverse parameters of the U3 rotation
                SX* sx_gate = static_cast<SX*>(gate);
                std::string error("Gates_block::convert_to_DFE_gates: SX_gate not implemented");
                throw error;	    	
            }
            else if (gate->get_type() == BLOCK_OPERATION) {
                Gates_block* block_gate = static_cast<Gates_block*>(gate);
                const Matrix_real parameters_layer_mtx(parameters_mtx.get_data() + parameter_idx - gate->get_parameter_num(), 1, gate->get_parameter_num() );
                block_gate->convert_to_DFE_gates( parameters_layer_mtx, DFEgates, gate_idx );
                parameter_idx = parameter_idx - block_gate->get_parameter_num();
            }
            else if (gate->get_type() == UN_OPERATION) {
                std::string error("Gates_block::convert_to_DFE_gates: UN_gate not implemented");
                throw error;
            }
            else if (gate->get_type() == ON_OPERATION) {

                // THE LAST GATE IS A GENERAL GATE APPENDED IN THE BLOCK-WISE OPTIMISATION ROUTINE OF DECOMPOSITION_BASE
                std::string error("Gates_block::convert_to_DFE_gates: ON_gate not implemented");
                throw error;
            }
            else if (gate->get_type() == COMPOSITE_OPERATION) {
                std::string error("Gates_block::convert_to_DFE_gates: Composite_gate not implemented");
                throw error;
            }
            else if (gate->get_type() == GENERAL_OPERATION) {
                std::string error("Gates_block::convert_to_DFE_gates: general_gate not implemented");
                throw error;
            }
            else if (gate->get_type() == ADAPTIVE_OPERATION) {
                // definig the rotation parameter
                double Phi;
                // get the inverse parameters of the U3 rotation
                Adaptive* ad_gate = static_cast<Adaptive*>(gate);
                double varthetaOver2 = std::fmod( activation_function(parameters_data[parameter_idx-1], ad_gate->get_limit()), 2*M_PI);
                parameter_idx = parameter_idx - 1;
                DFEGate.target_qbit = ad_gate->get_target_qbit();
                DFEGate.control_qbit = ad_gate->get_control_qbit();
                DFEGate.gate_type = ADAPTIVE_OPERATION;
                DFEGate.ThetaOver2 = (int32_t)(varthetaOver2*(1<<25)); 
                DFEGate.Phi = (int32_t)(0);  
                DFEGate.Lambda = (int32_t)(0); 
                DFEGate.metadata = 0;
	    		                    
                gate_idx = gate_idx + 1;
            }
            else {
                std::string err("Gates_block::convert_to_DFE_gates: unimplemented gate"); 
                throw err;
            }

        }


    return;

}

void Gates_block::get_matrices_target_control(std::vector<Matrix> &u3_qbit, std::vector<int> &target_qbit, std::vector<int> &control_qbit, Matrix_real& parameters_mtx)
{   u3_qbit.reserve(u3_qbit.capacity() + gates.size());
    target_qbit.reserve(target_qbit.capacity() + gates.size());
    control_qbit.reserve(control_qbit.capacity() + gates.size());
    double* parameters = parameters_mtx.get_data();
    for( int idx=0; idx<gates.size(); idx++) {
        Gate* operation = gates[idx];
        parameters = parameters + operation->get_parameter_num();
        Matrix_real params_mtx(parameters, 1, operation->get_parameter_num());
        switch (operation->get_type()) {
        case CNOT_OPERATION: case CZ_OPERATION:
        case CH_OPERATION: {
            CNOT* cnot_operation = static_cast<CNOT*>(operation);
            u3_qbit.push_back(cnot_operation->calc_one_qubit_u3());
            break;    
        }
        case H_OPERATION: {
            H* h_operation = static_cast<H*>(operation);
            u3_qbit.push_back(h_operation->calc_one_qubit_u3());
            break;
        }
        case X_OPERATION: {
            X* x_operation = static_cast<X*>(operation);
            u3_qbit.push_back(x_operation->calc_one_qubit_u3());
            break;
        }
        case Y_OPERATION: {
            Y* y_operation = static_cast<Y*>(operation);
            u3_qbit.push_back(y_operation->calc_one_qubit_u3());
            break;
        }
        case Z_OPERATION: {
            Z* z_operation = static_cast<Z*>(operation);
            u3_qbit.push_back(z_operation->calc_one_qubit_u3());
            break;
        }
        case SX_OPERATION: {
            SX* sx_operation = static_cast<SX*>(operation);
            u3_qbit.push_back(sx_operation->calc_one_qubit_u3());
            break;
        }
        case U3_OPERATION: {
            U3* u3_operation = static_cast<U3*>(operation);
            u3_qbit.push_back(u3_operation->calc_one_qubit_u3(params_mtx[0], params_mtx[1], params_mtx[2]));
            break;
        }
        case RX_OPERATION: {
            RX* rx_operation = static_cast<RX*>(operation);
            double ThetaOver2, Phi, Lambda;
            ThetaOver2 = params_mtx[0];
            rx_operation->parameters_for_calc_one_qubit(ThetaOver2, Phi, Lambda);
            u3_qbit.push_back(rx_operation->calc_one_qubit_u3(ThetaOver2, Phi, Lambda));
            break;
        }
        case RY_OPERATION: {
            RY* ry_operation = static_cast<RY*>(operation);
            double ThetaOver2, Phi, Lambda;
            ThetaOver2 = params_mtx[0];
            ry_operation->parameters_for_calc_one_qubit(ThetaOver2, Phi, Lambda);
            u3_qbit.push_back(ry_operation->calc_one_qubit_u3(ThetaOver2, Phi, Lambda));
            break;
        }
        case CRY_OPERATION: {
            CRY* cry_operation = static_cast<CRY*>(operation);
            double ThetaOver2, Phi, Lambda;
            ThetaOver2 = params_mtx[0];
            cry_operation->parameters_for_calc_one_qubit(ThetaOver2, Phi, Lambda);
            u3_qbit.push_back(cry_operation->calc_one_qubit_u3(ThetaOver2, Phi, Lambda));
            break;
        }
        case RZ_OPERATION: {
            RZ* rz_operation = static_cast<RZ*>(operation);
            u3_qbit.push_back(rz_operation->calc_one_qubit_u3(params_mtx[0]));
            break;
        }
        case RZ_P_OPERATION: {
            RZ_P* rz_p_operation = static_cast<RZ_P*>(operation);
            double ThetaOver2, Phi, Lambda;
            Phi = params_mtx[0];
            rz_p_operation->parameters_for_calc_one_qubit(ThetaOver2, Phi, Lambda);
            u3_qbit.push_back(rz_p_operation->calc_one_qubit_u3(ThetaOver2, Phi, Lambda));
            break;
        }
        case BLOCK_OPERATION: {
            Gates_block* block_operation = static_cast<Gates_block*>(operation);
            block_operation->get_matrices_target_control(u3_qbit, target_qbit, control_qbit, params_mtx);
            continue;
        }
        //case ADAPTIVE_OPERATION:
        //case SYC_OPERATION:
        //case UN_OPERATION:
        //case ON_OPERATION:
        //case COMPOSITE_OPERATION:
        //case GENERAL_OPERATION:
        default:
            std::string err("Optimization_Interface::apply_to: unimplemented gate (" + std::to_string(operation->get_type()) + ")"); 
            throw err;
        }
        target_qbit.push_back(operation->get_target_qbit());
        control_qbit.push_back(operation->get_control_qbit());
    }    
}
#endif



/**
@brief ?????????
@return Return with ?????????
*/
void 
export_gate_list_to_binary(Matrix_real& parameters, Gates_block* gates_block, const std::string& filename, int verbosity) {

    std::stringstream sstream;
    sstream << "Exporting circuit into binary format. Filename: " << filename << std::endl;
    logging log;
    log.verbose = verbosity;
    log.print(sstream, 3);	

    FILE* pFile;
    const char* c_filename = filename.c_str();
    
    pFile = fopen(c_filename, "wb");
    if (pFile==NULL) {fputs ("File error",stderr); exit (1);}

    export_gate_list_to_binary( parameters, gates_block, pFile, verbosity );

    fclose(pFile);
    return;

}



/**
@brief ?????????
@return Return with ?????????
*/
void 
export_gate_list_to_binary(Matrix_real& parameters, Gates_block* gates_block, FILE* pFile, int verbosity) {

    int qbit_num = gates_block->get_qbit_num();   
    fwrite(&qbit_num, sizeof(int), 1, pFile);

    int parameter_num = gates_block->get_parameter_num();
    fwrite(&parameter_num, sizeof(int), 1, pFile);


    int gates_num = gates_block->get_gate_num();
    fwrite(&gates_num, sizeof(int), 1, pFile);



    std::vector<Gate*> gates = gates_block->get_gates();
    double* parameters_data = parameters.get_data();

    for ( std::vector<Gate*>::iterator it=gates.begin(); it != gates.end(); ++it ) {
        Gate* op = *it;

        gate_type gt_type = op->get_type();

        fwrite(&gt_type, sizeof(gate_type), 1, pFile);

        int parameter_num = op->get_parameter_num();

        if (gt_type == CNOT_OPERATION || gt_type == CZ_OPERATION || gt_type == CH_OPERATION || gt_type == SYC_OPERATION) {
            int target_qbit = op->get_target_qbit();
            int control_qbit = op->get_control_qbit();
            fwrite(&target_qbit, sizeof(int), 1, pFile);
            fwrite(&control_qbit, sizeof(int), 1, pFile);
        }
        else if (gt_type == U3_OPERATION) {
            int target_qbit = op->get_target_qbit();
            fwrite(&target_qbit, sizeof(int), 1, pFile);

            U3* u3_op = static_cast<U3*>( op );

            int theta_bool  = u3_op->is_theta_parameter();
            int phi_bool    = u3_op->is_phi_parameter();
            int lambda_bool = u3_op->is_lambda_parameter();

            fwrite(&theta_bool, sizeof(int), 1, pFile);
            fwrite(&phi_bool, sizeof(int), 1, pFile);
            fwrite(&lambda_bool, sizeof(int), 1, pFile);


            fwrite(parameters_data, sizeof(double), parameter_num, pFile);

            
        }
        else if (gt_type == RX_OPERATION || gt_type == RY_OPERATION || gt_type == RZ_OPERATION ) {
            int target_qbit = op->get_target_qbit();
            fwrite(&target_qbit, sizeof(int), 1, pFile);

            fwrite(parameters_data, sizeof(double), parameter_num, pFile);
        }
        else if (gt_type == CRY_OPERATION) {
            int target_qbit = op->get_target_qbit();
            int control_qbit = op->get_control_qbit();
            fwrite(&target_qbit, sizeof(int), 1, pFile);
            fwrite(&control_qbit, sizeof(int), 1, pFile);

            fwrite(parameters_data, sizeof(double), parameter_num, pFile);
        }
        
        else if (gt_type == X_OPERATION || gt_type == Y_OPERATION || gt_type == Z_OPERATION || gt_type == SX_OPERATION || gt_type == H_OPERATION) {
            int target_qbit = op->get_target_qbit();
            fwrite(&target_qbit, sizeof(int), 1, pFile);
        }
        else if (gt_type == BLOCK_OPERATION) {
            Gates_block* block_op = static_cast<Gates_block*>( op );
            Matrix_real parameters_loc(parameters_data, 1, parameter_num);
            export_gate_list_to_binary( parameters_loc, block_op, pFile );

        }
        else if (gt_type == ADAPTIVE_OPERATION) {
            int target_qbit = op->get_target_qbit();
            int control_qbit = op->get_control_qbit();
            fwrite(&target_qbit, sizeof(int), 1, pFile);
            fwrite(&control_qbit, sizeof(int), 1, pFile);

            fwrite(parameters_data, sizeof(double), parameter_num, pFile);
        }
        else {
            std::string err("export_gate_list_to_binary: unimplemented gate"); 
            throw err;
        }


        parameters_data = parameters_data + parameter_num;

    }

}


/**
@brief ?????????
@return Return with ?????????
*/
Gates_block* import_gate_list_from_binary(Matrix_real& parameters, const std::string& filename, int verbosity) {

    std::stringstream sstream;
    sstream << "Importing quantum circuit from binary file " << filename << std::endl;	
    logging log;
    log.verbose = verbosity;
    log.print(sstream, 2);	

    FILE* pFile;
    const char* c_filename = filename.c_str();
    
    pFile = fopen(c_filename, "rb");
    if (pFile==NULL) {fputs ("File error",stderr); exit (1);}

    Gates_block* ret = import_gate_list_from_binary(parameters, pFile, verbosity);

    fclose(pFile);
    return ret;
}

/**
@brief ?????????
@return Return with ?????????
*/
Gates_block* import_gate_list_from_binary(Matrix_real& parameters, FILE* pFile, int verbosity) {

    std::stringstream sstream;

    int qbit_num;
    size_t fread_status;

    fread_status = fread(&qbit_num, sizeof(int), 1, pFile);
    sstream << "qbit_num: " << qbit_num << std::endl;
    Gates_block* gate_block = new Gates_block(qbit_num);

    int parameter_num;
    fread_status = fread(&parameter_num, sizeof(int), 1, pFile);
    sstream << "parameter_num: " << parameter_num << std::endl;
    parameters = Matrix_real(1, parameter_num);
    double* parameters_data = parameters.get_data();

    int gates_num;
    fread_status = fread(&gates_num, sizeof(int), 1, pFile);
    sstream << "gates_num: " << gates_num << std::endl;

    std::vector<int> gate_block_level_gates_num;
    std::vector<Gates_block*> gate_block_levels;
    gate_block_level_gates_num.push_back( gates_num );
    gate_block_levels.push_back(gate_block);
    int current_level = 0;

    

    int iter_max = 1e5;
    int iter = 0;
    while ( gate_block_level_gates_num[0] > 0 && iter < iter_max) {

        gate_type gt_type;
        fread_status = fread(&gt_type, sizeof(gate_type), 1, pFile);

        //std::cout << "gate type: " << gt_type << std::endl;

        if (gt_type == CNOT_OPERATION) {
            sstream << "importing CNOT gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            int control_qbit;
            fread_status = fread(&control_qbit, sizeof(int), 1, pFile);
            sstream << "control_qbit: " << control_qbit << std::endl;

            gate_block_levels[current_level]->add_cnot(target_qbit, control_qbit);
            gate_block_level_gates_num[current_level]--;
        }
        else if (gt_type == CZ_OPERATION) {
            sstream << "importing CZ gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            int control_qbit;
            fread_status = fread(&control_qbit, sizeof(int), 1, pFile);
            sstream << "control_qbit: " << control_qbit << std::endl;

            gate_block_levels[current_level]->add_cz(target_qbit, control_qbit);
            gate_block_level_gates_num[current_level]--;
        }
        else if (gt_type == CH_OPERATION) {
            sstream << "importing CH gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            int control_qbit;
            fread_status = fread(&control_qbit, sizeof(int), 1, pFile);
            sstream << "control_qbit: " << control_qbit << std::endl;

            gate_block_levels[current_level]->add_ch(target_qbit, control_qbit);
            gate_block_level_gates_num[current_level]--;
        }
        else if (gt_type == SYC_OPERATION) {
            sstream << "importing SYCAMORE gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            int control_qbit;
            fread_status = fread(&control_qbit, sizeof(int), 1, pFile);
            sstream << "control_qbit: " << control_qbit << std::endl;

            gate_block_levels[current_level]->add_syc(target_qbit, control_qbit);
            gate_block_level_gates_num[current_level]--;
        }
        else if (gt_type == U3_OPERATION) {
            sstream << "importing U3 gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            int Theta;
            int Phi;
            int Lambda;

            fread_status = fread(&Theta, sizeof(int), 1, pFile);
            fread_status = fread(&Phi, sizeof(int), 1, pFile);
            fread_status = fread(&Lambda, sizeof(int), 1, pFile);

            int parameter_num = Theta + Phi + Lambda;
            fread_status = fread(parameters_data, sizeof(double), parameter_num, pFile);
            parameters_data = parameters_data + parameter_num;

            gate_block_levels[current_level]->add_u3(target_qbit, Theta, Phi, Lambda);
            gate_block_level_gates_num[current_level]--;
            
        }
        else if (gt_type == RX_OPERATION) {

            sstream << "importing RX gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            fread_status = fread(parameters_data, sizeof(double), 1, pFile);
            parameters_data++;

            gate_block_levels[current_level]->add_rx(target_qbit);
            gate_block_level_gates_num[current_level]--;

        }
        else if (gt_type == RY_OPERATION) {

            sstream << "importing RY gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            fread_status = fread(parameters_data, sizeof(double), 1, pFile);
            parameters_data++;

            gate_block_levels[current_level]->add_ry(target_qbit);
            gate_block_level_gates_num[current_level]--;

        }
        else if (gt_type == CRY_OPERATION) {

            sstream << "importing CRY gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            int control_qbit;
            fread_status = fread(&control_qbit, sizeof(int), 1, pFile);
            sstream << "control_qbit: " << control_qbit << std::endl;

            fread_status = fread(parameters_data, sizeof(double), 1, pFile);
            parameters_data++;

            gate_block_levels[current_level]->add_cry(target_qbit, control_qbit);
            gate_block_level_gates_num[current_level]--;

        }
        else if (gt_type == RZ_OPERATION) {

            sstream << "importing RZ gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            fread_status = fread(parameters_data, sizeof(double), 1, pFile);
            parameters_data++;

            gate_block_levels[current_level]->add_rz(target_qbit);
            gate_block_level_gates_num[current_level]--;

        }
        else if (gt_type == H_OPERATION) {

            sstream << "importing Hadamard gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            gate_block_levels[current_level]->add_h(target_qbit);
            gate_block_level_gates_num[current_level]--;

        }
        else if (gt_type == X_OPERATION) {

            sstream << "importing X gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            gate_block_levels[current_level]->add_x(target_qbit);
            gate_block_level_gates_num[current_level]--;

        }
        else if (gt_type == Y_OPERATION) {

            sstream << "importing Y gate" << std::endl;

            int target_qbit;
            fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            gate_block_levels[current_level]->add_y(target_qbit);
            gate_block_level_gates_num[current_level]--;

        }
        else if (gt_type == Z_OPERATION) {

            sstream << "importing Z gate" << std::endl;

            int target_qbit;
            fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            gate_block_levels[current_level]->add_z(target_qbit);
            gate_block_level_gates_num[current_level]--;

        }
        else if (gt_type == SX_OPERATION) {

            sstream << "importing SX gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            gate_block_levels[current_level]->add_sx(target_qbit);
            gate_block_level_gates_num[current_level]--;

        }
        else if (gt_type == BLOCK_OPERATION) {


            sstream << "******* importing gates block ********" << std::endl;

            int qbit_num_loc;
            fread_status = fread(&qbit_num_loc, sizeof(int), 1, pFile);
            //std::cout << "qbit_num_loc: " << qbit_num_loc << std::endl;
            Gates_block* gate_block_inner = new Gates_block(qbit_num_loc);

            int parameter_num_loc;
            fread_status = fread(&parameter_num_loc, sizeof(int), 1, pFile);
            //std::cout << "parameter_num_loc: " << parameter_num_loc << std::endl;
        

            int gates_num_loc;
            fread_status = fread(&gates_num_loc, sizeof(int), 1, pFile);
            //std::cout << "gates_num_loc: " << gates_num_loc << std::endl;
                        
            gate_block_levels.push_back( gate_block_inner );
            gate_block_level_gates_num.push_back(gates_num_loc);
            current_level++;
        }
        else if (gt_type == ADAPTIVE_OPERATION) {

            sstream << "importing adaptive gate" << std::endl;

            int target_qbit;
            fread_status = fread(&target_qbit, sizeof(int), 1, pFile);
            sstream << "target_qbit: " << target_qbit << std::endl;

            int control_qbit;
            fread_status = fread(&control_qbit, sizeof(int), 1, pFile);
            sstream << "control_qbit: " << control_qbit << std::endl;

            fread_status = fread(parameters_data, sizeof(double), 1, pFile);
            parameters_data++;

            gate_block_levels[current_level]->add_adaptive(target_qbit, control_qbit);
            gate_block_level_gates_num[current_level]--;

        }
        else {
            std::string err("import_gate_list_from_binary: unimplemented gate"); 
            throw err;
        }


        if ( gate_block_level_gates_num[current_level] == 0 ) {
            gate_block_levels[ current_level-1 ]->add_gate( static_cast<Gate*>(gate_block_levels[ current_level ]) );
            gate_block_levels.pop_back();
            gate_block_level_gates_num.pop_back();
            current_level--;
            gate_block_level_gates_num[current_level]--;
            sstream << "finishing gates block" << std::endl;
        }


        iter++;
    }

    logging log;
    log.verbose = verbosity;
    log.print(sstream, 4);	

  
    if ( iter == iter_max ) {
        std::string error("Corrupted input file, reached end of the file before contructing the whole gate structure");
        throw error;
    }

    return gate_block;

}



/**
@brief Call to reverse the order of the parameters in an array
@param parameters_in The real input vector.
@param gates_it Iterator over the gates. (does not get reversed)
@return Return with the reversed array
*/
Matrix_real reverse_parameters( const Matrix_real& parameters_in, std::vector<Gate*>::iterator gates_it, int num_of_gates ) {

	if ( parameters_in.cols > 1 && parameters_in.rows > 1 ) {
		std::string error("reverse_parameters: Input array should have a single column or a single row.");
		throw error;
	}
	
//return parameters_in.copy();
	
    // determine the number of parameters
    int parameters_num_total = 0;
    for (int idx=0; idx<num_of_gates; idx++) {

        // The current gate
        Gate* gate = *(gates_it++);
        parameters_num_total = parameters_num_total + gate->get_parameter_num();

    }
    
    
	if ( parameters_num_total  == 0) {
		return Matrix_real(0,0);
	}	    
	
	//std::cout << "uu" << std::endl;
	//parameters_in.print_matrix();
    

	Matrix_real parameters_ret(1, parameters_num_total);
	int parameter_num_copied = 0;

    // apply the gate operations on the inital matrix
    for (int idx=num_of_gates-0; idx>0; idx--) {

        // The current gate
        Gate* gate = *(--gates_it);
        
        int parameter_num_gate = gate->get_parameter_num();
        
        if ( parameter_num_gate == 0 ) {
        	continue;
        }     
           
        else if (gate->get_type() == BLOCK_OPERATION ) {
        
	        //std::cout << "block: " << parameter_num_gate << " " << parameters_num_total << std::endl;
       	    parameters_num_total = parameters_num_total - gate->get_parameter_num();
	                
        	Matrix_real parameters_of_block( parameters_in.get_data()+parameters_num_total, 1, parameter_num_gate );
        	//parameters_of_block.print_matrix();
        	
            Gates_block* block_gate = static_cast<Gates_block*>( gate );
            
            std::vector<Gate*> gates_loc = block_gate->get_gates();
            
            Matrix_real parameters_of_block_reversed = reverse_parameters( parameters_of_block, gates_loc.begin(), gates_loc.size() );
            
            //parameters_of_block_reversed.print_matrix();
            
			memcpy( parameters_ret.get_data()+parameter_num_copied, parameters_of_block_reversed.get_data(), parameters_of_block_reversed.size()*sizeof(double) );
			parameter_num_copied = parameter_num_copied + parameters_of_block_reversed.size();
			
			//parameters_ret.print_matrix();
			
        }
        
        else {
        
	        //std::cout << parameter_num_gate << std::endl;
        
    	    parameters_num_total = parameters_num_total - gate->get_parameter_num();
			memcpy( parameters_ret.get_data()+parameter_num_copied, parameters_in.get_data()+parameters_num_total, gate->get_parameter_num()*sizeof(double) );
			parameter_num_copied = parameter_num_copied + gate->get_parameter_num();
			
		}


    }



	
	return parameters_ret;
	
	


}




/**
@brief Call to inverse-reverse the order of the parameters in an array
@param parameters_in The real input vector.
@param gates_it Iterator over the gates. (does not get reversed)
@return Return with the reversed array
*/
Matrix_real inverse_reverse_parameters( const Matrix_real& parameters_in, std::vector<Gate*>::iterator gates_it, int num_of_gates ) {

	if ( parameters_in.cols > 1 && parameters_in.rows > 1 ) {
		std::string error("reverse_parameters: Input array should have a single column or a single row.");
		throw error;
	}
	
	
    // determine the number of parameters
    int parameters_num_total = 0;
    for (int idx=0; idx<num_of_gates; idx++) {

        // The current gate
        Gate* gate = *(gates_it++);
        parameters_num_total = parameters_num_total + gate->get_parameter_num();

    }
    
    
	if ( parameters_num_total  == 0) {
		return Matrix_real(0,0);
	}	    
	
	//std::cout << "uu" << std::endl;
	//parameters_in.print_matrix();
    

	Matrix_real parameters_ret(1, parameters_num_total);
	int parameter_num_copied = 0;

    // apply the gate operations on the inital matrix
    for (int idx=num_of_gates-0; idx>0; idx--) {

        // The current gate
        Gate* gate = *(--gates_it);
        
        int parameter_num_gate = gate->get_parameter_num();
        
        if ( parameter_num_gate == 0 ) {
        	continue;
        }     
         
        else if (gate->get_type() == BLOCK_OPERATION ) {
        
	        //std::cout << "block: " << parameter_num_gate << " " << parameters_num_total << std::endl;
       	    parameters_num_total = parameters_num_total - gate->get_parameter_num();
	                
        	Matrix_real parameters_of_block( parameters_in.get_data()+parameters_num_total, 1, parameter_num_gate );
        	//parameters_of_block.print_matrix();
        	
            Gates_block* block_gate = static_cast<Gates_block*>( gate );
            
            std::vector<Gate*> gates_loc = block_gate->get_gates();
            
            Matrix_real parameters_of_block_reversed = reverse_parameters( parameters_of_block, gates_loc.begin(), gates_loc.size() );
            
            //parameters_of_block_reversed.print_matrix();
            
			memcpy( parameters_ret.get_data()+parameter_num_copied, parameters_of_block_reversed.get_data(), parameters_of_block_reversed.size()*sizeof(double) );
			parameter_num_copied = parameter_num_copied + parameters_of_block_reversed.size();
			
			//parameters_ret.print_matrix();
			
        }
        
        else {
        
	        //std::cout << parameter_num_gate << std::endl;
        
    	    parameters_num_total = parameters_num_total - gate->get_parameter_num();
            memcpy( parameters_ret.get_data()+parameters_num_total, parameters_in.get_data()+parameter_num_copied, gate->get_parameter_num()*sizeof(double) );
            parameter_num_copied = parameter_num_copied + gate->get_parameter_num();
			
		}


    }



	
	return parameters_ret;
	
	


}

