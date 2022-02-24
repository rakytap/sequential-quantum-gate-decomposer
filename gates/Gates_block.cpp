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
/*! \file Gates_block.cpp
    \brief Class responsible for grouping two-qubit (CNOT,CZ,CH) and one-qubit gates into layers
*/

#include "CZ.h"
#include "CH.h"
#include "CNOT.h"
#include "U3.h"
#include "RX.h"
#include "RY.h"
#include "CRY.h"
#include "RZ.h"
#include "X.h"
#include "SX.h"
#include "SYC.h"
#include "UN.h"
#include "ON.h"
#include "Adaptive.h"
#include "Composite.h"
#include "Gates_block.h"

//The stringstream input to store the output messages.
std::stringstream sstream;

//Integer value to set the verbosity level of the output messages.
int verbose_level;

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

        Gate* operation = *it;

        if (operation->get_type() == CNOT_OPERATION) {
            CNOT* cnot_operation = static_cast<CNOT*>(operation);
            delete cnot_operation;
        }
        else if (operation->get_type() == CZ_OPERATION) {
            CZ* cz_operation = static_cast<CZ*>(operation);
            delete cz_operation;
        }
        else if (operation->get_type() == CH_OPERATION) {
            CH* ch_operation = static_cast<CH*>(operation);
            delete ch_operation;
        }
        else if (operation->get_type() == SYC_OPERATION) {
            SYC* syc_operation = static_cast<SYC*>(operation);
            delete syc_operation;
        }
        else if (operation->get_type() == U3_OPERATION) {

            U3* u3_operation = static_cast<U3*>(operation);
            delete u3_operation;

        }
        else if (operation->get_type() == RY_OPERATION) {

            RY* ry_operation = static_cast<RY*>(operation);
            delete ry_operation;

        }
        else if (operation->get_type() == CRY_OPERATION) {

            CRY* cry_operation = static_cast<CRY*>(operation);
            delete cry_operation;

        }
        else if (operation->get_type() == RX_OPERATION) {

            RX* rx_operation = static_cast<RX*>(operation);
            delete rx_operation;

        }
        else if (operation->get_type() == RZ_OPERATION) {

            RZ* rz_operation = static_cast<RZ*>(operation);
            delete rz_operation;

        }
        else if (operation->get_type() == X_OPERATION) {

            X* x_operation = static_cast<X*>(operation);
            delete x_operation;

        }
        else if (operation->get_type() == SX_OPERATION) {

            SX* sx_operation = static_cast<SX*>(operation);            
            delete sx_operation;

        }
        else if (operation->get_type() == BLOCK_OPERATION) {

            Gates_block* block_operation = static_cast<Gates_block*>(operation);
            delete block_operation;

        }
        else if (operation->get_type() == GENERAL_OPERATION) {
            delete operation;
        }
        else if (operation->get_type() == UN_OPERATION) {
            UN* un_operation = static_cast<UN*>(operation);
            delete un_operation;
        }
        else if (operation->get_type() == ON_OPERATION) {
            ON* on_operation = static_cast<ON*>(operation);
            delete on_operation;
        }
        else if (operation->get_type() == COMPOSITE_OPERATION) {
            Composite* com_operation = static_cast<Composite*>(operation);
            delete com_operation;
        }
        else if (operation->get_type() == ADAPTIVE_OPERATION) {

            Adaptive* ad_operation = static_cast<Adaptive*>(operation);
            delete ad_operation;

        }
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

    if ( idx>= gates.size() ) return;

    // fist decrese the number of parameters
    Gate* gate = gates[idx];
    parameter_num -= gate->get_parameter_num();

    gates.erase( gates.begin() + idx );

}

/**
@brief Call to retrieve the operation matrix (Which is the product of all the operation matrices stored in the operation block)
@param parameters An array pointing to the parameters of the gates
@return Returns with the operation matrix
*/
Matrix
Gates_block::get_matrix( Matrix_real& parameters ) {

    // create matrix representation of the gate operations
    Matrix block_mtx = create_identity(matrix_size);
    apply_to(parameters, block_mtx);

#ifdef DEBUG
    if (block_mtx.isnan()) {
	sstream << "Gates_block::get_matrix: block_mtx contains NaN." << std::endl;
	verbose_level=1;
        print(sstream,verbose_level);	
	
        
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
Gates_block::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix> input ) {


    for ( std::vector<Matrix>::iterator it=input.begin(); it != input.end(); it++ ) {
        apply_to( parameters_mtx, *it );
    }

}


/**
@brief Call to apply the gate on the input array/matrix Gates_block*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
Gates_block::apply_to( Matrix_real& parameters_mtx, Matrix& input ) {

    double* parameters = parameters_mtx.get_data();

    parameters = parameters + parameter_num;

    for( int idx=gates.size()-1; idx>=0; idx--) {

        Gate* operation = gates[idx];
        parameters = parameters - operation->get_parameter_num();
        Matrix_real parameters_mtx(parameters, 1, operation->get_parameter_num());

        if (operation->get_type() == CNOT_OPERATION) {
            CNOT* cnot_operation = static_cast<CNOT*>(operation);
            cnot_operation->apply_to(input);
        }
        else if (operation->get_type() == CZ_OPERATION) {
            CZ* cz_operation = static_cast<CZ*>(operation);
            cz_operation->apply_to(input);
        }
        else if (operation->get_type() == CH_OPERATION) {
            CH* ch_operation = static_cast<CH*>(operation);
            ch_operation->apply_to(input);
        }
        else if (operation->get_type() == SYC_OPERATION) {
            SYC* syc_operation = static_cast<SYC*>(operation);
            syc_operation->apply_to(input);
        }
        else if (operation->get_type() == U3_OPERATION) {
            U3* u3_operation = static_cast<U3*>(operation);
            u3_operation->apply_to( parameters_mtx, input );    
        }
        else if (operation->get_type() == RX_OPERATION) {
            RX* rx_operation = static_cast<RX*>(operation);
            rx_operation->apply_to( parameters_mtx, input ); 
        }
        else if (operation->get_type() == RY_OPERATION) {
            RY* ry_operation = static_cast<RY*>(operation);
            ry_operation->apply_to( parameters_mtx, input ); 
        }
        else if (operation->get_type() == CRY_OPERATION) {
            CRY* cry_operation = static_cast<CRY*>(operation);
            cry_operation->apply_to( parameters_mtx, input ); 
        }
        else if (operation->get_type() == RZ_OPERATION) {
            RZ* rz_operation = static_cast<RZ*>(operation);
            rz_operation->apply_to( parameters_mtx, input ); 
        }
        else if (operation->get_type() == X_OPERATION) {
            X* x_operation = static_cast<X*>(operation);
            x_operation->apply_to( input ); 
        }
        else if (operation->get_type() == SX_OPERATION) {
            SX* sx_operation = static_cast<SX*>(operation);
            sx_operation->apply_to( input ); 
        }
        else if (operation->get_type() == GENERAL_OPERATION) {
            operation->apply_to(input);
        }
        else if (operation->get_type() == UN_OPERATION) {
            UN* un_operation = static_cast<UN*>(operation);
            un_operation->apply_to(parameters_mtx, input);
        }
        else if (operation->get_type() == ON_OPERATION) {
            ON* on_operation = static_cast<ON*>(operation);
            on_operation->apply_to(parameters_mtx, input);
        }
        else if (operation->get_type() == BLOCK_OPERATION) {
            Gates_block* block_operation = static_cast<Gates_block*>(operation);
            block_operation->apply_to(parameters_mtx, input);
        }
        else if (operation->get_type() == COMPOSITE_OPERATION) {
            Composite* com_operation = static_cast<Composite*>(operation);
            com_operation->apply_to(parameters_mtx, input);
        }
        else if (operation->get_type() == ADAPTIVE_OPERATION) {
            Adaptive* ad_operation = static_cast<Adaptive*>(operation);
            ad_operation->apply_to( parameters_mtx, input ); 
        }

#ifdef DEBUG
        if (input.isnan()) {
	    sstream << "Gates_block::apply_to: transformed matrix contains NaN." << std::endl;
	    verbose_level=1;
            print(sstream,verbose_level);	
	    
            
        }
#endif


    }



}


/**
@brief Call to apply the gate on the input array/matrix by input*Gate_block
@param input The input array on which the gate is applied
*/
void 
Gates_block::apply_from_right( Matrix_real& parameters_mtx, Matrix& input ) {

    double* parameters = parameters_mtx.get_data();

    for( int idx=0; idx<gates.size(); idx++) {

        Gate* operation = gates[idx];
        Matrix_real parameters_mtx(parameters, 1, operation->get_parameter_num());

        if (operation->get_type() == CNOT_OPERATION) {
            CNOT* cnot_operation = static_cast<CNOT*>(operation);
            cnot_operation->apply_from_right(input);
        }
        else if (operation->get_type() == CZ_OPERATION) {
            CZ* cz_operation = static_cast<CZ*>(operation);
            cz_operation->apply_from_right(input);
        }
        else if (operation->get_type() == CH_OPERATION) {
            CH* ch_operation = static_cast<CH*>(operation);
            ch_operation->apply_from_right(input);
        }
        else if (operation->get_type() == SYC_OPERATION) {
            SYC* syc_operation = static_cast<SYC*>(operation);
            syc_operation->apply_from_right(input);
        }
        else if (operation->get_type() == U3_OPERATION) {
            U3* u3_operation = static_cast<U3*>(operation);
            u3_operation->apply_from_right( parameters_mtx, input ); 
        }
        else if (operation->get_type() == RX_OPERATION) {
            RX* rx_operation = static_cast<RX*>(operation);
            rx_operation->apply_from_right( parameters_mtx, input ); 
        }
        else if (operation->get_type() == RY_OPERATION) {
            RY* ry_operation = static_cast<RY*>(operation);
            ry_operation->apply_from_right( parameters_mtx, input ); 
        }
        else if (operation->get_type() == CRY_OPERATION) {
            CRY* cry_operation = static_cast<CRY*>(operation);
            cry_operation->apply_from_right( parameters_mtx, input ); 
        }
        else if (operation->get_type() == RZ_OPERATION) {
            RZ* rz_operation = static_cast<RZ*>(operation);
            rz_operation->apply_from_right( parameters_mtx, input );         
        }
        else if (operation->get_type() == X_OPERATION) {
            X* x_operation = static_cast<X*>(operation);
            x_operation->apply_from_right( input ); 
        }
        else if (operation->get_type() == SX_OPERATION) {
            SX* sx_operation = static_cast<SX*>(operation);
            sx_operation->apply_from_right( input ); 
        }
        else if (operation->get_type() == GENERAL_OPERATION) {
            operation->apply_from_right(input);
        }
        else if (operation->get_type() == UN_OPERATION) {
            UN* un_operation = static_cast<UN*>(operation);
            un_operation->apply_from_right( parameters_mtx, input ); 
        }
        else if (operation->get_type() == ON_OPERATION) {
            ON* on_operation = static_cast<ON*>(operation);
            on_operation->apply_from_right( parameters_mtx, input ); 
        }
        else if (operation->get_type() == BLOCK_OPERATION) {
            Gates_block* block_operation = static_cast<Gates_block*>(operation);
            block_operation->apply_from_right(parameters_mtx, input);
        }
        else if (operation->get_type() == COMPOSITE_OPERATION) {
            Composite* com_operation = static_cast<Composite*>(operation);
            com_operation->apply_from_right( parameters_mtx, input ); 
        }
        else if (operation->get_type() == ADAPTIVE_OPERATION) {
            Adaptive* ad_operation = static_cast<Adaptive*>(operation);
            ad_operation->apply_from_right( parameters_mtx, input ); 
        }

        parameters = parameters + operation->get_parameter_num();

#ifdef DEBUG
        if (input.isnan()) {
	    sstream << "Gates_block::apply_from_right: transformed matrix contains NaN." << std::endl;
	    verbose_level=1;
            print(sstream,verbose_level);	
	    
            
        }
#endif


    }


}



/**
@brief ???????????????
*/
std::vector<Matrix> 
Gates_block::apply_derivate_to( Matrix_real& parameters_mtx_in, Matrix& input ) {

  
    std::vector<Matrix> grad(parameter_num, Matrix(0,0));

    // deriv_idx ... the index of the gate block for which the gradient is to be calculated
    tbb::parallel_for( tbb::blocked_range<int>(0,gates.size()), [&](tbb::blocked_range<int> r) {
        for (int deriv_idx=r.begin(); deriv_idx<r.end(); ++deriv_idx) { 


            Gate* gate_deriv = gates[deriv_idx];

            // for constant gate no gardient component is calculated
            if ( gate_deriv->get_parameter_num() == 0 ) {
                continue;
            }

            int deriv_parameter_idx = 0;
            for ( int idx=0; idx<deriv_idx; idx++ ) {
                deriv_parameter_idx += gates[idx]->get_parameter_num();
            }

            Matrix&& input_loc = input.copy();

            double* parameters = parameters_mtx_in.get_data();
            parameters = parameters + parameter_num;


            std::vector<Matrix> grad_loc;

            for( int idx=gates.size()-1; idx>=0; idx--) {

                Gate* operation = gates[idx];
                parameters = parameters - operation->get_parameter_num();
                Matrix_real parameters_mtx(parameters, 1, operation->get_parameter_num());

                if (operation->get_type() == CNOT_OPERATION) {
                    CNOT* cnot_operation = static_cast<CNOT*>(operation);
                    if ( deriv_idx < idx ) {
                        cnot_operation->apply_to( input_loc );    
                    }
                    else {
                        cnot_operation->apply_to_list( grad_loc );
                    }
                }
                else if (operation->get_type() == CZ_OPERATION) {
                    CZ* cz_operation = static_cast<CZ*>(operation);
                    if ( deriv_idx < idx ) {
                        cz_operation->apply_to( input_loc );    
                    }
                    else {
                        cz_operation->apply_to_list( grad_loc );
                    }
                }
                else if (operation->get_type() == CH_OPERATION) {
                    CH* ch_operation = static_cast<CH*>(operation);
                    if ( deriv_idx < idx ) {
                        ch_operation->apply_to( input_loc );    
                    }
                    else {
                        ch_operation->apply_to_list( grad_loc );
                    }
                }    
    
                else if (operation->get_type() == SYC_OPERATION) {
	    	    sstream << "Sycamore operation not supported in gardient calculation" << std::endl;
	            verbose_level=1;
                    print(sstream,verbose_level);	
	            
                    
                    exit(-1);
                }

                else if (operation->get_type() == U3_OPERATION) {
    
                    U3* u3_operation = static_cast<U3*>(operation);
                    if ( deriv_idx < idx ) {
                        u3_operation->apply_to( parameters_mtx, input_loc );    
                    }
                    else if ( deriv_idx == idx ) {
    
                        grad_loc = u3_operation->apply_derivate_to( parameters_mtx, input_loc );    
                    }
                    else {
                        u3_operation->apply_to_list( parameters_mtx, grad_loc );
                    }
                }

                else if (operation->get_type() == RX_OPERATION) {
                    RX* rx_operation = static_cast<RX*>(operation);
                    if ( deriv_idx < idx ) {
                        rx_operation->apply_to( parameters_mtx, input_loc );    
                    }
                    else if ( deriv_idx == idx ) {
    
                        grad_loc = rx_operation->apply_derivate_to( parameters_mtx, input_loc );    
                    }
                    else {
                        rx_operation->apply_to_list( parameters_mtx, grad_loc );
                    }
                }


                else if (operation->get_type() == RY_OPERATION) {
                    RY* ry_operation = static_cast<RY*>(operation);
                    if ( deriv_idx < idx ) {
                        ry_operation->apply_to( parameters_mtx, input_loc );    
                    }
                    else if ( deriv_idx == idx ) {
    
                        grad_loc = ry_operation->apply_derivate_to( parameters_mtx, input_loc );    
                    }
                    else {
                        ry_operation->apply_to_list( parameters_mtx, grad_loc );    
                    }    
                }

                else if (operation->get_type() == CRY_OPERATION) {
                    CRY* cry_operation = static_cast<CRY*>(operation);
                    if ( deriv_idx < idx ) {
                        cry_operation->apply_to( parameters_mtx, input_loc );    
                    }
                    else if ( deriv_idx == idx ) {
                        grad_loc = cry_operation->apply_derivate_to( parameters_mtx, input_loc );    
                    }
                    else {
                        cry_operation->apply_to_list( parameters_mtx, grad_loc );
                    }
                }

                else if (operation->get_type() == RZ_OPERATION) {
                    RZ* rz_operation = static_cast<RZ*>(operation);
                    if ( deriv_idx < idx ) {
                        rz_operation->apply_to( parameters_mtx, input_loc );    
                    }
                    else if ( deriv_idx == idx ) {
    
                        grad_loc = rz_operation->apply_derivate_to( parameters_mtx, input_loc );    
                    }
                    else {
                        rz_operation->apply_to_list( parameters_mtx, grad_loc );
                    }
                }

                else if (operation->get_type() == X_OPERATION) {
                    X* x_operation = static_cast<X*>(operation);
                    if ( deriv_idx < idx ) {
                        x_operation->apply_to( input_loc );    
                    }
                    else {
                        x_operation->apply_to_list( grad_loc );
                    }
                }
                else if (operation->get_type() == SX_OPERATION) {
                    SX* sx_operation = static_cast<SX*>(operation);
                    if ( deriv_idx < idx ) {
                        sx_operation->apply_to( input_loc );    
                    }
                    else {
                        sx_operation->apply_to_list( grad_loc );
                    }
                }
                else if (operation->get_type() == GENERAL_OPERATION) {
                    if ( deriv_idx < idx ) {
                        operation->apply_to( input_loc );    
                    }
                    else {
                        operation->apply_to_list( grad_loc );
                    }
                }

                else if (operation->get_type() == UN_OPERATION) {
	    	    sstream << "UN operation not supported in gardient calculation" << std::endl;
	            verbose_level=1;
                    print(sstream,verbose_level);	
	            
                    
                    exit(-1);
                }
                else if (operation->get_type() == ON_OPERATION) {
	    	    sstream << "ON operation not supported in gardient calculation" << std::endl;
	            verbose_level=1;
                    print(sstream,verbose_level);	
	            
                    
                    exit(-1);
                }

                else if (operation->get_type() == BLOCK_OPERATION) {
                    Gates_block* block_operation = static_cast<Gates_block*>(operation);
                    if ( deriv_idx < idx ) {
                        block_operation->apply_to( parameters_mtx, input_loc );    
                    }
                    else if ( deriv_idx == idx ) {
                        grad_loc = block_operation->apply_derivate_to( parameters_mtx, input_loc );    
                    }
                    else {
                        block_operation->apply_to_list( parameters_mtx, grad_loc );
                    }
                }

                else if (operation->get_type() == COMPOSITE_OPERATION) {
	    	    sstream << "Composite  operation not supported in gardient calculation" << std::endl;
	            verbose_level=1;
                    print(sstream,verbose_level);	
	            
                    
                    exit(-1);
                }

                else if (operation->get_type() == ADAPTIVE_OPERATION) {
                    Adaptive* ad_operation = static_cast<Adaptive*>(operation);
                    if ( deriv_idx < idx ) {
                        ad_operation->apply_to( parameters_mtx, input_loc );    
                    }
                    else if ( deriv_idx == idx ) {
                        grad_loc = ad_operation->apply_derivate_to( parameters_mtx, input_loc );    
                    }
                    else {
                        ad_operation->apply_to_list( parameters_mtx, grad_loc );
                    }
                }

            }


            for ( int idx = 0; idx<grad_loc.size(); idx++ ) {
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
void Gates_block::add_u3_to_end(int target_qbit, bool Theta, bool Phi, bool Lambda) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new U3( qbit_num, target_qbit, Theta, Phi, Lambda ));

        // adding the operation to the end of the list of gates
        add_gate_to_end( operation );
}

/**
@brief Add a U3 gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param Theta The Theta parameter of the U3 operation
@param Phi The Phi parameter of the U3 operation
@param Lambda The Lambda parameter of the U3 operation
*/
void Gates_block::add_u3(int target_qbit, bool Theta, bool Phi, bool Lambda) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new U3( qbit_num, target_qbit, Theta, Phi, Lambda ));

        // adding the operation to the front of the list of gates
        add_gate( gate );

}



/**
@brief Append a RX gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_rx_to_end(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new RX( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate_to_end( operation );
}

/**
@brief Add a RX gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_rx(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new RX( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate( gate );

}



/**
@brief Append a RY gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_ry_to_end(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new RY( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate_to_end( operation );
}


/**
@brief Add a RY gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_ry(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new RY( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate( gate );

}




/**
@brief Append a RY gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cry_to_end(int target_qbit, int control_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new CRY( qbit_num, target_qbit, control_qbit));

        // adding the operation to the end of the list of gates
        add_gate_to_end( operation );
}



/**
@brief Add a RY gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cry(int target_qbit, int control_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new CRY( qbit_num, target_qbit, control_qbit ));

        // adding the operation to the front of the list of gates
        add_gate( gate );

}



/**
@brief Append a RZ gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_rz_to_end(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new RZ( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate_to_end( operation );
}

/**
@brief Add a RZ gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_rz(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new RZ( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate( gate );

}

/**
@brief Append a C_NOT gate operation to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cnot_to_end(  int target_qbit, int control_qbit) {

        // new cnot operation
        Gate* gate = static_cast<Gate*>(new CNOT(qbit_num, target_qbit, control_qbit ));

        // append the operation to the list
        add_gate_to_end(gate);

}



/**
@brief Add a C_NOT gate operation to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cnot( int target_qbit, int control_qbit) {

        // new cnot operation
        Gate* gate = static_cast<Gate*>(new CNOT(qbit_num, target_qbit, control_qbit ));

        // put the operation to tghe front of the list
        add_gate(gate);

}




/**
@brief Append a CZ gate operation to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cz_to_end(  int target_qbit, int control_qbit) {

        // new cz operation
        Gate* gate = static_cast<Gate*>(new CZ(qbit_num, target_qbit, control_qbit ));

        // append the operation to the list
        add_gate_to_end(gate);

}



/**
@brief Add a CZ gate operation to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_cz(  int target_qbit, int control_qbit) {

        // new cz operation
        Gate* gate = static_cast<Gate*>(new CZ(qbit_num, target_qbit, control_qbit ));

        // put the operation to tghe front of the list
        add_gate(gate);

}


/**
@brief Append a X gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_x_to_end(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new X( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate_to_end( operation );
}

/**
@brief Add a X gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_x(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new X( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate( gate );

}



/**
@brief Append a SX gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_sx_to_end(int target_qbit) {

        // create the operation
        Gate* operation = static_cast<Gate*>(new SX( qbit_num, target_qbit));

        // adding the operation to the end of the list of gates
        add_gate_to_end( operation );
}

/**
@brief Add a SX gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_sx(int target_qbit ) {

        // create the operation
        Gate* gate = static_cast<Gate*>(new SX( qbit_num, target_qbit ));

        // adding the operation to the front of the list of gates
        add_gate( gate );

}





/**
@brief Append a Sycamore gate operation to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_syc_to_end(  int target_qbit, int control_qbit) {

        // new cz operation
        Gate* gate = static_cast<Gate*>(new SYC(qbit_num, target_qbit, control_qbit ));

        // append the operation to the list
        add_gate_to_end(gate);

}



/**
@brief Add a Sycamore gate operation to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_syc(  int target_qbit, int control_qbit) {

        // new cz operation
        Gate* gate = static_cast<Gate*>(new SYC(qbit_num, target_qbit, control_qbit ));

        // put the operation to tghe front of the list
        add_gate(gate);

}




/**
@brief Append a CH gate (i.e. controlled Hadamard gate) operation to the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_ch_to_end(  int target_qbit, int control_qbit) {

        // new cz operation
        Gate* gate = static_cast<Gate*>(new CH(qbit_num, target_qbit, control_qbit ));

        // append the operation to the list
        add_gate_to_end(gate);

}



/**
@brief Add a CH gate (i.e. controlled Hadamard gate) operation to the front of the list of gates
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_ch(  int target_qbit, int control_qbit) {

        // new cz operation
        Gate* gate = static_cast<Gate*>(new CH(qbit_num, target_qbit, control_qbit ));

        // put the operation to tghe front of the list
        add_gate(gate);

}

/**
@brief Append a list of gates to the list of gates
@param gates_in A list of operation class instances.
*/
void Gates_block::add_gates_to_end( std::vector<Gate*> gates_in) {

        for(std::vector<Gate*>::iterator it = gates_in.begin(); it != gates_in.end(); ++it) {
            add_gate_to_end( *it );
        }

}


/**
@brief Add an array of gates to the front of the list of gates
@param gates_in A list of operation class instances.
*/
void Gates_block::add_gates( std::vector<Gate*>  gates_in) {

        // adding gates in reversed order!!
        for(std::vector<Gate*>::iterator it = gates_in.end(); it != gates_in.begin(); --it) {
            add_gate( *it );
        }

}



/**
@brief Append a UN gate to the list of gates
*/
void Gates_block::add_un_to_end() {

        // create the operation
        Gate* operation = static_cast<Gate*>(new UN( qbit_num ));

        // adding the operation to the end of the list of gates
        add_gate_to_end( operation );
}

/**
@brief Add a UN gate to the front of the list of gates
*/
void Gates_block::add_un() {

        // create the operation
        Gate* gate = static_cast<Gate*>(new UN( qbit_num ));

        // adding the operation to the front of the list of gates
        add_gate( gate );

}


/**
@brief Append a ON gate to the list of gates
*/
void Gates_block::add_on_to_end() {

        // create the operation
        Gate* operation = static_cast<Gate*>(new ON( qbit_num ));

        // adding the operation to the end of the list of gates
        add_gate_to_end( operation );
}

/**
@brief Add a ON gate to the front of the list of gates
*/
void Gates_block::add_on() {

        // create the operation
        Gate* gate = static_cast<Gate*>(new ON( qbit_num ));

        // adding the operation to the front of the list of gates
        add_gate( gate );

}


/**
@brief Append a Composite gate to the list of gates
*/
void Gates_block::add_composite_to_end()  {

        // create the operation
        Gate* operation = static_cast<Gate*>(new Composite( qbit_num ));

        // adding the operation to the end of the list of gates
        add_gate_to_end( operation );
}

/**
@brief Add a Composite gate to the front of the list of gates
*/
void Gates_block::add_composite()  {

        // create the operation
        Gate* gate = static_cast<Gate*>(new Composite( qbit_num ));

        // adding the operation to the front of the list of gates
        add_gate( gate );

}



/**
@brief Append a Adaptive gate to the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_adaptive_to_end(int target_qbit, int control_qbit)  {

        // create the operation
        Gate* operation = static_cast<Gate*>(new Adaptive( qbit_num, target_qbit, control_qbit));

        // adding the operation to the end of the list of gates
        add_gate_to_end( operation );
}


/**
@brief Add a Adaptive gate to the front of the list of gates
@param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
void Gates_block::add_adaptive(int target_qbit, int control_qbit)  {

        // create the operation
        Gate* gate = static_cast<Gate*>(new Adaptive( qbit_num, target_qbit, control_qbit ));

        // adding the operation to the front of the list of gates
        add_gate( gate );

}



/**
@brief Append a general gate to the list of gates
@param gate A pointer to a class Gate describing a gate operation.
*/
void Gates_block::add_gate_to_end( Gate* gate ) {

        //set the number of qubit in the gate
        gate->set_qbit_num( qbit_num );

        // append the gate to the list
        gates.push_back(gate);

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
 void Gates_block::add_gate( Gate* gate) {


        // set the number of qubit in the gate
        gate->set_qbit_num( qbit_num );

        gates.insert( gates.begin(), gate);

        // increase the number of U3 gate parameters by the number of parameters
        parameter_num = parameter_num + gate->get_parameter_num();

        // increase the number of layers if necessary
        if (gate->get_type() == BLOCK_OPERATION) {
            layer_num = layer_num + 1;
        }

}



/**
@brief ??????
@param gate A pointer to a class Gate describing an gate.
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
        gate_nums.sx      = 0;
        gate_nums.syc     = 0;
        gate_nums.un     = 0;
        gate_nums.on     = 0;
        gate_nums.com     = 0;
        gate_nums.general = 0;
        gate_nums.adap = 0;

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
                gate_nums.x  = gate_nums.x + gate_nums_loc.x;
                gate_nums.sx = gate_nums.sx + gate_nums_loc.sx;
                gate_nums.syc   = gate_nums.syc + gate_nums_loc.syc;
                gate_nums.un   = gate_nums.un + gate_nums_loc.un;
                gate_nums.on   = gate_nums.on + gate_nums_loc.on;
                gate_nums.com  = gate_nums.com + gate_nums_loc.com;
                gate_nums.adap = gate_nums.adap + gate_nums_loc.adap;
            }
            else if (gate->get_type() == U3_OPERATION) {
                gate_nums.u3   = gate_nums.u3 + 1;
            }
            else if (gate->get_type() == RX_OPERATION) {
                gate_nums.rx   = gate_nums.rx + 1;
            }
            else if (gate->get_type() == RY_OPERATION) {
                gate_nums.ry   = gate_nums.ry + 1;
            }
            else if (gate->get_type() == CRY_OPERATION) {
                gate_nums.cry   = gate_nums.cry + 1;
            }
            else if (gate->get_type() == RZ_OPERATION) {
                gate_nums.rz   = gate_nums.rz + 1;
            }
            else if (gate->get_type() == CNOT_OPERATION) {
                gate_nums.cnot   = gate_nums.cnot + 1;
            }
            else if (gate->get_type() == CZ_OPERATION) {
                gate_nums.cz   = gate_nums.cz + 1;
            }
            else if (gate->get_type() == CH_OPERATION) {
                gate_nums.ch   = gate_nums.ch + 1;
            }
            else if (gate->get_type() == X_OPERATION) {
                gate_nums.x   = gate_nums.x + 1;
            }
            else if (gate->get_type() == SX_OPERATION) {
                gate_nums.sx   = gate_nums.sx + 1;
            }
            else if (gate->get_type() == SYC_OPERATION) {
                gate_nums.syc   = gate_nums.syc + 1;
            }
            else if (gate->get_type() == GENERAL_OPERATION) {
                gate_nums.general   = gate_nums.general + 1;
            }
            else if (gate->get_type() == UN_OPERATION) {
                gate_nums.un   = gate_nums.un + 1;
            }
            else if (gate->get_type() == ON_OPERATION) {
                gate_nums.on   = gate_nums.on + 1;
            }
            else if (gate->get_type() == COMPOSITE_OPERATION) {
                gate_nums.com   = gate_nums.com + 1;
            }
            else if (gate->get_type() == ADAPTIVE_OPERATION) {
                gate_nums.adap   = gate_nums.adap + 1;
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

		verbose_level=1;
		sstream << std::endl << "The gates in the list of gates:" << std::endl;
		print(sstream,verbose_level);	    	
		

      

        int gate_idx = start_index;
        int parameter_idx = parameter_num;

        for(int op_idx = gates.size()-1; op_idx>=0; op_idx--) {

            Gate* gate = gates[op_idx];

            if (gate->get_type() == CNOT_OPERATION) {
                CNOT* cnot_gate = static_cast<CNOT*>(gate);

		verbose_level=1;
		sstream << gate_idx << "th gate: CNOT with control qubit: " << cnot_gate->get_control_qbit() << " and target qubit: " << cnot_gate->get_target_qbit() << std::endl;
		print(sstream,verbose_level);	    	
		

                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == CZ_OPERATION) {
                CZ* cz_gate = static_cast<CZ*>(gate);

		verbose_level=1;
		sstream << gate_idx << "th gate: CZ with control qubit: " << cz_gate->get_control_qbit() << " and target qubit: " << cz_gate->get_target_qbit() << std::endl;
		print(sstream,verbose_level);	    	
		
                
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == CH_OPERATION) {
                CH* ch_gate = static_cast<CH*>(gate);

		verbose_level=1;
		sstream << gate_idx << "th gate: CH with control qubit: " << ch_gate->get_control_qbit() << " and target qubit: " << ch_gate->get_target_qbit() << std::endl;
		print(sstream,verbose_level);	    	
		
               
                gate_idx = gate_idx + 1;
            }
            else if (gate->get_type() == SYC_OPERATION) {
                SYC* syc_gate = static_cast<SYC*>(gate);

		verbose_level=1;
		sstream << gate_idx << "th gate: Sycamore gate with control qubit: " << syc_gate->get_control_qbit() << " and target qubit: " << syc_gate->get_target_qbit() << std::endl;
		print(sstream,verbose_level);	    	
		
                
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
                    vartheta = std::fmod( parameters[parameter_idx-1], 4*M_PI);
                    varphi = 0;
                    varlambda =0;
                    parameter_idx = parameter_idx - 1;

                }
                else if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_phi_parameter()) {
                    vartheta = 0;
                    varphi = std::fmod( parameters[ parameter_idx-1 ], 2*M_PI);
                    varlambda =0;
                    parameter_idx = parameter_idx - 1;
                }
                else if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_lambda_parameter()) {
                    vartheta = 0;
                    varphi =  0;
                    varlambda = std::fmod( parameters[ parameter_idx-1 ], 2*M_PI);
                    parameter_idx = parameter_idx - 1;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_theta_parameter() && u3_gate->is_phi_parameter() ) {
                    vartheta = std::fmod( parameters[ parameter_idx-2 ], 4*M_PI);
                    varphi = std::fmod( parameters[ parameter_idx-1 ], 2*M_PI);
                    varlambda = 0;
                    parameter_idx = parameter_idx - 2;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_theta_parameter() && u3_gate->is_lambda_parameter() ) {
                    vartheta = std::fmod( parameters[ parameter_idx-2 ], 4*M_PI);
                    varphi = 0;
                    varlambda = std::fmod( parameters[ parameter_idx-1 ], 2*M_PI);
                    parameter_idx = parameter_idx - 2;
                }
                else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_phi_parameter() && u3_gate->is_lambda_parameter() ) {
                    vartheta = 0;
                    varphi = std::fmod( parameters[ parameter_idx-2], 2*M_PI);
                    varlambda = std::fmod( parameters[ parameter_idx-1 ], 2*M_PI);
                    parameter_idx = parameter_idx - 2;
                }
                else if ((u3_gate->get_parameter_num() == 3)) {
                    vartheta = std::fmod( parameters[ parameter_idx-3 ], 4*M_PI);
                    varphi = std::fmod( parameters[ parameter_idx-2 ], 2*M_PI);
                    varlambda = std::fmod( parameters[ parameter_idx-1 ], 2*M_PI);
                    parameter_idx = parameter_idx - 3;
                }

//                message = message + "U3 on target qubit %d with parameters theta = %f, phi = %f and lambda = %f";

		verbose_level=1;
		sstream << gate_idx << "th gate: U3 on target qubit: " << u3_gate->get_target_qbit() << " and with parameters theta = " << vartheta << ", phi = " << varphi << " and lambda = " << varlambda << std::endl;
		print(sstream,verbose_level);	    	
		
                
                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == RX_OPERATION) {

                // definig the rotation parameter
                double vartheta;

                // get the inverse parameters of the U3 rotation

                RX* rx_gate = static_cast<RX*>(gate);

                vartheta = std::fmod( parameters[parameter_idx-1], 4*M_PI);
                parameter_idx = parameter_idx - 1;

		verbose_level=1;
		sstream << gate_idx << "th gate: RX on target qubit: " << rx_gate->get_target_qbit() << " and with parameters theta = " << vartheta << std::endl;
		print(sstream,verbose_level);	    	
		

                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == RY_OPERATION) {

                // definig the rotation parameter
                double vartheta;

                // get the inverse parameters of the U3 rotation

                RY* ry_gate = static_cast<RY*>(gate);

                vartheta = std::fmod( parameters[parameter_idx-1], 4*M_PI);
                parameter_idx = parameter_idx - 1;

		verbose_level=1;
		sstream << gate_idx << "th gate: RY on target qubit: " << ry_gate->get_target_qbit() << " and with parameters theta = " << vartheta << std::endl; 
		print(sstream,verbose_level);	    	
		
               
                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == CRY_OPERATION) {

                // definig the rotation parameter
                double Phi;

                // get the inverse parameters of the U3 rotation

                CRY* cry_gate = static_cast<CRY*>(gate);

                Phi = std::fmod( parameters[parameter_idx-1], 2*M_PI);
                parameter_idx = parameter_idx - 1;

		verbose_level=1;
		sstream << gate_idx << "th gate: CRY on target qubit: " << cry_gate->get_target_qbit() << ", control qubit" << cry_gate->get_control_qbit() << " and with parameters Phi = " << Phi << std::endl;
		print(sstream,verbose_level);	    	
		
                
                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == RZ_OPERATION) {

                // definig the rotation parameter
                double varphi;

                // get the inverse parameters of the U3 rotation

                RZ* rz_gate = static_cast<RZ*>(gate);

                varphi = std::fmod( parameters[parameter_idx-1], 2*M_PI);
                parameter_idx = parameter_idx - 1;

		verbose_level=1;
		sstream << gate_idx << "th gate: RZ on target qubit: " << rz_gate->get_target_qbit() << " and with parameters varphi = " << varphi << std::endl;
		print(sstream,verbose_level);	    	
		
                
                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == X_OPERATION) {

                // get the inverse parameters of the U3 rotation

                X* x_gate = static_cast<X*>(gate);

		verbose_level=1;
		sstream << gate_idx << "th gate: X on target qubit: " << x_gate->get_target_qbit() << std::endl;
		print(sstream,verbose_level);	    	
		
                
                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == SX_OPERATION) {

                // get the inverse parameters of the U3 rotation

                SX* sx_gate = static_cast<SX*>(gate);

		verbose_level=1;
		sstream << gate_idx << "th gate: SX on target qubit: " << sx_gate->get_target_qbit() << std::endl;
		print(sstream,verbose_level);	    	
		
                
                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == BLOCK_OPERATION) {
                Gates_block* block_gate = static_cast<Gates_block*>(gate);
                const Matrix_real parameters_layer(parameters.get_data() + parameter_idx - gate->get_parameter_num(), 1, gate->get_parameter_num() );
                block_gate->list_gates( parameters_layer, gate_idx );
                parameter_idx = parameter_idx - block_gate->get_parameter_num();
                gate_idx = gate_idx + block_gate->get_gate_num();
            }
            else if (gate->get_type() == UN_OPERATION) {

                parameter_idx = parameter_idx - gate->get_parameter_num();

		verbose_level=1;
		sstream << gate_idx << "th gate: UN " << gate->get_parameter_num() << " parameters" << std::endl;
		print(sstream,verbose_level);	    	
		
                
                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == ON_OPERATION) {

                parameter_idx = parameter_idx - gate->get_parameter_num();

		verbose_level=1;
		sstream << gate_idx << "th gate: ON " << gate->get_parameter_num() << " parameters" << std::endl;
		print(sstream,verbose_level);	    	
		
                
                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == COMPOSITE_OPERATION) {

                parameter_idx = parameter_idx - gate->get_parameter_num();

		verbose_level=1;
		sstream << gate_idx << "th gate: Composite " << gate->get_parameter_num() << " parameters" << std::endl;
		print(sstream,verbose_level);	    	
		
                
                gate_idx = gate_idx + 1;

            }
            else if (gate->get_type() == ADAPTIVE_OPERATION) {

                // definig the rotation parameter
                double Phi;

                // get the inverse parameters of the U3 rotation

                Adaptive* ad_gate = static_cast<Adaptive*>(gate);

                Phi = std::fmod( parameters[parameter_idx-1], 2*M_PI);
                parameter_idx = parameter_idx - 1;

		verbose_level=1;
		sstream << gate_idx << "th gate: Adaptive gate on target qubit: " << ad_gate->get_target_qbit() << ", control qubit " << ad_gate->get_control_qbit() << " and with parameters Phi = " << Phi << std::endl;
		print(sstream,verbose_level);	    	
		
                
                gate_idx = gate_idx + 1;

            }

        }

        fflush(stdout);

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
         else if (gate->get_type() == X_OPERATION) {
             X* x_gate = static_cast<X*>(gate);
             x_gate->reorder_qubits( qbit_list );
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
@brief Call to get the gates stored in the class.
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

    if (idx > gates.size() ) {
        return NULL;
    }

    return gates[idx];
}


/**
@brief Call to append the gates of a gate block to the current block
@param op_block A pointer to an instance of class Gates_block
*/
void Gates_block::combine(Gates_block* op_block) {

    // getting the list of gates
    std::vector<Gate*> gates_in = op_block->get_gates();

    for(std::vector<Gate*>::iterator it = (gates_in).begin(); it != (gates_in).end(); ++it) {
        Gate* op = *it;

        if (op->get_type() == CNOT_OPERATION) {
            CNOT* cnot_op = static_cast<CNOT*>( op );
            CNOT* cnot_op_cloned = cnot_op->clone();
            Gate* op_cloned = static_cast<Gate*>( cnot_op_cloned );
            add_gate_to_end(op_cloned);
        }
        else if (op->get_type() == CZ_OPERATION) {
            CZ* cz_op = static_cast<CZ*>( op );
            CZ* cz_op_cloned = cz_op->clone();
            Gate* op_cloned = static_cast<Gate*>( cz_op_cloned );
            add_gate_to_end(op_cloned);
        }
        else if (op->get_type() == CH_OPERATION) {
            CH* ch_op = static_cast<CH*>( op );
            CH* ch_op_cloned = ch_op->clone();
            Gate* op_cloned = static_cast<Gate*>( ch_op_cloned );
            add_gate_to_end(op_cloned);
        }
        else if (op->get_type() == SYC_OPERATION) {
            SYC* syc_op = static_cast<SYC*>( op );
            SYC* syc_op_cloned = syc_op->clone();
            Gate* op_cloned = static_cast<Gate*>( syc_op_cloned );
            add_gate_to_end(op_cloned);
        }
        else if (op->get_type() == U3_OPERATION) {
            U3* u3_op = static_cast<U3*>( op );
            U3* u3_op_cloned = u3_op->clone();
            Gate* op_cloned = static_cast<Gate*>( u3_op_cloned );
            add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == RX_OPERATION) {
            RX* rx_op = static_cast<RX*>( op );
            RX* rx_op_cloned = rx_op->clone();
            Gate* op_cloned = static_cast<Gate*>( rx_op_cloned );
            add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == RY_OPERATION) {
            RY* ry_op = static_cast<RY*>( op );
            RY* ry_op_cloned = ry_op->clone();
            Gate* op_cloned = static_cast<Gate*>( ry_op_cloned );
            add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == CRY_OPERATION) {
            CRY* cry_op = static_cast<CRY*>( op );
            CRY* cry_op_cloned = cry_op->clone();
            Gate* op_cloned = static_cast<Gate*>( cry_op_cloned );
            add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == RZ_OPERATION) {
            RZ* rz_op = static_cast<RZ*>( op );
            RZ* rz_op_cloned = rz_op->clone();
            Gate* op_cloned = static_cast<Gate*>( rz_op_cloned );
            add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == X_OPERATION) {
            X* x_op = static_cast<X*>( op );
            X* x_op_cloned = x_op->clone();
            Gate* op_cloned = static_cast<Gate*>( x_op_cloned );
            add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == SX_OPERATION) {
            SX* sx_op = static_cast<SX*>( op );
            SX* sx_op_cloned = sx_op->clone();
            Gate* op_cloned = static_cast<Gate*>( sx_op_cloned );
            add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == BLOCK_OPERATION) {
            Gates_block* block_op = static_cast<Gates_block*>( op );
            Gates_block* block_op_cloned = block_op->clone();
            Gate* op_cloned = static_cast<Gate*>( block_op_cloned );
            add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == GENERAL_OPERATION) {
            Gate* op_cloned = op->clone();
            add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == UN_OPERATION) {
            UN* un_op = static_cast<UN*>( op );
            UN* un_op_cloned = un_op->clone();
            Gate* op_cloned = static_cast<Gate*>( un_op_cloned );
            add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == ON_OPERATION) {
            ON* on_op = static_cast<ON*>( op );
            ON* on_op_cloned = on_op->clone();
            Gate* op_cloned = static_cast<Gate*>( on_op_cloned );
            add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == COMPOSITE_OPERATION) {
            Composite* com_op = static_cast<Composite*>( op );
            Composite* com_op_cloned = com_op->clone();
            Gate* op_cloned = static_cast<Gate*>( com_op_cloned );
            add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == ADAPTIVE_OPERATION) {
            Adaptive* ad_op = static_cast<Adaptive*>( op );
            Adaptive* ad_op_cloned = ad_op->clone();
            Gate* op_cloned = static_cast<Gate*>( ad_op_cloned );
            add_gate_to_end( op_cloned );
        }

    }

}


/**
@brief Set the number of qubits spanning the matrix of the gates stored in the block of gates.
@param qbit_num_in The number of qubits spanning the matrices.
*/
void Gates_block::set_qbit_num( int qbit_num_in ) {

    // setting the number of qubits
    Gate::set_qbit_num(qbit_num_in);

    // setting the number of qubit in the gates
    for(std::vector<Gate*>::iterator it = gates.begin(); it != gates.end(); ++it) {
        Gate* op = *it;

        if (op->get_type() == CNOT_OPERATION) {
            CNOT* cnot_op = static_cast<CNOT*>( op );
            cnot_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == CZ_OPERATION) {
            CZ* cz_op = static_cast<CZ*>( op );
            cz_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == CH_OPERATION) {
            CH* ch_op = static_cast<CH*>( op );
            ch_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == SYC_OPERATION) {
            SYC* syc_op = static_cast<SYC*>( op );
            syc_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == U3_OPERATION) {
            U3* u3_op = static_cast<U3*>( op );
            u3_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == RX_OPERATION) {
            RX* rx_op = static_cast<RX*>( op );
            rx_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == RY_OPERATION) {
            RY* ry_op = static_cast<RY*>( op );
            ry_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == CRY_OPERATION) {
            CRY* cry_op = static_cast<CRY*>( op );
            cry_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == RZ_OPERATION) {
            RZ* rz_op = static_cast<RZ*>( op );
            rz_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == X_OPERATION) {
            X* x_op = static_cast<X*>( op );
            x_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == SX_OPERATION) {
            SX* sx_op = static_cast<SX*>( op );
            sx_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == BLOCK_OPERATION) {
            Gates_block* block_op = static_cast<Gates_block*>( op );
            block_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == UN_OPERATION) {
            UN* un_op = static_cast<UN*>( op );
            un_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == ON_OPERATION) {
            ON* on_op = static_cast<ON*>( op );
            on_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == COMPOSITE_OPERATION) {
            Composite* com_op = static_cast<Composite*>( op );
            com_op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == GENERAL_OPERATION) {
            op->set_qbit_num( qbit_num_in );
        }
        else if (op->get_type() == ADAPTIVE_OPERATION) {
            Adaptive* ad_op = static_cast<Adaptive*>( op );
            ad_op->set_qbit_num( qbit_num_in );
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

    // extracting the gates from the current class
    if (extract_gates( ret ) != 0 ) {

	verbose_level=1;
	sstream << "Gates_block::clone(): extracting gates was not succesfull" << std::endl;
	print(sstream,verbose_level);	    	
	
        
        exit(-1);
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

        if (op->get_type() == CNOT_OPERATION) {
            CNOT* cnot_op = static_cast<CNOT*>( op );
            CNOT* cnot_op_cloned = cnot_op->clone();
            Gate* op_cloned = static_cast<Gate*>( cnot_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == CZ_OPERATION) {
            CZ* cz_op = static_cast<CZ*>( op );
            CZ* cz_op_cloned = cz_op->clone();
            Gate* op_cloned = static_cast<Gate*>( cz_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == CH_OPERATION) {
            CH* ch_op = static_cast<CH*>( op );
            CH* ch_op_cloned = ch_op->clone();
            Gate* op_cloned = static_cast<Gate*>( ch_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == SYC_OPERATION) {
            SYC* syc_op = static_cast<SYC*>( op );
            SYC* syc_op_cloned = syc_op->clone();
            Gate* op_cloned = static_cast<Gate*>( syc_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == U3_OPERATION) {
            U3* u3_op = static_cast<U3*>( op );
            U3* u3_op_cloned = u3_op->clone();
            Gate* op_cloned = static_cast<Gate*>( u3_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == RX_OPERATION) {
            RX* rx_op = static_cast<RX*>( op );
            RX* rx_op_cloned = rx_op->clone();
            Gate* op_cloned = static_cast<Gate*>( rx_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == RY_OPERATION) {
            RY* ry_op = static_cast<RY*>( op );
            RY* ry_op_cloned = ry_op->clone();
            Gate* op_cloned = static_cast<Gate*>( ry_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == CRY_OPERATION) {
            CRY* cry_op = static_cast<CRY*>( op );
            CRY* cry_op_cloned = cry_op->clone();
            Gate* op_cloned = static_cast<Gate*>( cry_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == RZ_OPERATION) {
            RZ* rz_op = static_cast<RZ*>( op );
            RZ* rz_op_cloned = rz_op->clone();
            Gate* op_cloned = static_cast<Gate*>( rz_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == X_OPERATION) {
            X* x_op = static_cast<X*>( op );
            X* x_op_cloned = x_op->clone();
            Gate* op_cloned = static_cast<Gate*>( x_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == SX_OPERATION) {
            SX* sx_op = static_cast<SX*>( op );
            SX* sx_op_cloned = sx_op->clone();
            Gate* op_cloned = static_cast<Gate*>( sx_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == BLOCK_OPERATION) {
            Gates_block* block_op = static_cast<Gates_block*>( op );
            Gates_block* block_op_cloned = block_op->clone();
            Gate* op_cloned = static_cast<Gate*>( block_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == UN_OPERATION) {
            UN* un_op = static_cast<UN*>( op );
            UN* un_op_cloned = un_op->clone();
            Gate* op_cloned = static_cast<Gate*>( un_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == ON_OPERATION) {
            ON* on_op = static_cast<ON*>( op );
            ON* on_op_cloned = on_op->clone();
            Gate* op_cloned = static_cast<Gate*>( on_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == COMPOSITE_OPERATION) {
            Composite* com_op = static_cast<Composite*>( op );
            Composite* com_op_cloned = com_op->clone();
            Gate* op_cloned = static_cast<Gate*>( com_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == GENERAL_OPERATION) {
            Gate* op_cloned = op->clone();
            op_block->add_gate_to_end( op_cloned );
        }
        else if (op->get_type() == ADAPTIVE_OPERATION) {
            Adaptive* ad_op = static_cast<Adaptive*>( op );
            Adaptive* ad_op_cloned = ad_op->clone();
            Gate* op_cloned = static_cast<Gate*>( ad_op_cloned );
            op_block->add_gate_to_end( op_cloned );
        }

    }

    return 0;

}



/**
@brief ?????????
@return Return with ?????????
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
@brief ?????????
@return Return with ?????????
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


