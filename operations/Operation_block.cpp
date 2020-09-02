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

//
// @brief A base class responsible for constructing matrices of C-NOT, U3
// gates acting on the N-qubit space


#include "qgd/CNOT.h"
#include "qgd/U3.h"
#include "qgd/Operation_block.h"

using namespace std;

//
// @brief Constructor of the class.
// @return An instance of the class
Operation_block::Operation_block() : Operation() {

    // A string describing the type of the operation
    type = "block";
    // number of operation layers
    layer_num = 0;
}



//
// @brief Constructor of the class.
// @param qbit_num The number of qubits in the unitaries
// @return An instance of the class
Operation_block::Operation_block(int qbit_num_in) : Operation(qbit_num_in) {

    // A string describing the type of the operation
    type = "block";
    // number of operation layers
    layer_num = 0;
}


//
// @brief Destructor of the class.
Operation_block::~Operation_block() {

    //free the alloctaed memory of the stored operations
    for(std::vector<Operation*>::iterator it = operations.begin(); it != operations.end(); ++it) {

        Operation* operation = *it;
            
        if (operation->get_type().compare("cnot")==0) {
            CNOT* cnot_operation = static_cast<CNOT*>(operation);
            delete cnot_operation;
        }                
        else if (operation->get_type().compare("u3")==0) {

            U3* u3_operation = static_cast<U3*>(operation);
            delete u3_operation;

        }         
        else if (operation->get_type().compare("block")==0) {

            Operation_block* block_operation = static_cast<Operation_block*>(operation);
            delete block_operation;

        }                                 
        else if (operation->get_type().compare("general")==0) {
            delete operation;
        }
    }

    operations.clear();

}


////
// @brief Call to get the product of the matrices of the operations grouped in the block.
// @param parameters List of parameters to calculate the matrix of the operation block
// @param free_after_used Logical value indicating whether the cteated matrix can be freed after it was used. (For example U3 allocates the matrix on demand, but CNOT is returning with a pointer to the stored matrix in attribute matrix_allocate)
// @return Returns with the matrix of the operation
QGD_Complex16* Operation_block::matrix( const double* parameters  ) {

    // preallocate array for the composite u3 operation
    QGD_Complex16* block_matrix = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);

    matrix( parameters, block_matrix  );

    return block_matrix;
}



////
// @brief Call to get the product of the matrices of the operations grouped in the block.
// @param parameters List of parameters to calculate the matrix of the operation block
// @param free_after_used Logical value indicating whether the cteated matrix can be freed after it was used. (For example U3 allocates the matrix on demand, but CNOT is returning with a pointer to the stored matrix in attribute matrix_allocate)
// @return Returns with the matrix of the operation
int Operation_block::matrix( const double* parameters, QGD_Complex16* block_mtx  ) {


    // get the matrices of the operations grouped in the block
    vector<QGD_Complex16*> operation_mtxs = get_matrices( parameters );

    // calculate the product of the matrices
    reduce_zgemm( operation_mtxs, block_mtx, matrix_size );

    // free the constituent matrices if possible    
    for ( std::vector<QGD_Complex16*>::iterator it=operation_mtxs.begin(); it!=operation_mtxs.end(); it++) {
        qgd_free(*it);
    }

    operation_mtxs.clear();


/*
    QGD_Complex16* tmp = (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size*sizeof(QGD_Complex16), 64 );
    QGD_Complex16* operation_mtx = (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size*sizeof(QGD_Complex16), 64 );

    int idx = 0;
    for(std::vector<Operation*>::iterator it = operations.begin(); it != operations.end(); ++it) {
      
        Operation* operation = *it;

            
        if (operation->get_type().compare("cnot")==0) {
            CNOT* cnot_operation = static_cast<CNOT*>(operation);
            cnot_operation->matrix(operation_mtx);
        }                
        else if (operation->get_type().compare("u3")==0) {

            U3* u3_operation = static_cast<U3*>(operation);

            if (u3_operation->get_parameter_num() == 1 ) {
                u3_operation->matrix( parameters, operation_mtx );
                parameters = parameters + 1;
            }        
            else if (u3_operation->get_parameter_num() == 2 ) {
                u3_operation->matrix( parameters, operation_mtx );
                parameters = parameters + 2;
            }        
            else if (u3_operation->get_parameter_num() == 3 ) {
                u3_operation->matrix( parameters, operation_mtx );
                parameters = parameters + 3;
            }
            else {
                printf("The U3 operation has wrong number of parameters");
                throw "The U3 operation has wrong number of parameters";
            }


        }                                 
        else if (operation->get_type().compare("general")==0) {
            operation->matrix(operation_mtx);
        }

      
        if (idx == 0) {
            memcpy( block_mtx, operation_mtx, matrix_size*matrix_size*sizeof(QGD_Complex16) );
        }
        else {
            zgemm3m_wrapper(block_mtx, operation_mtx, tmp, matrix_size);
            memcpy( block_mtx, tmp, matrix_size*matrix_size*sizeof(QGD_Complex16) ); 
        }


    }

    qgd_free(tmp);
    qgd_free(operation_mtx);

*/

    return 0;
}




////
// @brief Call to get the list of matrix representation of the operations grouped in the block.
// @param parameters List of parameters to calculate the matrix of the operation block
// @param free_after_used Logical value indicating whether the cteated matrix can be freed after it was used. (For example U3 allocates the matrix on demand, but CNOT is returning with a pointer to the stored matrix in attribute matrix_allocate)
// @return Returns with the matrix of the operation
std::vector<QGD_Complex16*> Operation_block::get_matrices( const double* parameters ) {

    std::vector<QGD_Complex16*> matrices;
    QGD_Complex16* operation_mtx;
             
    for(std::vector<Operation*>::iterator it = operations.begin(); it != operations.end(); ++it) {
      
        Operation* operation = *it;

            
        if (operation->get_type().compare("cnot")==0) {
            CNOT* cnot_operation = static_cast<CNOT*>(operation);
            operation_mtx = cnot_operation->matrix();

        }                
        else if (operation->get_type().compare("u3")==0) {

            U3* u3_operation = static_cast<U3*>(operation);

            if (u3_operation->get_parameter_num() == 1 ) {
                operation_mtx = u3_operation->matrix( parameters );
                parameters = parameters + 1;
            }        
            else if (u3_operation->get_parameter_num() == 2 ) {
                operation_mtx = u3_operation->matrix( parameters );
                parameters = parameters + 2;
            }        
            else if (u3_operation->get_parameter_num() == 3 ) {
                operation_mtx = u3_operation->matrix( parameters );
                parameters = parameters + 3;
            }
            else {
                printf("The U3 operation has wrong number of parameters");
                throw "The U3 operation has wrong number of parameters";
            }

        }                                 
        else if (operation->get_type().compare("general")==0) {
            operation_mtx = operation->matrix();
        }

        matrices.push_back(operation_mtx);


    }
            
    return matrices;

}



////
// @brief Append a U3 gate to the list of operations
// @param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
// @param theta_in ...
void Operation_block::add_u3_to_end(int target_qbit, bool Theta, bool Phi, bool Lambda) {    
        
        // create the operation
        Operation* operation = static_cast<Operation*>(new U3( qbit_num, target_qbit, Theta, Phi, Lambda ));
        
        // adding the operation to the end of the list of operations
        add_operation_to_end( operation );  
}            
    
////
// @brief Add a U3 gate to the front of the list of operations
// @param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
// @param parameter_labels A list of strings 'Theta', 'Phi' or 'Lambda' indicating the free parameters of the U3 operations. (Paremetrs which are not labeled are set to zero)
void Operation_block::add_u3_to_front(int target_qbit, bool Theta, bool Phi, bool Lambda) {
        
        // create the operation
        Operation* operation = static_cast<Operation*>(new U3( qbit_num, target_qbit, Theta, Phi, Lambda ));
        
        // adding the operation to the front of the list of operations
        add_operation_to_front( operation );

}
        
//// 
// @brief Append a C_NOT gate operation to the list of operations
// @param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
// @param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
void Operation_block::add_cnot_to_end( int control_qbit, int target_qbit) {
        
        // new cnot operation
        Operation* operation = static_cast<Operation*>(new CNOT(qbit_num, control_qbit, target_qbit ));
        
        // append the operation to the list
        add_operation_to_end(operation);       

}
        
        
    
//// add_cnot_to_front
// @brief Add a C_NOT gate operation to the front of the list of operations
// @param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
// @param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
void Operation_block::add_cnot_to_front( int control_qbit, int target_qbit) {
        
        // new cnot operation
        Operation* operation = static_cast<Operation*>(new CNOT(qbit_num, control_qbit, target_qbit ));
        
        // put the operation to tghe front of the list
        add_operation_to_front(operation);    
        
}
    
////
// @brief Append an array of operations to the list of operations
// @param operations A list of operation class instances.
void Operation_block::add_operations_to_end( vector<Operation*> operations_in) {

        for(std::vector<Operation*>::iterator it = operations_in.begin(); it != operations_in.end(); ++it) {
            add_operation_to_end( *it );
        }

}
            
    
//// add_operations_to_front
// @brief Add an array of operations to the front of the list of operations
// @param operations A list of operation class instances.
void Operation_block::add_operations_to_front( vector<Operation*>  operations_in) {
        
        // adding operations in reversed order!!
        for(std::vector<Operation*>::iterator it = operations_in.end(); it != operations_in.begin(); --it) {
            add_operation_to_front( *it );
        }

}
    
    
//// add_operation_to_end
// @brief Append an operation to the list of operations
// @param operation An instance of class describing an operation.
void Operation_block::add_operation_to_end( Operation* operation ) {
        
        //set the number of qubit in the operation
        operation->set_qbit_num( qbit_num );
        
        // append the operation to the list
        operations.push_back(operation);
        
        
        // increase the number of parameters by the number of parameters
        parameter_num = parameter_num + operation->get_parameter_num();
        
        // increase the number of layers if necessary
        if (operation->get_type().compare("block")==0) {
            layer_num = layer_num + 1;
        }

}
    
//// add_operation_to_front
// @brief Add an operation to the front of the list of operations
// @param operation A class describing an operation.
 void Operation_block::add_operation_to_front( Operation* operation) {
        
        
        // set the number of qubit in the operation
        operation->set_qbit_num( qbit_num );

        operations.insert( operations.begin(), operation);
            
        // increase the number of U3 gate parameters by the number of parameters
        parameter_num = parameter_num + operation->get_parameter_num();
        
        // increase the number of layers if necessary
        if (operation->get_type().compare("block")==0) {
            layer_num = layer_num + 1;
        }
    
}

            
            
////
// @brief Call to get the number of specific gates in the decomposition
// @return Returns with a dictionary containing the number of specific gates. 
gates_num Operation_block::get_gate_nums() {
        
        gates_num gate_nums;

        gate_nums.u3   = 0;
        gate_nums.cnot = 0;


        for(std::vector<Operation*>::iterator it = operations.begin(); it != operations.end(); ++it) {
            // get the specific operation or block of operations
            Operation* operation = *it;

            if (operation->get_type().compare("block")==0) {
                Operation_block* block_operation = static_cast<Operation_block*>(operation);
                gates_num gate_nums_loc = block_operation->get_gate_nums();
                gate_nums.u3   = gate_nums.u3 + gate_nums_loc.u3;
                gate_nums.cnot = gate_nums.cnot + gate_nums_loc.cnot;
            }
            else if (operation->get_type().compare("u3")==0) {
                gate_nums.u3   = gate_nums.u3 + 1;
            }
            else if (operation->get_type().compare("cnot")==0) {
                gate_nums.cnot   = gate_nums.cnot + 1;
            }

        }
                 
                    
        return gate_nums;
    
}    


//
// @brief Call to get the number of free parameters
// @return Return with the index of the target qubit (return with -1 if target qubit was not set)
int Operation_block::get_parameter_num() {
    return parameter_num;
}


//
// @brief Call to get the number of operations grouped in the block
// @return Return with the number of the operations
int Operation_block::get_operation_num() {
    return operations.size();
}
    
    
////
// @brief Lists the operations decomposing the initial unitary. (These operations are the inverse operations of the operations bringing the intial matrix into unity.)
// @param parameters The parameters of the operations that should be inverted
// @param start_index The index of the first inverse operation
void Operation_block::list_operations( const double* parameters, int start_index ) {
               
        int operation_idx = start_index;        
        int parameter_idx = parameter_num;
        
        for(std::vector<Operation*>::iterator it = operations.end(); it != operations.begin(); --it) {
            string message = "%dth operation:";
            
            Operation* operation = *it;
            
            if (operation->get_type().compare("cnot")==0) {
                CNOT* cnot_operation = static_cast<CNOT*>(operation);

                //message = message + " CNOT with control qubit: " + int_to_string(cnot_operation->get_control_qbit()) + " and target qubit: "  + int_to_string(cnot_operation->get_target_qbit());
                message = message + " CNOT with control qubit: %d and target qubit: %d";
                printf( message.c_str(), operation_idx, cnot_operation->get_control_qbit(), cnot_operation->get_target_qbit() );
                operation_idx = operation_idx + 1;
            }    
            else if (operation->get_type().compare("u3")==0) {

                // definig the U3 parameters
                double vartheta;
                double varphi;
                double varlambda;
                
                // get the inverse parameters of the U3 rotation

                U3* u3_operation = static_cast<U3*>(operation);
                
                if ((u3_operation->get_parameter_num() == 1) && u3_operation->is_theta_parameter()) {
                    vartheta = std::fmod( parameters[parameter_idx-1], 4*M_PI);
                    varphi = 0;
                    varlambda =0;                    
                    parameter_idx = parameter_idx - 1;                    
                    
                }   
                else if ((u3_operation->get_parameter_num() == 1) && u3_operation->is_phi_parameter()) {
                    vartheta = 0;
                    varphi = std::fmod( parameters[ parameter_idx-1 ], 2*M_PI);
                    varlambda =0;                    
                    parameter_idx = parameter_idx - 1;                   
                }    
                else if ((u3_operation->get_parameter_num() == 1) && u3_operation->is_lambda_parameter()) {
                    vartheta = 0;
                    varphi =  0;
                    varlambda = std::fmod( parameters[ parameter_idx-1 ], 2*M_PI);                
                    parameter_idx = parameter_idx - 1;   
                }    
                else if ((u3_operation->get_parameter_num() == 2) && u3_operation->is_theta_parameter() && u3_operation->is_phi_parameter() ) {                  
                    vartheta = std::fmod( parameters[ parameter_idx-2 ], 4*M_PI); 
                    varphi = std::fmod( parameters[ parameter_idx-1 ], 2*M_PI); 
                    varlambda = 0;       
                    parameter_idx = parameter_idx - 2;
                }                
                else if ((u3_operation->get_parameter_num() == 2) && u3_operation->is_theta_parameter() && u3_operation->is_lambda_parameter() ) {                  
                    vartheta = std::fmod( parameters[ parameter_idx-2 ], 4*M_PI);
                    varphi = 0;
                    varlambda = std::fmod( parameters[ parameter_idx-1 ], 2*M_PI);                 
                    parameter_idx = parameter_idx - 2;
                }
                else if ((u3_operation->get_parameter_num() == 2) && u3_operation->is_phi_parameter() && u3_operation->is_lambda_parameter() ) {                  
                    vartheta = 0;
                    varphi = std::fmod( parameters[ parameter_idx-2], 2*M_PI); 
                    varlambda = std::fmod( parameters[ parameter_idx-1 ], 2*M_PI);                 
                    parameter_idx = parameter_idx - 2;
                }    
                else if ((u3_operation->get_parameter_num() == 3)) {                  
                    vartheta = std::fmod( parameters[ parameter_idx-3 ], 4*M_PI); 
                    varphi = std::fmod( parameters[ parameter_idx-2 ], 2*M_PI); 
                    varlambda = std::fmod( parameters[ parameter_idx-1 ], 2*M_PI);                    
                    parameter_idx = parameter_idx - 3;
                }    
                message = message + "U3 on target qubit %d with parameters theta = %f, phi = %f and lambda = %f";
                printf( message.c_str(), operation_idx, u3_operation->get_target_qbit(), vartheta, varphi, varlambda );
                operation_idx = operation_idx + 1;
            }    
            else if (operation->get_type().compare("block")==0) {
                Operation_block* block_operation = static_cast<Operation_block*>(operation);
                const double* parameters_layer = parameters + parameter_idx -operation->get_parameter_num();
                block_operation->list_operations( parameters_layer, operation_idx );   
                parameter_idx = parameter_idx - block_operation->get_parameter_num();
                operation_idx = operation_idx + block_operation->get_operation_num();
            }      
            
        }
            
}
    
    
////
// @brief Call to reorder the qubits in the in the stored operations
// @param qbit_list A list of the permutation of the qubits (for example [1 3 0 2])
void Operation_block::reorder_qubits( vector<int>  qbit_list) {
        
    for(std::vector<Operation*>::iterator it = operations.begin(); it != operations.end(); ++it) {
 
        Operation* operation = *it;
  
        if (operation->get_type().compare("cnot")==0) {
            CNOT* cnot_operation = static_cast<CNOT*>(operation);
            cnot_operation->reorder_qubits( qbit_list );
         }    
         else if (operation->get_type().compare("u3")==0) {
             U3* u3_operation = static_cast<U3*>(operation);
             u3_operation->reorder_qubits( qbit_list );
         }
         else if (operation->get_type().compare("block")==0) {
             Operation_block* block_operation = static_cast<Operation_block*>(operation);
             block_operation->reorder_qubits( qbit_list );
         }       


    }

}



////
// @biref Call to get the involved qubits in the operations stored in the block
// @return Return with an array of the invovled qubits
std::vector<int> Operation_block::get_involved_qubits() {
        
    std::vector<int> involved_qbits;

    int qbit;


    for(std::vector<Operation*>::iterator it = operations.begin(); it != operations.end(); ++it) {

        Operation* operation = *it;

        qbit = operation->get_target_qbit();
        if (qbit != -1) {
            add_unique_elelement( involved_qbits, qbit );
        }


        qbit = operation->get_control_qbit();
        if (qbit != -1) {
            add_unique_elelement( involved_qbits, qbit );
        }

    }

    return involved_qbits;
}


////
// @biref Call to get the operations stored in the block
// @return Return with a vector of Operations.
std::vector<Operation*> Operation_block::get_operations() {
    return operations;
}
    
    
////
// @biref Call to append the operations of an operation bolck to the current block
// @param an instance of class @operation_block
void Operation_block::combine(Operation_block* op_block) {

    // getting the list of operations
    std::vector<Operation*> operations_in = op_block->get_operations();

    for(std::vector<Operation*>::iterator it = (operations_in).begin(); it != (operations_in).end(); ++it) {
        Operation* operation = *it;
        add_operation_to_end(operation);
    }

}
    
    
//// 
// @brief Set the number of qubits spanning the matrix of the operation stored in the block
// @param qbit_num_in The number of qubits spanning the matrix
void Operation_block::set_qbit_num( int qbit_num_in ) {
        
    // setting the number of qubits
    Operation::set_qbit_num(qbit_num_in);

    // setting the number of qubit in the operations
    for(std::vector<Operation*>::iterator it = operations.begin(); it != operations.end(); ++it) {
        Operation* op = *it;

        if (op->get_type().compare("cnot")==0) {
            CNOT* cnot_op = static_cast<CNOT*>( op );
            cnot_op->set_qbit_num( qbit_num_in );         
        }
        else if (op->get_type().compare("u3")==0) {
            U3* u3_op = static_cast<U3*>( op ); 
            u3_op->set_qbit_num( qbit_num_in );               
        }
        else if (op->get_type().compare("block")==0) {
            Operation_block* block_op = static_cast<Operation_block*>( op );          
            block_op->set_qbit_num( qbit_num_in );          
        }
    }
}
         

//
// @brief Create a clone of the present class
// @return Return with a pointer pointing to the cloned object
Operation_block* Operation_block::clone() {

    Operation_block* ret = new Operation_block( qbit_num );
 
    for ( std::vector<Operation*>::iterator it=operations.begin(); it != operations.end(); ++it ) {
        Operation* op = *it;

        if (op->get_type().compare("cnot")==0) {
            CNOT* cnot_op = static_cast<CNOT*>( op );
            CNOT* cnot_op_cloned = cnot_op->clone();
            Operation* op_cloned = static_cast<Operation*>( cnot_op_cloned );
            ret->add_operation_to_end( op_cloned );            
        }
        else if (op->get_type().compare("u3")==0) {
            U3* u3_op = static_cast<U3*>( op );
            U3* u3_op_cloned = u3_op->clone();
            Operation* op_cloned = static_cast<Operation*>( u3_op_cloned );
            ret->add_operation_to_end( op_cloned );            
        }
        else if (op->get_type().compare("block")==0) {
            Operation_block* block_op = static_cast<Operation_block*>( op );
            Operation_block* block_op_cloned = block_op->clone();
            Operation* op_cloned = static_cast<Operation*>( block_op_cloned );
            ret->add_operation_to_end( op_cloned );            
        }
        
    }
 

    return ret;

}


