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

#include "U3.h"


using namespace std;


////
// @brief A class responsible for constructing matrices of U3
// gates acting on the N-qubit space



    ////
    // @brief Constructor of the class.
    // @param qbit_num The number of qubits in the unitaries
    // @param parameter_labels A list of strings 'Theta', 'Phi' or 'Lambda' indicating the free parameters of the U3 operations. (Paremetrs which are not labeled are set to zero)
U3::U3(int qbit_num_in, int target_qbit_in, bool theta_in, bool phi_in, bool lambda_in) {

        // number of qubits spanning the matrix of the operation
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the operation
        type = "u3";
        // The index of the qubit on which the operation acts (target_qbit >= 0) 
        target_qbit = target_qbit_in;
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
        control_qbit = -1;
        // the base indices of the target qubit for state |0>
        indexes_target_qubit_0 = NULL;
        // the base indices of the target qubit for state |1>
        indexes_target_qubit_1 = NULL;

        // logical value indicating whether the matrix creation takes an argument theta
        bool theta = theta_in;
        // logical value indicating whether the matrix creation takes an argument phi
        bool phi = phi_in;
        // logical value indicating whether the matrix creation takes an argument lambda
        bool lambda = lambda_in;        

        // determione the basis indices of the |0> and |1> states of the target qubit
        get_base_indices();

}

////    
// @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
// @param parameters One, Two or Three component array containing the parameters in order Theta, Phi, Lambda
MKL_Complex16* U3::matrix( const double* parameters ) {
        
        
        if (theta && !phi && lambda) {
            // function handle to calculate the operation on the target qubit
            return composite_u3_Theta_Lambda( parameters );
        }

        else if (theta && phi && lambda) {
            // function handle to calculate the operation on the target qubit
            return composite_u3_Theta_Phi_Lambda( parameters );
        }
            
        else if (!theta && phi && lambda) {
            // function handle to calculate the operation on the target qubit
            return composite_u3_Phi_Lambda( parameters );
        }
            
        
           
        else if (theta && phi && !lambda) {
            // function handle to calculate the operation on the target qubit
            return composite_u3_Theta_Phi( parameters );
        }
            
        else if (!theta && !phi && lambda) { 
            // function handle to calculate the operation on the target qubit
            return composite_u3_Lambda( parameters );
        }
             
        else if (!theta && phi && !lambda) {
            // function handle to calculate the operation on the target qubit
            return composite_u3_Phi( parameters );
        }
        
        else if (theta && !phi && !lambda) {
            // function handle to calculate the operation on the target qubit
            return composite_u3_Theta( parameters );
        }
            
        else {
            throw "Input error in the ceration of the operator U3";
        }

}
        
        
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
MKL_Complex16* U3::composite_u3_Theta_Phi_Lambda( const double* parameters ) {
        return composite_u3( parameters[0], parameters[1], parameters[2] );
}
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* U3::composite_u3_Phi_Lambda( const double* parameters ) {
        return composite_u3( 0, parameters[0], parameters[1] );
}
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* U3::composite_u3_Theta_Lambda( const double* parameters ) {
        return composite_u3( parameters[0], 0, parameters[1] );
}
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* U3::composite_u3_Theta_Phi( const double* parameters ){
        return composite_u3( parameters[0], parameters[1], 0 );
}
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* U3::composite_u3_Lambda( const double* parameters ) {
        return composite_u3( 0, 0, parameters[0] );
}
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* U3::composite_u3_Phi( const double* parameters ) {
        return composite_u3( 0, parameters[0], 0 );
}
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* U3::composite_u3_Theta( const double* parameters ) {
        return composite_u3( parameters[0], 0, 0 );
}
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param Theta Real parameter standing for the parameter theta.
    // @param Phi Real parameter standing for the parameter phi.
    // @param Lambda Real parameter standing for the parameter lambda.
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* U3::composite_u3(double Theta, double Phi, double Lambda ) {

        // get the U3 operation of one qubit
        MKL_Complex16* u3_1qbit = one_qubit_u3(Theta, Phi, Lambda );

        // preallocate array for the composite u3 operation
        MKL_Complex16* matrix_array = (MKL_Complex16*)mkl_calloc(matrix_size*matrix_size, sizeof(MKL_Complex16), 64);

        // setting the operation elements
        #pragma omp parallel for
        for(int idx = 0; idx < matrix_size/2; ++idx)
        {
            int element_index;

            // element |0> -> |0>
            element_index = indexes_target_qubit_0[idx]*matrix_size + indexes_target_qubit_0[idx];
            matrix_array[element_index] = u3_1qbit[0];

            // element |1> -> |0>
            element_index = indexes_target_qubit_0[idx]*matrix_size + indexes_target_qubit_1[idx];
            matrix_array[element_index] = u3_1qbit[1];

            // element |0> -> |1>
            element_index = indexes_target_qubit_1[idx]*matrix_size + indexes_target_qubit_0[idx];
            matrix_array[element_index] = u3_1qbit[2];

            // element |1> -> |1>
            element_index = indexes_target_qubit_1[idx]*matrix_size + indexes_target_qubit_1[idx];
            matrix_array[element_index] = u3_1qbit[3];


        }


        // free the allocated single qubit matrix
        mkl_free( u3_1qbit );


        return matrix_array;
}

/*np.identity(2 ** self.qbit_num, dtype=np.complex128)

        u3 = self.u3( Theta, Phi, Lambda )

        matrix[self.indexes_target_qubit['0'], self.indexes_target_qubit['0']] = u3[0,0]
        matrix[self.indexes_target_qubit['0'], self.indexes_target_qubit['1']] = u3[0,1]
        matrix[self.indexes_target_qubit['1'], self.indexes_target_qubit['0']] = u3[1,0]
        matrix[self.indexes_target_qubit['1'], self.indexes_target_qubit['1']] = u3[1,1]

        return matrix

*/

////
// @brief Determine the base indices corresponding to the target qubit state of |0> and |1>
// @return Returns with the matrix of the U3 gate.
void U3::get_base_indices() {
        

        // fre the previously allocated memories
        if ( indexes_target_qubit_1 != NULL ) {
            mkl_free( indexes_target_qubit_1 );
        }
        if ( indexes_target_qubit_0 != NULL ) {
            mkl_free( indexes_target_qubit_0 );
        }

        indexes_target_qubit_1 = (int*)mkl_malloc(matrix_size/2*sizeof(int), 64);
        indexes_target_qubit_0 = (int*)mkl_malloc(matrix_size/2*sizeof(int), 64);

        int target_qbit_power = Power_of_2(target_qbit);
        int indexes_target_qubit_1_idx = 0;
        int indexes_target_qubit_0_idx = 0;      

        // generate the reordered  basis set
        for(int idx = 0; idx<matrix_size; ++idx)
        {
            int bit = int(idx/target_qbit_power) % 2;
            if (bit == 0) {
                indexes_target_qubit_0[indexes_target_qubit_0_idx] = idx;
                indexes_target_qubit_0_idx++;
            }
            else {
                indexes_target_qubit_1[indexes_target_qubit_1_idx] = idx;
                indexes_target_qubit_1_idx++;
            }            
        }

        /*
        // print the result
        for(int idx = 0; idx<matrix_size/2; ++idx) {
            printf ("base indexes for target bit state 0 and 1 are: %d and %d \n", indexes_target_qubit_0[idx], indexes_target_qubit_1[idx]);
        }
        */
}

////
// @brief Sets the number of qubits spanning the matrix of the operation
// @param qbit_num The number of qubits
void U3::set_qbit_num(int qbit_num) {
        // setting the number of qubits
        Operation::set_qbit_num(qbit_num);

        // get the base indices of the target qubit
        get_base_indices();

}



////
// @brief Call to reorder the qubits in the matrix of the operation
// @param qbit_list The list of qubits spanning the matrix
void U3::reorder_qubits( vector<int> qbit_list) {

        Operation::reorder_qubits(qbit_list);

        // get the base indices of the target qubit
        get_base_indices();
}


    
////   
// @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on a single qbit space.
// @param Theta Real parameter standing for the parameter theta.
// @param Phi Real parameter standing for the parameter phi.
// @param Lambda Real parameter standing for the parameter lambda.
// @return Returns with the matrix of the U3 gate.
MKL_Complex16* U3::one_qubit_u3(double Theta, double Phi, double Lambda ) {

    // preallocate array for the composite u3 operation
    MKL_Complex16* matrix_array = (MKL_Complex16*)mkl_calloc(4, sizeof(MKL_Complex16), 64);

    double cos_theta = cos(Theta/2);
    double sin_theta = sin(Theta/2);

    // the 1,1 element
    matrix_array[0].real = cos_theta;
    matrix_array[0].imag = 0;
    // the 1,2 element
    matrix_array[1].real = -cos(Lambda)*sin_theta;
    matrix_array[1].imag = -sin(Lambda)*sin_theta;
    // the 2,1 element
    matrix_array[2].real = cos(Phi)*sin_theta;
    matrix_array[2].imag = sin(Phi)*sin_theta;
    // the 2,2 element
    matrix_array[3].real = cos(Phi+Lambda)*cos_theta;
    matrix_array[3].imag = sin(Phi+Lambda)*cos_theta;

        
    return matrix_array;

}
                   
