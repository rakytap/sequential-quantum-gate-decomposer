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
// @brief A base class responsible for constructing matrices of U3
// gates acting on the N-qubit space

#pragma once
#include "Operation.h"
#include <math.h>

using namespace std;

class U3: public Operation {

protected:

// the base indices of the target qubit for state |0>
int* indexes_target_qubit_0;
// the base indices of the target qubit for state |1>
int* indexes_target_qubit_1;
// logical value indicating whether the matrix creation takes an argument theta
bool theta;
// logical value indicating whether the matrix creation takes an argument phi
bool phi;
// logical value indicating whether the matrix creation takes an argument lambda
bool lambda;        
        

public: 
////
// @brief Constructor of the class.
// @param qbit_num The number of qubits in the unitaries
// @param theta_in ...
U3(int, int, bool, bool, bool);

//
// @brief Destructor of the class
~U3();


//
// @brief Call to terive the operation matrix
// @param parameters List of parameters to calculate the matrix of the operation block
// @return Returns with a pointer to the operation matrix
MKL_Complex16* matrix( const double* );

//
// @brief Call to terive the operation matrix
// @param parameters List of parameters to calculate the matrix of the operation block
// @param free_after_used Logical value indicating whether the cteated matrix can be freed after it was used. (For example U3 allocates the matrix on demand, but CNOT is returning with a pointer to the stored matrix in attribute matrix_allocate)
// @return Returns with a pointer to the operation matrix
MKL_Complex16* matrix( const double*, bool& free_after_used );


        
////    
// @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
// @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
MKL_Complex16* composite_u3_Theta_Phi_Lambda(const double* parameters);
        
////    
// @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
// @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
// @return Returns with the matrix of the U3 gate.
MKL_Complex16* composite_u3_Phi_Lambda(const double* parameters);
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* composite_u3_Theta_Lambda(const double* parameters);
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* composite_u3_Theta_Phi(const double* parameters);
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* composite_u3_Lambda(const double* parameters);
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* composite_u3_Phi(const double* parameters);
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* composite_u3_Theta(const double* parameters );
        
        
    ////    
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    // @param Theta Real parameter standing for the parameter theta.
    // @param Phi Real parameter standing for the parameter phi.
    // @param Lambda Real parameter standing for the parameter lambda.
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* composite_u3(double Theta, double Phi, double Lambda );

    ////
    // @brief Determine the base indices corresponding to the target qubit state of |0> and |1>
void get_base_indices();

////
// @brief Sets the number of qubits spanning the matrix of the operation
// @param qbit_num The number of qubits
void set_qbit_num(int qbit_num);



////
// @brief Call to reorder the qubits in the matrix of the operation
// @param qbit_list The list of qubits spanning the matrix
void reorder_qubits( vector<int> qbit_list);

////
// @brief Call to check whethet theta is a free parameter of the gate
// @return Retturns with true if theta is a free parameter of the gate, or false otherwise.
bool is_theta_parameter();

////
// @brief Call to check whethet Phi is a free parameter of the gate
// @return Retturns with true if Phi is a free parameter of the gate, or false otherwise.
bool is_phi_parameter();

////
// @brief Call to check whethet Lambda is a free parameter of the gate
// @return Retturns with true if Lambda is a free parameter of the gate, or false otherwise.
bool is_lambda_parameter();


    
    ////   
    // @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on a single qbit space.
    // @param Theta Real parameter standing for the parameter theta.
    // @param Phi Real parameter standing for the parameter phi.
    // @param Lambda Real parameter standing for the parameter lambda.
    // @return Returns with the matrix of the U3 gate.
MKL_Complex16* one_qubit_u3(double Theta, double Phi, double Lambda );

};

                   
