# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:12:18 2020
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
"""

from .Operation import Operation
import numpy as np

##
# @brief A class responsible for constructing matrices of U3
# gates acting on the N-qubit space

class U3( Operation ):

    ##
    # @brief Constructor of the class.
    # @param qbit_num The number of qubits in the unitaries
    # @param parameter_labels A list of strings 'Theta', 'Phi' or 'Lambda' indicating the free parameters of the U3 operations. (Paremetrs which are not labeled are set to zero)
    def __init__(self, qbit_num, target_qbit, parameter_labels):
        # number of qubits spanning the matrix of the operation
        self.qbit_num = qbit_num
        # A string describing the type of the operation
        self.type = 'u3'
        # A list of parameter names which are used to evaluate the matrix of the operation
        self.parameters = list()
        # The index of the qubit on which the operation acts (target_qbit >= 0) 
        self.target_qbit = target_qbit
        # The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
        self.control_qbit = None
        # the base indices of the target qubit
        self.target_qbit_indices = None
        # the preallocated memory for the matrix function
        self.matrix_prealloc = np.identity( 2**self.qbit_num, dtype=np.complex128)

        # determione the basis indices of the |0> and |1> states of the target qubit
        self.get_base_indices()
        
        if 'Theta' in parameter_labels and 'Phi' in parameter_labels and 'Lambda' in parameter_labels :
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Theta_Phi_Lambda
            self.parameters = ["Theta", "Phi", "Lambda"]
            
        elif not ('Theta' in parameter_labels) and 'Phi' in parameter_labels and 'Lambda' in parameter_labels:
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Phi_Lambda
            self.parameters = ["phi", "Lambda"]
            
        elif 'Theta' in parameter_labels and (not ('Phi' in parameter_labels)) and 'Lambda' in parameter_labels :
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Theta_Lambda
            self.parameters = ["Theta", "Lambda"]
           
        elif 'Theta' in parameter_labels and 'Phi' in parameter_labels and (not ('Lambda' in parameter_labels))  :          
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Theta_Phi
            self.parameters = ["Theta", "Phi"]
            
        elif (not ('Theta' in parameter_labels)) and (not ('Phi' in parameter_labels) ) and 'Lambda' in parameter_labels:
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Lambda
            self.parameters = ["Lambda"]
             
        elif (not ('Theta' in parameter_labels)) and 'Phi' in parameter_labels and (not ('Lambda' in parameter_labels)):
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Phi
            self.parameters = ["Phi"]
        
        elif 'Theta' in parameter_labels and (not ('Phi' in parameter_labels)) and (not ('Lambda' in parameter_labels)):
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Theta
            self.parameters = ["Theta"]
            
        else:
            raise BaseException('Input error in the ceration of the operator U3')
        
        
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    def composite_u3_Theta_Phi_Lambda(self, parameters ):
        return self.composite_u3( Theta = parameters[0], Phi = parameters[1], Lambda = parameters[2] )
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    # @return Returns with the matrix of the U3 gate.
    def composite_u3_Phi_Lambda(self, parameters ):
        return self.composite_u3( Theta = 0, Phi = parameters[0], Lambda = parameters[1] )
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    # @return Returns with the matrix of the U3 gate.
    def composite_u3_Theta_Lambda(self, parameters ):
        return self.composite_u3( Theta = parameters[0], Phi = 0, Lambda = parameters[1] )
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    # @return Returns with the matrix of the U3 gate.
    def composite_u3_Theta_Phi(self, parameters ):
        return self.composite_u3( Theta = parameters[0], Phi = parameters[1], Lambda = 0 )
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    # @return Returns with the matrix of the U3 gate.
    def composite_u3_Lambda(self, parameters ):
        return self.composite_u3( Theta = 0, Phi = 0, Lambda = parameters[0] )
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    # @return Returns with the matrix of the U3 gate.
    def composite_u3_Phi(self, parameters ):
        return self.composite_u3( Theta = 0, Phi = parameters[0], Lambda = 0 )
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array containing the parameters in order Theta, Phi, Lambda
    # @return Returns with the matrix of the U3 gate.
    def composite_u3_Theta(self, parameters ):
        return self.composite_u3( Theta = parameters[0], Phi = 0, Lambda = 0 )
        
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param Theta Real parameter standing for the parameter theta.
    # @param Phi Real parameter standing for the parameter phi.
    # @param Lambda Real parameter standing for the parameter lambda.
    # @return Returns with the matrix of the U3 gate.
    def composite_u3(self, Theta, Phi, Lambda ):

        u3 = self.u3( Theta, Phi, Lambda )

        self.matrix_prealloc[self.indexes_target_qubit['0'], self.indexes_target_qubit['0']] = u3[0,0]
        self.matrix_prealloc[self.indexes_target_qubit['0'], self.indexes_target_qubit['1']] = u3[0,1]
        self.matrix_prealloc[self.indexes_target_qubit['1'], self.indexes_target_qubit['0']] = u3[1,0]
        self.matrix_prealloc[self.indexes_target_qubit['1'], self.indexes_target_qubit['1']] = u3[1,1]

        return self.matrix_prealloc

    ##
    # @brief Determine the base indices corresponding to the target qubit state of |0> and |1>
    # @return Returns with the matrix of the U3 gate.
    def get_base_indices(self):
        
        number_of_basis = 2**self.qbit_num

        indexes_target_qubit_1 = list()
        indexes_target_qubit_0 = list()

        # generate the reordered  basis set
        for idx in range(0, 2 ** self.qbit_num):
            state = bin(idx)
            state = state[2:].zfill(self.qbit_num)
            if state[-self.target_qbit-1] == '0':
                indexes_target_qubit_0.append(idx)
            else:
                indexes_target_qubit_1.append(idx)

        self.indexes_target_qubit = {'0':indexes_target_qubit_0, '1':indexes_target_qubit_1}
        #print(indexes_target_qubit_0)
        #print(indexes_target_qubit_1)

    ##
    # @brief Sets the number of qubits spanning the matrix of the operation
    # @param qbit_num The number of qubits
    def set_qbit_num(self, qbit_num):
        # setting the number of qubits
        Operation.set_qbit_num(self, qbit_num)

        # get the base indices of the target qubit
        self.get_base_indices()

        # preallocate array for the u3 operation
        self.matrix_prealloc = np.identity(2 ** self.qbit_num, dtype=np.complex128)

    ##
    # @brief Call to reorder the qubits in the matrix of the operation
    # @param qbit_list The list of qubits spanning the matrix
    def reorder_qubits(self, qbit_list):

        Operation.reorder_qubits(self, qbit_list)

        # get the base indices of the target qubit
        self.get_base_indices()

        # preallocate array for the u3 operation
        self.matrix_prealloc = np.identity(2 ** self.qbit_num, dtype=np.complex128)


    
    ##   
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on a single qbit space.
    # @param Theta Real parameter standing for the parameter theta.
    # @param Phi Real parameter standing for the parameter phi.
    # @param Lambda Real parameter standing for the parameter lambda.
    # @return Returns with the matrix of the U3 gate.
    @staticmethod
    def u3(Theta=0, Phi=0, Lambda=0 ):
        
        return np.array([[np.cos(Theta/2), -np.exp(np.complex(0,Lambda))*np.sin(Theta/2)],
                        [np.exp(np.complex(0,Phi))*np.sin(Theta/2), np.exp(np.complex(0,Lambda+Phi))*np.cos(Theta/2)]])
                   