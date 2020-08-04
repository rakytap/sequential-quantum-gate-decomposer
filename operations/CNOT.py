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
# @brief A class responsible for constructing matrices of CNOT
# gates acting on the N-qubit space

class CNOT( Operation ):
    
    # Pauli x matrix
    pauli_x = np.array([[0,1],[1,0]])
    
    ##
    # @brief Constructor of the class.
    # @param qbit_num The number of qubits in the unitaries
    # @param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
    # @param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
    # @return An instance of the class
    def __init__(self, qbit_num, control_qbit, target_qbit):
        # number of qubits spanning the matrix of the operation
        self.qbit_num = qbit_num
        # A string describing the type of the operation
        self.type = 'cnot'
        # A list of parameter names which are used to evaluate the matrix of the operation
        self.parameters = list()
        
        # The index of the qubit on which the operation acts (target_qbit >= 0) 
        if target_qbit >= qbit_num:
            raise BaseException('target qubit index should be 0<=target_qbit<qbit_num')   
        self.target_qbit = target_qbit
        
        
        # The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
        if control_qbit >= qbit_num:
            raise BaseException('control qubit index should be 0<=target_qbit<qbit_num')   
        self.control_qbit = control_qbit
        
        
        # constructing the matrix of the operation             
        self.matrix = self.composite_cnot( control_qbit, target_qbit )

        #converting the float matrix into complex type
        self.matrix = self.matrix.astype(complex)
        
    ##
    # @brief Sets the number of qubits spanning the matrix of the operation
    # @param qbit_num The number of qubits
    def set_qbit_num( self, qbit_num ):
        # setting the number of qubits
        Operation.set_qbit_num( self, qbit_num )
        
        # recreate the operation matrix
        self.matrix = self.composite_cnot( self.control_qbit, self.target_qbit )
        
        
        
    # @brief Calculate the matrix of a C_NOT gate operation acting on the space of qbit_num qubits.
    # @param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
    # @param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
    # @return Returns with the matrix of the C-NOT gate.
    def composite_cnot(self, control_qubit, target_qubit ):
        
        if control_qubit == target_qubit:
            raise BaseException('control qubit and target qubit should be different')
        
                
        # find base indices where the control qubit is in state |0>
        indexes_control_qubit = None        
        for iidx in range(0,self.qbit_num):
            if iidx == control_qubit:
                if indexes_control_qubit is None:
                    indexes_control_qubit = np.array([1,0])
                else:
                    indexes_control_qubit = np.kron(np.array([1,0]), indexes_control_qubit)
                    
            else:
                if indexes_control_qubit is None:
                    indexes_control_qubit = np.array([1,1])
                else:
                    indexes_control_qubit = np.kron(np.array([1,1]), indexes_control_qubit)
                    
        #print(indexes_control_qubit)                 
        
        ret = None
        for iidx in range(0,self.qbit_num):
            if iidx == target_qubit:
                if ret is None:
                   ret = self.pauli_x
                else:
                    ret = np.kron(self.pauli_x, ret)
            else:
                if ret is None:
                   ret = np.identity(2)
                else:
                    ret = np.kron(np.identity(2), ret)
        
        for index in range(0, len(indexes_control_qubit)):
            if indexes_control_qubit[index] == 1:
                ret[index,:] = 0
                # setting the identity element, when nothing happens to the target qubits
                ret[index,index] = 1
            
        return ret
        
    
    ##
    # @brief Call to reored the qubits in the matrix of the operation
    # @param qbit_list The list of qubits spanning the matrix
    def reorder_qubits( self, qbit_list ):
        
        Operation.reorder_qubits( self, qbit_list )
                
        #% setting the new value for the control qubit
        if not(self.control_qbit is None) :
            self.control_qbit = qbit_list[-self.control_qbit-1]
        
        # recreate the operation matrix
        self.matrix = self.composite_cnot( self.control_qbit, self.target_qbit )
        
    
                   