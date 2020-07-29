#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:29:39 2020
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

from operations.Operations import Operations
import numpy as np
from functools import reduce


##
# @brief A base class responsible for constructing matrices of C-NOT, U3
# gates acting on the N-qubit space

class operation_block(Operations):
     
##
# @brief Constructor of the class.
# @param qbit_num The number of qubits
# @return An instance of the class
    def __init__( self, qbit_num ):
        
        Operations.__init__( self, qbit_num ) 
        
        # labels of the free parameters
        self.parameters = list()
        
        # logical value. Set true if block is active, false otherwise
        self.active = True
        # The type of the operation
        self.type = 'block'            
     
    
    
    
##
# @brief Call to get the product of the matrices of the operations grouped in the block.
# @param parameters List of parameters to calculate the matrix of the operation block
# @return Returns with the matrix of the operation
    def matrix( self, parameters ) :

        # get the matrices of the operations grouped in the block
        operation_mtxs = self.get_matrices( parameters )

        if len(operation_mtxs) == 0:
            return np.identity(2 ** self.qbit_num, dtype=complex)

        return reduce(np.dot, operation_mtxs)

##
# @brief Call to get the list of matrix representation of the operations grouped in the block.
# @param parameters List of parameters to calculate the matrix of the operation block
# @return Returns with the matrix of the operation
    def get_matrices(self, parameters):

        matrices = []

        # return with identity if not active
        if not self.active:
            return matrices
         
        if len(parameters) != len(self.parameters):
            raise BaseException('Number of parameters should be ' + str(len(self.parameters)) + ' instead of ' + str(len(parameters)) )
         
                
        parameter_idx = 0
        
        
        for idx in range(0,len(self.operations)):
            
            operation = self.operations[idx]
            
            if operation.type == 'cnot':
                operation_mtx = operation.matrix
                
            elif operation.type == 'u3': 
                
                if len(operation.parameters) == 1:
                    operation_mtx = operation.matrix( parameters[parameter_idx] )
                    parameter_idx = parameter_idx + 1
                    
                elif len(operation.parameters) == 2:
                    operation_mtx = operation.matrix( parameters[parameter_idx:parameter_idx+2] )
                    parameter_idx = parameter_idx + 2
                    
                elif len(operation.parameters) == 3:
                    operation_mtx = operation.matrix( parameters[parameter_idx:parameter_idx+3] )
                    parameter_idx = parameter_idx + 3
                else:
                    raise BaseException('The U3 operation has wrong number of parameters')
                                 
            elif operation.type == 'general':
                operation_mtx = operation.matrix


            matrices.append(operation_mtx)
            
        return matrices
        

##
# @biref Call to get the involved qubits in the operations stored in the block
# @return Return with an array of the invovled qubits
    def get_involved_qubits(self):
        
        involved_qbits = list()
        
        for op_idx in range(0,len(self.operations)):
            operation = self.operations[op_idx]
            if not operation.target_qbit is None:
                if not operation.target_qbit in involved_qbits:
                    involved_qbits.append(operation.target_qbit)
                    
            if not operation.control_qbit is None:
                if not operation.control_qbit in involved_qbits:
                    involved_qbits.append(operation.control_qbit)
            
        involved_qbits.sort()
        return np.array(involved_qbits)
    
    
##
# @biref Call to append the operations of an operation bolck to the current block
# @param an instance of class @operation_block
    def combine(self, op_block):
        
        for op_idx in range(0, len(op_block.operations)):
            operation = op_block.operations[op_idx]
            
            self.add_operation_to_end(operation)
    
    
## 
# @brief Set the number of qubits spanning the matrix of the operation stored in the block
# @param qbit_num The number of qubits spanning the matrix
    def set_qbit_num( self, qbit_num ):
        
        self.qbit_num = qbit_num;
        
        # setting the number of qubit in the operations
        for idx in range(0,len(self.operations)):
           self.operations[idx].set_qbit_num( qbit_num )
         
     
      

    
## add_operation_to_end 
# @brief App  an operation to the list of operations
# @param operation A class describing an operation
    def add_operation_to_end (self, operation ):
                
        # set the number of qubit in the operation
        operation.set_qbit_num( self.qbit_num )
        
        
        self.operations.append(operation)
        
        
        # increase the number of U3 gate parameters by the number of parameters
        self.parameter_num = self.parameter_num + len(operation.parameters)
        
        # increase the number of CNOT operations if necessary
        if operation.type == 'block':
            self.layer_num = self.layer_num + 1
         
        
        # adding parameter labels
        if len( operation.parameters ) > 0:
            self.parameters = self.parameters + operation.parameters
         
        
     
    
## add_operation_to_front
# @brief Add an operation to the front of the list of operations
# @param operation A class describing an operation.
    def add_operation_to_front(self, operation):
        
        
        # set the number of qubit in the operation
        operation.set_qbit_num( self.qbit_num )
        
        if len(self.operations) > 0:
            self.operations = [operation] + self.operations
        else:
            self.operations = [operation]
         
            
        # increase the number of U3 gate parameters by the number of parameters
        self.parameter_num = self.parameter_num + len(operation.parameters)   
        
        # increase the number of CNOT operations if necessary
        if operation.type == 'block':
            self.layer_num = self.layer_num + 1;
         
        
        # adding parameter labels    
        if len( operation.parameters ) > 0:
            self.parameters = operation.parameters + self.parameters
         
        
    
    
    
    
    
     
    
 

