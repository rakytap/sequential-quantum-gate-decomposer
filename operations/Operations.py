# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 22:03:59 2020
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

import numpy as np
from .CNOT import CNOT
from .U3 import U3
from .Operation import Operation

from qiskit import QuantumCircuit


##
# @brief A class to group quantum gate operations into layers (or blocks)

class  Operations():
 
     
    
## Contructor of the class
# @brief Constructor of the class.
# @param qbit_num The number of qubits in the unitaries
# @return An instance of the class
    def __init__(self, qbit_num):   
        
        # setting the number of qubits
        self.qbit_num = qbit_num
        
        # reset the list of operations
        self.operations = list()
        
        # the current number of parameters
        self.parameter_num = 0
                
        # number of operation layers
        self.layer_num = 0
        
   
##
# @brief Append a U3 gate to the list of operations
# @param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
# @param parameter_labels A list of strings 'Theta', 'Phi' or 'Lambda' indicating the free parameters of the U3 operations. (Paremetrs which are not labeled are set to zero)
    def add_u3_to_end(self, target_qbit, parameter_labels):        
        
        # create the operation
        operation = U3( self.qbit_num, target_qbit, parameter_labels )
        
        # adding the operation to the end of the list of operations
        self.add_operation_to_end( operation )              
    
##
# @brief Add a U3 gate to the front of the list of operations
# @param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
# @param parameter_labels A list of strings 'Theta', 'Phi' or 'Lambda' indicating the free parameters of the U3 operations. (Paremetrs which are not labeled are set to zero)
    def add_u3_to_front(self, target_qbit, parameter_labels):
        
        # create the operation
        operation = U3( self.qbit_num, target_qbit, parameter_labels )
        
        # adding the operation to the front of the list of operations
        self.add_operation_to_front( operation )
        
## 
# @brief Append a C_NOT gate operation to the list of operations
# @param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
# @param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
    def add_cnot_to_end(self, control_qbit, target_qbit) :  
        
        # new cnot operation
        operation = CNOT(self.qbit_num, control_qbit, target_qbit )
        
        # append the operation to the list
        self.add_operation_to_end(operation)       
        
        
    
## add_cnot_to_front
# @brief Add a C_NOT gate operation to the front of the list of operations
# @param control_qbit The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
# @param target_qbit The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
    def add_cnot_to_front(self, control_qbit, target_qbit):
        
        # new cnot operation
        operation = CNOT(self.qbit_num, control_qbit, target_qbit )
        
        # put the operation to tghe front of the list
        self.add_operation_to_front(operation)        
        
    
##
# @brief Append an array of operations to the list of operations
# @param operations A list of operation class instances.
    def add_operations_to_end(self, operations):
        
        for idx in range(0,len( operations )):
            self.add_operation_to_end( operations[idx] )
            
    
## add_operations_to_front
# @brief Add an array of operations to the front of the list of operations
# @param operations A list of operation class instances.
    def add_operations_to_front(self, operations):
        
        # adding operations in reversed order!!
        for idx  in range(len( operations )-1,0,-1):
            self.add_operation_to_end( operations(idx) )     
    
    
## add_operation_to_end
# @brief Append an operation to the list of operations
# @param operation An instance of class describing an operation.
    def add_operation_to_end(self, operation ):
                
        #set the number of qubit in the operation
        operation.set_qbit_num( self.qbit_num )
        
        # append the operation to the list
        self.operations.append(operation)
        
        
        # increase the number of parameters by the number of parameters
        self.parameter_num = self.parameter_num + len(operation.parameters)
        
        # increase the number of layers if necessary
        if operation.type == 'block' :
            self.layer_num = self.layer_num + 1
    
## add_operation_to_front
# @brief Add an operation to the front of the list of operations
# @param operation A class describing an operation.
    def add_operation_to_front(self, operation):
        
        
        # set the number of qubit in the operation
        operation.set_qbit_num( self.qbit_num );
        
        if len(self.operations) > 0:
            self.operations = [operation] + self.operations
        else:
            self.operations.append(operation)
            
        # increase the number of U3 gate parameters by the number of parameters
        self.parameter_num = self.parameter_num + len(operation.parameters)
        
        # increase the number of CNOT operations if necessary
        if operation.type == 'block':
            self.layer_num = self.layer_num + 1
    
    

            
            
##
# @brief Call to get the number of specific gates in the decomposition
# @return Returns with a dictionary containing the number of specific gates. 
    def get_gate_nums(self): 
        
        gate_nums = dict()
        
        
        for operation_idx in range(0,len(self.operations)):
            
            # get the specific operation or block of operations
            operation = self.operations[operation_idx]
            if operation.type == 'block':
                
                gate_nums_loc = operation.get_gate_nums()
                for key in gate_nums_loc.keys():
                    gate_nums[key] = gate_nums.get(key, 0) + gate_nums_loc[key]
                    
            elif operation.type == 'u3':
                gate_nums['u3'] = gate_nums.get('u3', 0) + 1
            elif operation.type == 'cnot':
                gate_nums['cnot'] = gate_nums.get('cnot', 0) + 1
                
        return gate_nums
    
    
    
    
##
# @brief Lists the operations decomposing the initial unitary. (These operations are the inverse operations of the operations bringing the intial matrix into unity.)
# @param parameters The parameters of the operations that should be inverted
# @param start_index The index of the first inverse operation
    def list_operations( self, parameters, start_index = 1 ):
               
        operation_idx = start_index        
        parameter_idx = len(parameters)
        
        for idx in range(len(self.operations)-1,-1,-1):
            message = str(operation_idx) + 'th operation:'
            
            operation = self.operations[idx]
            
            if operation.type == 'cnot':
                message = message + ' CNOT with control qubit: ' + str(operation.control_qbit) + ' and target qubit: '  + str(operation.target_qbit)
                operation_idx = operation_idx + 1
                
            elif operation.type == 'u3':
                
                # get the inverse parameters of the U3 rotation
                
                if len(operation.parameters) == 1 and operation.parameters[0] == 'Theta':
                    vartheta = parameters[ parameter_idx-1 ]  % (4*np.pi) 
                    varphi = 0
                    varlambda =0
                    
                    parameter_idx = parameter_idx - 1                    
                    
                    
                elif len(operation.parameters) == 1 and operation.parameters[0] == 'Phi':
                    vartheta = 0
                    varphi = parameters[ parameter_idx-1 ] % (2*np.pi) 
                    varlambda =0
                    
                    parameter_idx = parameter_idx - 1                    
                    
                elif len(operation.parameters) == 1 and operation.parameters[0] == 'Lambda':
                    vartheta = 0
                    varphi =  0
                    varlambda =parameters[ parameter_idx-1 ]    % (2*np.pi)                
                    parameter_idx = parameter_idx - 1   
                    
                elif len(operation.parameters) == 2 and 'Theta' in operation.parameters and 'Phi' in operation.parameters:                    
                    vartheta = parameters[ parameter_idx-2 ]  % (4*np.pi) 
                    varphi = parameters[ parameter_idx-1 ]  % (2*np.pi) 
                    varlambda = 0       
                    parameter_idx = parameter_idx - 2
                    
                
                elif len(operation.parameters) == 2 and 'Theta' in operation.parameters and 'Lambda' in operation.parameters:
                    vartheta = parameters[ parameter_idx-2 ]  % (4*np.pi) 
                    varphi = 0
                    varlambda = parameters[ parameter_idx-1 ]  % (2*np.pi)                 
                    parameter_idx = parameter_idx - 2
                
                elif len(operation.parameters) == 2 and 'Phi' in operation.parameters and 'Lambda' in operation.parameters :
                    vartheta = 0
                    varphi = parameters[ parameter_idx-2] % (2*np.pi) 
                    varlambda = parameters[ parameter_idx-1 ]   % (2*np.pi)                  
                    parameter_idx = parameter_idx - 2
                    
                elif len(operation.parameters) == 3 and 'Theta' in operation.parameters and 'Phi' in operation.parameters and 'Lambda' in operation.parameters :
                    vartheta = parameters[ parameter_idx-3 ]  % (4*np.pi) 
                    varphi = parameters[ parameter_idx-2 ]  % (2*np.pi) 
                    varlambda = parameters[ parameter_idx-1 ]   % (2*np.pi)                    
                    parameter_idx = parameter_idx - 3
                    
                message = message + ' with parameters theta = ' + str(vartheta) + ', phi = ' + str(varphi) + ' and lambda = ' + str(varlambda)
                operation_idx = operation_idx + 1 
                
            elif operation.type == 'block':
                parameters_layer = parameters[ (parameter_idx-operation.parameter_num): (parameter_idx) ]            
                operation.list_operations( parameters_layer, start_index = start_index )   
                parameter_idx = parameter_idx - operation.parameter_num
                operation_idx = operation_idx + len(operation.operations)                
                continue
                    
            
            print( message )
            
            
##
# @brief Call to contruct Qiskit compatible quantum circuit from the operations reproducing the initial unitary.
# @param parameters Array of parameters corresponding to the U3 operation
# @param circuit Qiskit circut. Optional parameter
    def get_quantum_circuit_inverse(self, parameters, circuit=None):
        
        if circuit is None:
            circuit = QuantumCircuit(self.qbit_num)
            
        parameter_idx = 0
        
        
        for idx in range(0,len(self.operations)):
            
            operation = self.operations[idx]
            
            if operation.type == 'cnot':
                circuit.cx(operation.control_qbit, operation.target_qbit)
                
            elif operation.type == 'u3':
                
                # get the inverse parameters of the U3 rotation
                
                if len(operation.parameters) == 1 and operation.parameters[0] == 'Theta':
                    vartheta = parameters[ parameter_idx ]  % (4*np.pi) 
                    varphi = np.pi
                    varlambda = np.pi
                    
                    parameter_idx = parameter_idx + 1                    
                    
                    
                elif len(operation.parameters) == 1 and operation.parameters[0] == 'Phi':
                    vartheta = 0
                    varphi = np.pi
                    varlambda = (np.pi- parameters[ parameter_idx ])  % (2*np.pi) 
                    
                    parameter_idx = parameter_idx + 1                    
                    
                elif len(operation.parameters) == 1 and operation.parameters[0] == 'Lambda':
                    vartheta = 0
                    varphi =  (np.pi-parameters[ parameter_idx ])  % (2*np.pi) 
                    varlambda = np.pi
                    
                    parameter_idx = parameter_idx + 1   
                    
                elif len(operation.parameters) == 2 and 'Theta' in operation.parameters and 'Phi' in operation.parameters:                    
                    vartheta = parameters[ parameter_idx ]  % (4*np.pi) 
                    varphi = np.pi
                    varlambda = (np.pi-parameters[ parameter_idx+1 ])  % (2*np.pi) 
                    
                    parameter_idx = parameter_idx + 2
                    
                
                elif len(operation.parameters) == 2 and 'Theta' in operation.parameters and 'Lambda' in operation.parameters:
                    vartheta = parameters[ parameter_idx ] % (4*np.pi) 
                    varphi = (np.pi-parameters[ parameter_idx+1 ]) % (2*np.pi) 
                    varlambda = np.pi
                    
                    parameter_idx = parameter_idx + 2
                
                elif len(operation.parameters) == 2 and 'Phi' in operation.parameters and 'Lambda' in operation.parameters :
                    vartheta = 0
                    varphi = (np.pi-parameters[ parameter_idx+1 ])  % (2*np.pi) 
                    varlambda = (np.pi-parameters[ parameter_idx ])  % (2*np.pi)  
                    
                    parameter_idx = parameter_idx + 2
                    
                elif len(operation.parameters) == 3 and 'Theta' in operation.parameters and 'Phi' in operation.parameters and 'Lambda' in operation.parameters :
                    vartheta = parameters[ parameter_idx ]  % 4*np.pi
                    varphi = (np.pi-parameters[ parameter_idx+2 ])  % (2*np.pi) 
                    varlambda = (np.pi-parameters[ parameter_idx+1 ])  % (2*np.pi) 
                    
                    parameter_idx = parameter_idx + 3
                    
                    
                circuit.u3(vartheta, varphi, varlambda, operation.target_qbit)
                
            elif operation.type == 'block':
                parameters_layer = parameters[ (parameter_idx): (parameter_idx+operation.parameter_num) ]            
                circuit = operation.get_quantum_circuit_inverse( parameters_layer, circuit=circuit )   
                parameter_idx = parameter_idx + operation.parameter_num
                
                continue
                    
            
            
            
            
            
        
            
        return circuit

           

##
# @brief Call to contruct Qiskit compatible quantum circuit from the operations that brings the original unitary into identity
# @param parameters Array of parameters corresponding to the U3 operation
# @param circuit Qiskit circut. Optional parameter
    def get_quantum_circuit(self, parameters, circuit=None):
        
        if circuit is None:
            circuit = QuantumCircuit(self.qbit_num)
            
        parameter_idx = len(parameters)
        
        #initial_matrix = np.identity(2**self.qbit_num)
        
        
        for idx in range(len(self.operations)-1,-1,-1):
            
            operation = self.operations[idx]
            
            if operation.type == 'cnot':
                #operation_mtx = operation.matrix
                circuit.cx(operation.control_qbit, operation.target_qbit)
                
            elif operation.type == 'u3':
                
                # get the inverse parameters of the U3 rotation
                
                if len(operation.parameters) == 1 and operation.parameters[0] == 'Theta':
                    #operation_mtx = operation.matrix( parameters[parameter_idx-1] )
                    vartheta = parameters[ parameter_idx-1 ] % (4*np.pi)  
                    varphi = 0
                    varlambda =0
                    
                    parameter_idx = parameter_idx - 1                    
                    
                    
                elif len(operation.parameters) == 1 and operation.parameters[0] == 'Phi':
                    #operation_mtx = operation.matrix( parameters[parameter_idx-1] )
                    vartheta = 0
                    varphi = parameters[ parameter_idx-1 ]     % (2*np.pi)  
                    varlambda =0
                    
                    parameter_idx = parameter_idx - 1                    
                    
                elif len(operation.parameters) == 1 and operation.parameters[0] == 'Lambda':
                    #operation_mtx = operation.matrix( parameters[parameter_idx-1] )
                    vartheta = 0
                    varphi =  0
                    varlambda =parameters[ parameter_idx-1 ]    % (2*np.pi)
                    parameter_idx = parameter_idx - 1   
                    
                elif len(operation.parameters) == 2 and 'Theta' in operation.parameters and 'Phi' in operation.parameters:
                    #operation_mtx = operation.matrix( parameters[parameter_idx-2:parameter_idx] )                  
                    
                    vartheta = parameters[ parameter_idx-2 ] % (4*np.pi)  
                    varphi = parameters[ parameter_idx-1 ]     % (2*np.pi)  
                    varlambda = 0                  
                    parameter_idx = parameter_idx - 2
                    
                
                elif len(operation.parameters) == 2 and 'Theta' in operation.parameters and 'Lambda' in operation.parameters:
                    #operation_mtx = operation.matrix( parameters[parameter_idx-2:parameter_idx] )                    
                    vartheta = parameters[ parameter_idx-2 ]  % (4*np.pi)  
                    varphi = 0
                    varlambda = parameters[ parameter_idx-1 ]    % (2*np.pi)    
                    parameter_idx = parameter_idx - 2
                
                elif len(operation.parameters) == 2 and 'Phi' in operation.parameters and 'Lambda' in operation.parameters :
                    #operation_mtx = operation.matrix( parameters[parameter_idx-2:parameter_idx] )
                    vartheta = 0
                    varphi = parameters[ parameter_idx-2]    % (2*np.pi)  
                    varlambda = parameters[ parameter_idx-1 ]    % (2*np.pi)
                    
                    parameter_idx = parameter_idx - 2
                    
                elif len(operation.parameters) == 3 and 'Theta' in operation.parameters and 'Phi' in operation.parameters and 'Lambda' in operation.parameters :
                    #operation_mtx = operation.matrix( parameters[parameter_idx-3:parameter_idx] )
                    vartheta = parameters[ parameter_idx-3 ]  % (4*np.pi)  
                    varphi = parameters[ parameter_idx-2 ]    % (2*np.pi)  
                    varlambda = parameters[ parameter_idx-1 ]    % (2*np.pi)           
                    parameter_idx = parameter_idx - 3
                    
                    
                circuit.u3(vartheta, varphi, varlambda, operation.target_qbit)
                
            elif operation.type == 'block':
                #parameters_num = len(operation.parameters)
                #operation_mtx = operation.matrix( parameters[parameter_idx-parameters_num:parameter_idx] )   
                parameters_layer = parameters[ (parameter_idx-operation.parameter_num): (parameter_idx) ]            
                circuit = operation.get_quantum_circuit( parameters_layer, circuit=circuit )   
                parameter_idx = parameter_idx - operation.parameter_num
                
                continue
                    
            
            #initial_matrix = np.dot( operation_mtx, initial_matrix )
            
            
        return circuit
    
    
##
# @brief Call to reorder the qubits in the in the stored operations
# @param qbit_list A list of the permutation of the qubits (for example [1 3 0 2])
    def reorder_qubits(self, qbit_list):  
        
        for operation_idx in range(0, len(self.operations)):
            self.operations[operation_idx].reorder_qubits( qbit_list )
        
    