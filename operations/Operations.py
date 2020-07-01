# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 22:03:59 2020

@author: rakytap
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
        self.qbit_num = qbit_num;
        
        # reset the list of operations
        self.operations = list();
        
        # the current number of parameters
        self.parameter_num = 0;
                
        # number of operation layers
        self.layer_num = 0;        
        
   
##
# @brief Append a U3 gate to the list of operations
# @param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
# @param 'theta' Logical value. Set to true if parameter theta should be added as a free parameter to the U3 operations, or false (def ault) otherwised. In this case theta = 0 is set.
# @param 'phi' Logical value. Set to true if parameter phi should be added as a free parameter to the U3 operations, or false (def ault) otherwised. In this case phi = 0 is set.
# @param 'lambda' Logical value. Set to true if parameter lambda should be added as a free parameter to the U3 operations, or false (def ault) otherwised. In this case lambda = 0 is set.
    def add_u3_to_end(self, target_qbit, Theta=False, Phi=False, Lambda=False):        
        
        # create the operation
        operation = U3( self.qbit_num, target_qbit, Theta=Theta, Phi=Phi, Lambda=Lambda )
        
        # adding the operation to the end of the list of operations
        self.add_operation_to_end( operation )              
    
##
# @brief Add a U3 gate to the front of the list of operations
# @param target_qbit The identification number of the targt qubit. (0 <= target_qbit <= qbit_num-1)
# @param 'theta' Logical value. Set to true if parameter theta should be added as a free parameter to the U3 operations, or false (def ault) otherwised. In this case theta = 0 is set.
# @param 'phi' Logical value. Set to true if parameter phi should be added as a free parameter to the U3 operations, or false (def ault) otherwised. In this case phi = 0 is set.
# @param 'lambda' Logical value. Set to true if parameter lambda should be added as a free parameter to the U3 operations, or false (def ault) otherwised. In this case lambda = 0 is set.
    def add_u3_to_front(self, target_qbit, Theta=False, Phi=False, Lambda=False):
        
        # create the operation
        operation = U3( self.qbit_num, target_qbit, Theta=Theta, Phi=Phi, Lambda=Lambda );
        
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
    
    

## list_operation_inverses
# @brief Lists the operations decomposing the initial unitary. (These operations are the inverse operations of the operations bringing the intial matrix into unity.)
# @param parameters The parameters of the operations that should be inverted
# @param start_index The index of the first inverse operation
    def list_operation_inverses( self, parameters, start_index = 1 ):
               
        parameter_idx = 0
        operation_idx = start_index
        for idx in range(0,len(self.operations)):
            message = str(operation_idx) + 'th operation:'
            
            operation = self.operations[idx]
           
            if operation.type == 'cnot':
                message = message + ' CNOT with control qubit: ' + str(operation.control_qbit) + ' and target qubit: '  + str(operation.target_qbit)
                operation_idx = operation_idx + 1
                                
            elif operation.type == 'block':
                parameters_layer = parameters[ (parameter_idx): (parameter_idx+operation.parameter_num) ]
                operation.list_operation_inverses( parameters_layer, start_index=operation_idx )
                operation_idx = operation_idx + len(operation.operations)
                parameter_idx = parameter_idx + operation.parameter_num
                continue
               
            elif operation.type == 'u3':
               
                message = message + ' U3 on target qubit: ' + str(operation.target_qbit)
                
                # get the inverse parameters of the U3 rotation
                
                if len(operation.parameters) == 1 and operation.parameters[0] == 'Theta':
                    vartheta = parameters[ parameter_idx ]
                    varphi = np.pi
                    varlambda = np.pi
                    parameter_idx = parameter_idx + 1;
                    
                elif len(operation.parameters) == 1 and operation.parameters[0] == 'Phi':
                    vartheta = 0;
                    varphi = np.pi;
                    varlambda = np.pi-parameters[ parameter_idx ];
                    parameter_idx = parameter_idx + 1;
                    
                elif len(operation.parameters) == 1 and operation.parameters[0] == 'Lambda':
                    vartheta = 0;
                    varphi = np.pi-parameters[ parameter_idx ]
                    varlambda = np.pi;
                    parameter_idx = parameter_idx + 1
                    
                elif len(operation.parameters) == 2 and 'Theta' in operation.parameters and 'Phi' in operation.parameters:
                    vartheta = parameters[ parameter_idx ]
                    varphi = np.pi;
                    varlambda = np.pi-parameters[ parameter_idx+1 ];
                    parameter_idx = parameter_idx + 2
                    
                elif len(operation.parameters) == 2 and 'Theta' in operation.parameters and 'Lambda' in operation.parameters:
                    vartheta = parameters[ parameter_idx ]
                    varphi = np.pi-parameters[ parameter_idx ]
                    varlambda = np.pi;
                    parameter_idx = parameter_idx + 2
                    
                elif len(operation.parameters) == 2 and 'Phi' in operation.parameters and 'Lambda' in operation.parameters :
                    vartheta = 0;
                    varphi = np.pi-parameters[ parameter_idx ]
                    varlambda = np.pi-parameters[ parameter_idx+1 ]
                    parameter_idx = parameter_idx + 2
                    
                elif len(operation.parameters) == 3 and 'Theta' in operation.parameters and 'Phi' in operation.parameters and 'Lambda' in operation.parameters :
                    vartheta = parameters[ parameter_idx ]
                    varphi = np.pi-parameters[ parameter_idx+2 ]
                    varlambda = np.pi-parameters[ parameter_idx+1 ]
                    parameter_idx = parameter_idx + 3
                
                
                
                message = message + ' with parameters theta = ' + str(vartheta) + ', phi = ' + str(varphi) + ' and lambda = ' + str(varlambda)
                operation_idx = operation_idx + 1
                
            
            print( message )
            
            
            
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
# @brief Call to contruct Qiskit compatible quantum circuit from the operations
    def get_quantum_circuit(self, parameters, circuit=None):
        
        parameter_idx = 0
            
        if circuit is None:
            circuit = QuantumCircuit(self.qbit_num)
            
            
        # adding the operations to the circuit
        for idx in range(0, len(self.operations)):
            
            operation = self.operations[idx]
           
            if operation.type == 'cnot':
                circuit.cx(operation.control_qbit, operation.target_qbit)
                                
            elif operation.type == 'block':
                parameters_layer = parameters[ (parameter_idx): (parameter_idx+operation.parameter_num) ]
                circuit = operation.get_quantum_circuit( parameters_layer, circuit=circuit )
                parameter_idx = parameter_idx + operation.parameter_num
                continue
               
            elif operation.type == 'u3':
                
                # get the inverse parameters of the U3 rotation
                
                if len(operation.parameters) == 1 and operation.parameters[0] == 'Theta':
                    vartheta = parameters[ parameter_idx ]
                    varphi = np.pi
                    varlambda = np.pi
                    parameter_idx = parameter_idx + 1;
                    
                elif len(operation.parameters) == 1 and operation.parameters[0] == 'Phi':
                    vartheta = 0;
                    varphi = np.pi;
                    varlambda = np.pi-parameters[ parameter_idx ];
                    parameter_idx = parameter_idx + 1;
                    
                elif len(operation.parameters) == 1 and operation.parameters[0] == 'Lambda':
                    vartheta = 0;
                    varphi = np.pi-parameters[ parameter_idx ]
                    varlambda = np.pi;
                    parameter_idx = parameter_idx + 1
                    
                elif len(operation.parameters) == 2 and 'Theta' in operation.parameters and 'Phi' in operation.parameters:
                    vartheta = parameters[ parameter_idx ]
                    varphi = np.pi;
                    varlambda = np.pi-parameters[ parameter_idx+1 ];
                    parameter_idx = parameter_idx + 2
                
                elif len(operation.parameters) == 2 and 'Theta' in operation.parameters and 'Lambda' in operation.parameters:
                    vartheta = parameters[ parameter_idx ]
                    varphi = np.pi-parameters[ parameter_idx ]
                    varlambda = np.pi;
                    parameter_idx = parameter_idx + 2
                
                elif len(operation.parameters) == 2 and 'Phi' in operation.parameters and 'Lambda' in operation.parameters :
                    vartheta = 0;
                    varphi = np.pi-parameters[ parameter_idx ]
                    varlambda = np.pi-parameters[ parameter_idx+1 ]
                    parameter_idx = parameter_idx + 2
                
                elif len(operation.parameters) == 3 and 'Theta' in operation.parameters and 'Phi' in operation.parameters and 'Lambda' in operation.parameters :
                    vartheta = parameters[ parameter_idx ]
                    varphi = np.pi-parameters[ parameter_idx+2 ]
                    varlambda = np.pi-parameters[ parameter_idx+1 ]
                    parameter_idx = parameter_idx + 3
                    
                circuit.u3(vartheta, varphi, varlambda, operation.target_qbit)
            
                    
            
        return circuit
        
    