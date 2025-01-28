## #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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

"""

## \file qgd_N_Qubit_Decomposition.py
##    \brief A QGD Python interface class for the decomposition of N-qubit unitaries into a set of two-qubit and one-qubit gates.


import numpy as np
from os import path
from squander.variational_quantum_eigensolver.qgd_Variational_Quantum_Eigensolver_Base_Wrapper import qgd_Variational_Quantum_Eigensolver_Base_Wrapper
from squander.gates.qgd_Circuit import qgd_Circuit



##
# @brief A QGD Python interface class for the decomposition of N-qubit unitaries into U3 and CNOT gates.
class qgd_Variational_Quantum_Eigensolver_Base(qgd_Variational_Quantum_Eigensolver_Base_Wrapper):
    
    
## 
# @brief Constructor of the class.
# @param Umtx The unitary matrix to be decomposed.
# @param optimize_layer_num Set true to optimize the minimum number of operation layers required in the decomposition, or false when the predefined maximal number of layer gates is used (ideal for general unitaries).
# @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: "zeros" ,"random", "close_to_zero".
# @return An instance of the class
    def __init__( self, Hamiltonian, qbit_num, config):
    
        # call the constructor of the wrapper class
        super(qgd_Variational_Quantum_Eigensolver_Base, self).__init__(Hamiltonian.data, Hamiltonian.indices, Hamiltonian.indptr, qbit_num, config)
        self.qbit_num = qbit_num


## 
# @brief Call to set the optimizer used in the VQE process
# @param optimizer String indicating the optimizer. Possible values: "BFGS" ,"ADAM", "BFGS2", "ADAM_BATCHED", "AGENTS", "COSINE", "AGENTS_COMBINED".
    def set_Optimizer(self, alg):    

        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Optimizer(alg)

## 
# @brief Call to start solving the VQE problem to get the approximation for the ground state   
    def Start_Optimization(self):

	# call the C wrapper function
        super(qgd_Variational_Quantum_Eigensolver_Base, self).Start_Optimization()

## 
# @brief Call to get the optimized parameters set in numpy array
# @return Returns with the optimized parameters    
    def get_Optimized_Parameters(self):
    
        return super(qgd_Variational_Quantum_Eigensolver_Base, self).get_Optimized_Parameters()


## 
# @brief Call to set the parameters which are used as a starting point in the optimization
# @param A numpy array containing the parameters. The number of parameters can be  retrieved with method get_Parameter_Num        
    def set_Optimized_Parameters(self, new_params):
        
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Optimized_Parameters(new_params)



# TODO should be deleted!        
    def set_Optimization_Tolerance(self, tolerance):
    
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Optimization_Tolerance(tolerance)


##
# @brief Call to set the name of the SQUANDER project
# @param project_name_new new project name      
    def set_Project_Name(self, project_name):
    
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Project_Name(project_name)

##
# @brief Call to set custom layers to the gate structure that are intended to be used in the decomposition from a binary file created from SQUANDER
# @param filename String containing the filename        
    def set_Gate_Structure_from_Binary(self, filename):
    
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Gate_Structure_From_Binary(filename)


##
# @brief Call to set the ansatz type. Currently imp
# @param ansatz_new String of the ansatz . Possible values: "HEA" (hardware efficient ansatz with U3 and CNOT gates).
    def set_Ansatz(self, ansatz_new):
        
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Ansatz(ansatz_new)
        
##
# @brief Call to generate the circuit ansatz
# @param layers The number of layers. The depth of the generated circuit is 2*layers+1 (U3-CNOT-U3-CNOT...CNOT)
# @param inner_blocks The number of U3-CNOT repetition within a single layer
    def Generate_Circuit(self, layers, inner_blocks=1):
    
        super(qgd_Variational_Quantum_Eigensolver_Base, self).Generate_Circuit( layers, inner_blocks )
        
## 
# @brief Call to evaluate the VQE energy.
# @param parameters A float64 numpy array. The number of parameters can be  retrieved with method get_Parameter_Num 
    def Optimization_Problem(self, parameters):
    
        return super(qgd_Variational_Quantum_Eigensolver_Base, self).Optimization_Problem(parameters)


##
# @brief Call to get the second Rényi entropy
# @param parameters A float64 numpy array
# @param input_state A complex array storing the input state. If None |0> is created.
# @param qubit_list A subset of qubits for which the Rényi entropy should be calculated.
# @Return Returns with the calculated entropy
    def get_Second_Renyi_Entropy(self, parameters=None, input_state=None, qubit_list=None ):

        qbit_num = self.get_Qbit_Num()

        qubit_list_validated = list()
        if isinstance(qubit_list, list) or isinstance(qubit_list, tuple):
            for item in qubit_list:
                if isinstance(item, int):
                    qubit_list_validated.append(item)
                    qubit_list_validated = list(set(qubit_list_validated))
                else:
                    print("Elements of qbit_list should be integers")
                    return
        elif qubit_list == None:
            qubit_list_validated = [ x for x in range(qbit_num) ]

        else:
            print("Elements of qbit_list should be integers")
            return
        

        if parameters is None:
            print( "get_Second_Renyi_entropy: array of input parameters is None")
            return None


        if input_state is None:
            matrix_size = 1 << qbit_num
            input_state = np.zeros( (matrix_size,1) )
            input_state[0] = 1

        # evaluate the entropy
        entropy = super(qgd_Variational_Quantum_Eigensolver_Base, self).get_Second_Renyi_Entropy( parameters, input_state, qubit_list_validated)  


        return entropy


##
# @brief Call to get the number of qubits in the circuit
# @return Returns with the number of qubits
    def get_Qbit_Num(self):
    
        return super(qgd_Variational_Quantum_Eigensolver_Base, self).get_Qbit_Num()


##
# @brief Call to get the number of free parameters in the gate structure used for the decomposition
    def get_Parameter_Num( self ):

        return super(qgd_Variational_Quantum_Eigensolver_Base, self).get_Parameter_Num()
        
        

#@brief Call to apply the gate operation on the input state
#@param parameters_mtx Python array ontaining the parameter set
#@param state_to_be_transformed Numpy array storing the state on which the transformation should be applied
    def apply_to( self, parameters_mtx, state_to_be_transformed):

	# call the C wrapper function
        super().apply_to( parameters_mtx, state_to_be_transformed )


##
# @brief Call to retrieve the incorporated quantum circuit (Squander format)
# @return Return with a Qiskit compatible quantum circuit.
    def get_Circuit( self ):
        
        # call the C wrapper function
        return super().get_Circuit()



##
# @brief Export the unitary decomposition into Qiskit format.
# @return Return with a Qiskit compatible quantum circuit.        
    def get_Qiskit_Circuit(self):

        from squander import Qiskit_IO
        
        squander_circuit = self.get_Circuit()
        parameters       = self.get_Optimized_Parameters()
        
        return Qiskit_IO.get_Qiskit_Circuit( squander_circuit, parameters )


##
# @brief Call to get the number of free parameters in the gate structure used for the decomposition
    def set_Initial_State( self, initial_state ):

        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Initial_State( initial_state )


##
# @brief Call to set custom gate structure to used in the decomposition
# @param Gate_structure An instance of SQUANDER Circuit
    def set_Gate_Structure( self, Gate_structure ):  

        if not isinstance(Gate_structure, qgd_Circuit) :
            raise Exception("Input parameter Gate_structure should be a an instance of Circuit")
                    
                    
        return super().set_Gate_Structure( Gate_structure )        
        
            

