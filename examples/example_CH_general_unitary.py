# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
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
## \file example_CH.py
## \brief Example to use controlled Hadamard gates in the decomposition

from squander import N_Qubit_Decomposition 
from squander import Circuit

# cerate unitary q-bit matrix
from scipy.stats import unitary_group
from squander import utils
import numpy as np

    


def create_custom_gate_structure(qbit_num):
    # creating an instance of the wrapper class Circuit
    Circuit_ret = Circuit( qbit_num )

    disentangle_qbit = qbit_num - 1 

    for qbit in range(0, disentangle_qbit ):

        # creating an instance of the wrapper class Circuit
        layer = Circuit( qbit_num )


        # add U3 fate to the block
        Theta = True
        Phi = False
        Lambda = True      
        layer.add_U3( qbit, Theta, Phi, Lambda )    
        layer.add_U3( disentangle_qbit, Theta, Phi, Lambda )  


        # Connecting every second target qubit with CNOT gate, and use CH gate otherwise
        if ( qbit % 2 == 0 ):
            # add CH gate to the block
            layer.add_CH( qbit, disentangle_qbit  )
        else:
            # add CNOT gate to the block
            layer.add_CNOT( qbit, disentangle_qbit  )



        Circuit_ret.add_Circuit( layer )

    return Circuit_ret





# the number of qubits spanning the unitary
qbit_num = 2

# determine the soze of the unitary to be decomposed
matrix_size = int(2**qbit_num)
   
# creating a random unitary to be decomposed
Umtx = unitary_group.rvs(matrix_size)

# creating an instance of the C++ class
decomp = N_Qubit_Decomposition( Umtx.conj().T )


# create custom gate structure
gate_structure = { 4: create_custom_gate_structure(4), 2: create_custom_gate_structure(2)}        


# adding custom gate structure to the decomposition
decomp.set_Gate_Structure( gate_structure )

# set the number of block to be optimized in one shot
decomp.set_Optimization_Blocks( 20 )

# starting the decomposition
decomp.Start_Decomposition()

# get the decomposing operations
quantum_circuit = decomp.get_Quantum_Circuit()


# print the quantum circuit
print(quantum_circuit)

import numpy.linalg as LA
    
# the unitary matrix from the result object
decomposed_matrix = utils.get_unitary_from_qiskit_circuit( quantum_circuit )
product_matrix = np.dot(Umtx,decomposed_matrix.conj().T)
phase = np.angle(product_matrix[0,0])
product_matrix = product_matrix*np.exp(-1j*phase)
    
product_matrix = np.eye(matrix_size)*2 - product_matrix - product_matrix.conj().T
# the error of the decomposition
decomposition_error =  (np.real(np.trace(product_matrix)))/2
       
print('The error of the decomposition is ' + str(decomposition_error))

from qiskit import visualization
visualization.circuit_drawer(quantum_circuit, output="latex_source", interactive=True, filename="cirquit_export", fold=10)


