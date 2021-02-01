## #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:44:26 2020
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

## \file qgd_N_Qubit_Decomposition.py
##    \brief A QGD Python interface class for the decomposition of N-qubit unitaries into U3 and CNOT gates.


import ctypes
import numpy as np
from os import path
from qgd_python.qgd_N_Qubit_Decomposition_Wrapper import qgd_N_Qubit_Decomposition_Wrapper
from qiskit import QuantumCircuit


##
# @brief A QGD Python interface class for the decomposition of N-qubit unitaries into U3 and CNOT gates.
class qgd_N_Qubit_Decomposition(qgd_N_Qubit_Decomposition_Wrapper):
    
    
## 
# @brief Constructor of the class.
# @param Umtx The unitary matrix to be decomposed.
# @param optimize_layer_num Set true to optimize the minimum number of operation layers required in the decomposition, or false when the predefined maximal number of layer gates is used (ideal for general unitaries).
# @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: "zeros" ,"random", "close_to_zero".
# @return An instance of the class
    def __init__( self, Umtx, optimize_layer_num=False, initial_guess="zeros" ):

        ## the number of qubits
        self.qbit_num = int(round( np.log2( len(Umtx) ) ))

        # call the constructor of the wrapper class
        super(qgd_N_Qubit_Decomposition, self).__init__(Umtx, self.qbit_num, optimize_layer_num, initial_guess)



##
# @brief Export the unitary decomposition into Qiskit format.
# @return Return with a Qiskit compatible quantum circuit.
    def get_Quantum_Circuit( self ):

        

        # creating Qiskit quantum circuit
        circuit = QuantumCircuit(self.qbit_num)

        # retrive the list of decomposing operations
        operations = self.get_Operations()

        # constructing quantum circuit
        for idx in range(len(operations)-1, -1, -1):

            operation = operations[idx]

            if operation.get("type") == "CNOT":
                # adding CNOT operation to the quantum circuit
                circuit.cx(operation.get("control_qbit"), operation.get("target_qbit"))

            elif operation.get("type") == "U3":
                # adding U3 operation to the quantum circuit
                circuit.u(operation.get("Theta"), operation.get("Phi"), operation.get("Lambda"), operation.get("target_qbit"))


        return circuit


##
# @brief Call to set the number of blocks to be optimized in one shot
# @param optimalization_block The number of blocks to be optimized in one shot
    def set_optimalization_block( self, optimalization_block ):

        _qgd_library.iface_set_optimalization_block( self.c_instance, ctypes.c_int(optimalization_block) )



        
        
            

