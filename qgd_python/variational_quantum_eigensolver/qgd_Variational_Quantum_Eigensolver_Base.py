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
from qgd_python.variational_quantum_eigensolver.qgd_Variational_Quantum_Eigensolver_Base_Wrapper import qgd_Variational_Quantum_Eigensolver_Base_Wrapper



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
# @brief Wrapper function to call the start_decomposition method of C++ class N_Qubit_Decomposition
# @param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
    def set_Optimizer(self, alg):
    

        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Optimizer(alg)
        
    def get_Ground_State(self):

	# call the C wrapper function
        super(qgd_Variational_Quantum_Eigensolver_Base, self).get_Ground_State()
    
    def get_Optimized_Parameters(self):
    
        return super(qgd_Variational_Quantum_Eigensolver_Base, self).get_Optimized_Parameters()
        
    def set_Optimized_Parameters(self, new_params):
        
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Optimized_Parameters(new_params)
        
    def set_Optimization_Tolerance(self, tolerance):
    
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Optimization_Tolerance(tolerance)
        
    def set_Project_Name(self, project_name):
    
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Project_Name(project_name)
        
    def set_Gate_Structure_from_Binary(self, filename):
    
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Gate_Structure_From_Binary(filename)
        
    def set_Ansatz(self, ansatz_new):
        
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Ansatz(ansatz_new)
        
    def Generate_initial_circuit(self, layers, blocks, rot_layers):
    
        super(qgd_Variational_Quantum_Eigensolver_Base, self).Generate_initial_circuit(layers, blocks, rot_layers)
        
    def Optimization_Problem(self, parameters):
    
        return super(qgd_Variational_Quantum_Eigensolver_Base, self).Optimization_Problem(parameters)
        
    def get_Quantum_Circuit(self):
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(self.qbit_num)

        gates = self.get_gates()

        for idx in range(len(gates)-1, -1, -1):

            gate = gates[idx]

            if gate.get("type") == "CNOT":
                # adding CNOT gate to the quantum circuit
                circuit.cx(gate.get("control_qbit"), gate.get("target_qbit"))

            elif gate.get("type") == "CRY":
                # adding CNOT gate to the quantum circuit
                circuit.cry(gate.get("Theta"), gate.get("control_qbit"), gate.get("target_qbit"))

            elif gate.get("type") == "CZ":
                # adding CZ gate to the quantum circuit
                circuit.cz(gate.get("control_qbit"), gate.get("target_qbit"))

            elif gate.get("type") == "CH":
                # adding CZ gate to the quantum circuit
                circuit.ch(gate.get("control_qbit"), gate.get("target_qbit"))

            elif gate.get("type") == "SYC":
                # Sycamore gate
                print("Unsupported gate in the circuit export: Sycamore gate")
                return None;

            elif gate.get("type") == "U3":
                # adding U3 gate to the quantum circuit
                circuit.u(gate.get("Theta"), gate.get("Phi"), gate.get("Lambda"), gate.get("target_qbit"))

            elif gate.get("type") == "RX":
                # RX gate
                circuit.rx(gate.get("Theta"), gate.get("target_qbit"))

            elif gate.get("type") == "RY":
                # RY gate
                circuit.ry(gate.get("Theta"), gate.get("target_qbit"))

            elif gate.get("type") == "RZ":
                # RZ gate
                circuit.rz(gate.get("Phi"), gate.get("target_qbit"))

            elif gate.get("type") == "X":
                # X gate
                circuit.x(gate.get("target_qbit"))

            elif gate.get("type") == "Y":
                # Y gate
                circuit.x(gate.get("target_qbit"))

            elif gate.get("type") == "Z":
                # Z gate
                circuit.x(gate.get("target_qbit"))

            elif gate.get("type") == "SX":
                # SX gate
                circuit.sx(gate.get("target_qbit"))


        return circuit



        
        
            

