## #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:44:26 2020
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

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""

## \file qgd_N_Qubit_Decomposition_custom.py
##    \brief A QGD Python interface class for the decomposition of N-qubit unitaries into a set of two-qubit and one-qubit gates.


import numpy as np
from os import path
from qgd_python.decomposition.qgd_N_Qubit_Decomposition_custom_Wrapper import qgd_N_Qubit_Decomposition_custom_Wrapper



##
# @brief A QGD Python interface class for the decomposition of N-qubit unitaries into U3 and CNOT gates.
class qgd_N_Qubit_Decomposition_custom(qgd_N_Qubit_Decomposition_custom_Wrapper):
    
    
## 
# @brief Constructor of the class.
# @param Umtx The unitary matrix to be decomposed.
# @param optimize_layer_num Set true to optimize the minimum number of operation layers required in the decomposition, or false when the predefined maximal number of layer gates is used (ideal for general unitaries).
# @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: "zeros" ,"random", "close_to_zero".
# @return An instance of the class
    def __init__( self, Umtx, initial_guess="RANDOM", config={}, accelerator_num=0 ):

        ## the number of qubits
        self.qbit_num = int(round( np.log2( len(Umtx) ) ))
        
        # config
        if not( type(config) is dict):
            print("Input parameter config should be a dictionary describing the following hyperparameters:") #TODO
            return

        # call the constructor of the wrapper class
        super(qgd_N_Qubit_Decomposition_custom, self).__init__(Umtx, self.qbit_num, initial_guess, config=config, accelerator_num=accelerator_num)



##
# @brief Wrapper function to call the start_decomposition method of C++ class N_Qubit_Decomposition
# @param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
    def Start_Decomposition(self,prepare_export=True):

	# call the C wrapper function
        super(qgd_N_Qubit_Decomposition_custom, self).Start_Decomposition(prepare_export=prepare_export)


##
# @brief Call to reorder the qubits in the matrix of the gate
# @param qbit_list The reordered list of qubits spanning the matrix
    def Reorder_Qubits( self, qbit_list ):

	# call the C wrapper function
        super(qgd_N_Qubit_Decomposition_custom, self).Reorder_Qubits(qbit_list)


##
# @brief @brief Call to print the gates decomposing the initial unitary. These gates brings the intial matrix into unity.
    def List_Gates(self):

	# call the C wrapper function
        super(qgd_N_Qubit_Decomposition_custom, self).List_Gates()

##
# @brief Export the unitary decomposition into Qiskit format.
# @return Return with a Qiskit compatible quantum circuit.
    def get_Quantum_Circuit( self ):

        from qiskit import QuantumCircuit

        # creating Qiskit quantum circuit
        circuit = QuantumCircuit(self.qbit_num)

        # retrive the list of decomposing gate structure
        gates = self.get_Gates()

        # constructing quantum circuit
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

            elif gate.get("type") == "RZ" or gate.get("type") == "RZ_P":
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




##
# @brief Export the unitary decomposition into Qiskit format.
# @return Return with a Qiskit compatible quantum circuit.
    def get_Cirq_Circuit( self ):

        import cirq


        # creating Cirq quantum circuit
        circuit = cirq.Circuit()

        # creating qubit register
        q = cirq.LineQubit.range(self.qbit_num)

        # retrive the list of decomposing gate structure
        gates = self.get_Gates()

        # constructing quantum circuit
        for idx in range(len(gates)-1, -1, -1):

            gate = gates[idx]

            if gate.get("type") == "CNOT":
                # adding CNOT gate to the quantum circuit
                circuit.append(cirq.CNOT(q[self.qbit_num-1-gate.get("control_qbit")], q[self.qbit_num-1-gate.get("target_qbit")]))

            if gate.get("type") == "CRY":
                # adding CRY gate to the quantum circuit
                print("CRY gate needs to be implemented")

            elif gate.get("type") == "CZ":
                # adding CZ gate to the quantum circuit
                circuit.append(cirq.CZ(q[self.qbit_num-1-gate.get("control_qbit")], q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "CH":
                # adding CZ gate to the quantum circuit
                circuit.append(cirq.CH(q[self.qbit_num-1-gate.get("control_qbit")], q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "SYC":
                # Sycamore gate
                circuit.append(cirq.google.SYC(q[self.qbit_num-1-gate.get("control_qbit")], q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "U3":
                print("Unsupported gate in the Cirq export: U3 gate")
                return None;

            elif gate.get("type") == "RX":
                # RX gate
                circuit.append(cirq.rx(gate.get("Theta")).on(q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "RY":
                # RY gate
                circuit.append(cirq.ry(gate.get("Theta")).on(q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "RZ":
                # RZ gate
                circuit.append(cirq.rz(gate.get("Phi")).on(q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "X":
                # X gate
                circuit.append(cirq.x(q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "Y":
                # Y gate
                circuit.append(cirq.y(q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "Z":
                # Z gate
                circuit.append(cirq.z(q[self.qbit_num-1-gate.get("target_qbit")]))


            elif gate.get("type") == "SX":
                # RZ gate
                circuit.append(cirq.sx(q[self.qbit_num-1-gate.get("target_qbit")]))


        return circuit

        
##
# @brief Call to import initial quantum circuit in QISKIT format to be further comporessed
# @param qc_in The quantum circuit to be imported
    def import_Qiskit_Circuit( self, qc_in ):  

        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import ParameterExpression
        from qgd_python.gates.qgd_Circuit import qgd_Circuit

        qc = transpile(qc_in, optimization_level=3, basis_gates=['cz', 'cx', 'u3'], layout_method='sabre')
        #print('Depth of Qiskit transpiled quantum circuit:', qc.depth())
        #print('Gate counts in Qiskit transpiled quantum circuit:', qc.count_ops())

        # get the register of qubits
        q_register = qc.qubits

        # get the size of the register
        register_size = qc.num_qubits
        
        # prepare dictionary for single qubit gates
        single_qubit_gates = dict()
        for idx in range(register_size):
            single_qubit_gates[idx] = []

        # prepare dictionary for two-qubit gates
        two_qubit_gates = list()

        # construct the qgd gate structure        
        Circuit_ret = qgd_Circuit(register_size)

        optimized_parameters = list()


        for gate in qc.data:

            name = gate[0].name            

            if name == 'u3':
                # add u3 gate 
                qubits = gate[1]
                
                qubit = q_register.index( qubits[0] )   # qubits[0].index
                single_qubit_gates[qubit].append( {'params': gate[0].params, 'type': 'u3'} )

            elif name == 'cz' or name == 'cx':
                # add cz gate 
                qubits = gate[1]

                qubit0 = q_register.index( qubits[0] ) #qubits[0].index
                qubit1 = q_register.index( qubits[1] ) #qubits[1].index


                two_qubit_gate = {'type': name, 'qubits': [qubit0, qubit1]}

                # creating an instance of the wrapper class qgd_Circuit
                Layer = qgd_Circuit( register_size )

                # retrive the corresponding single qubit gates and create layer from them
                if len(single_qubit_gates[qubit0])>0:
                    gate0 = single_qubit_gates[qubit0][0]
                    single_qubit_gates[qubit0].pop(0)
                    
                    if gate0['type'] == 'u3':
                        Layer.add_U3( qubit0, True, True, True )
                        params = gate0['params']
                        params.reverse()
                        for param in params:
                            optimized_parameters = optimized_parameters + [float(param)]

                        optimized_parameters[-1] = optimized_parameters[-1]/2 #SQUADER works with theta/2
                        

                if len(single_qubit_gates[qubit1])>0:
                    gate1 = single_qubit_gates[qubit1][0]
                    single_qubit_gates[qubit1].pop(0)
                    
                    if gate1['type'] == 'u3':
                        Layer.add_U3( qubit1, True, True, True )
                        params = gate1['params']
                        params.reverse()
                        for param in params:
                            optimized_parameters = optimized_parameters + [float(param)]
                        optimized_parameters[-1] = optimized_parameters[-1]/2 #SQUADER works with theta/2

                ## add cz gate to the layer       
                if name == 'cz':
                    Layer.add_CZ( qubit1, qubit0 )
                elif name == 'cx':
                    Layer.add_CNOT( qubit1, qubit0 )

                Circuit_ret.add_Circuit( Layer )
   
       

        # add remaining single qubit gates
        # creating an instance of the wrapper class qgd_Circuit
        Layer = qgd_Circuit( register_size )

        for qubit in range(register_size):

            gates = single_qubit_gates[qubit]
            for gate in gates:

                if gate['type'] == 'u3':
                    Layer.add_U3( qubit, True, True, True )
                    params = gate['params']
                    params.reverse()
                    for param in params:
                        optimized_parameters = optimized_parameters + [float(param)]
                    optimized_parameters[-1] = optimized_parameters[-1]/2 #SQUADER works with theta/2

        Circuit_ret.add_Circuit( Layer )

        optimized_parameters = np.asarray(optimized_parameters, dtype=np.float64)

        # setting gate structure and optimized initial parameters
        self.set_Gate_Structure(Circuit_ret)
        self.set_Optimized_Parameters( np.flip(optimized_parameters,0) )      
        


## 
# @brief Call to set the optimizer used in the gate synthesis process
# @param optimizer String indicating the optimizer. Possible values: "BFGS" ,"ADAM", "BFGS2", "AGENTS", "COSINE", "AGENTS_COMBINED".
# @return An instance of the class
    def set_Optimizer( self, optimizer="BFGS" ):

        # Set the optimizer
        super(qgd_N_Qubit_Decomposition_custom, self).set_Optimizer(optimizer)  



## 
# @brief Call to prepare the circuit to be exported into Qiskit format. (parameters and gates gets bound together, gate block structure is converted to plain structure).
    def Prepare_Gates_To_Export(self):

        # Set the optimizer
        super(qgd_N_Qubit_Decomposition_custom, self).Prepare_Gates_To_Export()  
        
        
##
# @brief Call to set custom gate structure to used in the decomposition
# @param Gate_structure An instance of Gates_Block
    def set_Gate_Structure( self, Gate_structure ):  

        from qgd_python.gates.qgd_Circuit import qgd_Circuit

        if not isinstance(Gate_structure, qgd_Circuit) :
            raise Exception("Input parameter Gate_structure should be a an instance of Gates_Block")
               
                          
        super(qgd_N_Qubit_Decomposition_custom, self).set_Gate_Structure( Gate_structure )   



## 
# @brief Call to set the optimizer used in the gate synthesis process
# @param costfnc Variant of the cost function. Input argument 0 stands for FROBENIUS_NORM, 1 for FROBENIUS_NORM_CORRECTION1, 2 for FROBENIUS_NORM_CORRECTION2
    def set_Cost_Function_Variant( self, costfnc=0 ):

        # Set the optimizer
        super(qgd_N_Qubit_Decomposition_custom, self).set_Cost_Function_Variant(costfnc=costfnc)         

