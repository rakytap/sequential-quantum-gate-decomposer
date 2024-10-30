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

## \file Qiskit_IO.py
##    \brief Interface functions to import quantum circuits from Qiskit format into Squander and export Squander circuits into Qiskit format


import numpy as np
from squander import Circuit

from squander import CNOT
#from squander import CRY
from squander import CZ
from squander import CH
from squander import SYC
from squander import U3
from squander import RX
from squander import RY
from squander import RZ
from squander import H
from squander import X
from squander import Y
from squander import Z
from squander import SX







##
# @brief Export the unitary decomposition into Qiskit format.
# @param a Squander object containing a squander circuit (Circuit, any decomposing class, VQE class, etc..)
# @return Return with a Qiskit compatible quantum circuit.
def get_Qiskit_Circuit( Squander_circuit ):

    from qiskit import QuantumCircuit

    # creating Qiskit quantum circuit
    circuit = QuantumCircuit(Squander_circuit.qbit_num)
    
    # retrive the list of decomposing gate structure
    gates = Squander_circuit.get_Gates()
    '''
    squander_circuit_tmp = Squander_circuit.get_Circuit()

    gates_tmp = squander_circuit_tmp.get_Gates()

    # constructing quantum circuit
    for gate in gates_tmp:

        if isinstance( gate, CNOT ):
            # adding CNOT gate to the quantum circuit
            #circuit.cx(gate.get("control_qbit"), gate.get("target_qbit"))
            print( gate.get_Parameter_Num(), ', ', gate.get_Parameter_Start_Index(), ' ', gate.get_Target_Qbit(), ' ' , gate.get_Control_Qbit()   )
            ''
        elif isinstance( gate, CRY )
            # adding CNOT gate to the quantum circuit
            #circuit.cry(gate.get("Theta"), gate.get("control_qbit"), gate.get("target_qbit"))
            ''
        elif isinstance( gate, CZ ):
            # adding CZ gate to the quantum circuit
            #circuit.cz(gate.get("control_qbit"), gate.get("target_qbit"))
            print( gate.get_Parameter_Num(), ', ', gate.get_Parameter_Start_Index(), ' ', gate.get_Target_Qbit(), ' ' , gate.get_Control_Qbit()   )

        elif isinstance( gate, CH ):
            # adding CZ gate to the quantum circuit
            #circuit.ch(gate.get("control_qbit"), gate.get("target_qbit"))
            print( gate.get_Parameter_Num(), ', ', gate.get_Parameter_Start_Index(), ' ', gate.get_Target_Qbit(), ' ' , gate.get_Control_Qbit()   )

        elif isinstance( gate, SYC ):
            # Sycamore gate
            print("Unsupported gate in the circuit export: Sycamore gate")
            return None;

        elif isinstance( gate, U3 ):
            # adding U3 gate to the quantum circuit
            #circuit.u(gate.get("Theta"), gate.get("Phi"), gate.get("Lambda"), gate.get("target_qbit"))
            print( gate.get_Parameter_Num(), ', ', gate.get_Parameter_Start_Index(), ' ', gate.get_Target_Qbit(), ' ' , gate.get_Control_Qbit()   )

        elif isinstance( gate, RX ):
            # RX gate
            #circuit.rx(gate.get("Theta"), gate.get("target_qbit"))
            print( gate.get_Parameter_Num(), ', ', gate.get_Parameter_Start_Index(), ' ', gate.get_Target_Qbit(), ' ' , gate.get_Control_Qbit()   )

        elif isinstance( gate, RY ):
            # RY gate
            #circuit.ry(gate.get("Theta"), gate.get("target_qbit"))
            print( gate.get_Parameter_Num(), ', ', gate.get_Parameter_Start_Index(), ' ', gate.get_Target_Qbit(), ' ' , gate.get_Control_Qbit()   )

        elif isinstance( gate, RZ ):
            # RZ gate
            #circuit.rz(gate.get("Phi"), gate.get("target_qbit"))
            print( gate.get_Parameter_Num(), ', ', gate.get_Parameter_Start_Index(), ' ', gate.get_Target_Qbit(), ' ' , gate.get_Control_Qbit()   )
            
        elif isinstance( gate, H ):
            # Hadamard gate
            #circuit.h(gate.get("target_qbit"))    
            print( gate.get_Parameter_Num(), ', ', gate.get_Parameter_Start_Index(), ' ', gate.get_Target_Qbit(), ' ' , gate.get_Control_Qbit()   )        

        elif isinstance( gate, X ):
            # X gate
            #circuit.x(gate.get("target_qbit"))
            print( gate.get_Parameter_Num(), ', ', gate.get_Parameter_Start_Index(), ' ', gate.get_Target_Qbit(), ' ' , gate.get_Control_Qbit()   )

        elif isinstance( gate, Y ):
            # Y gate
            #circuit.x(gate.get("target_qbit"))
            print( gate.get_Parameter_Num(), ', ', gate.get_Parameter_Start_Index(), ' ', gate.get_Target_Qbit(), ' ' , gate.get_Control_Qbit()   )

        elif isinstance( gate, Z ):
            # Z gate
            #circuit.x(gate.get("target_qbit"))
            print( gate.get_Parameter_Num(), ', ', gate.get_Parameter_Start_Index(), ' ', gate.get_Target_Qbit(), ' ' , gate.get_Control_Qbit()   )

        elif isinstance( gate, SX ):
            # SX gate
            #circuit.sx(gate.get("target_qbit"))
            print( gate.get_Parameter_Num(), ', ', gate.get_Parameter_Start_Index(), ' ', gate.get_Target_Qbit(), ' ' , gate.get_Control_Qbit()   )
            
        else:
            print(gate)
            raise ValueError("Unsupported gate in the circuit export: " +  gate.get("type"))
    '''


    # constructing quantum circuit
    for idx in range(len(gates)):

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
            
        elif gate.get("type") == "H":
            # Hadamard gate
            circuit.h(gate.get("target_qbit"))            

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
            
        else:
            print(gate)
            raise ValueError("Unsupported gate in the circuit export: " +  gate.get("type"))


    return circuit
    
    

##
# @brief Export the inverse of a Squsnder circuit into Qiskit circuit.
# @param a Squander object containing a squander circuit (Circuit, any decomposing class, VQE class, etc..)
# @return Return with a Qiskit compatible quantum circuit.
def get_Qiskit_Circuit_inverse( Squander_circuit ):

	from qiskit import QuantumCircuit

	# creating Qiskit quantum circuit
	circuit = QuantumCircuit(Squander_circuit.qbit_num)

	# retrive the list of decomposing gate structure
	gates = Squander_circuit.get_Gates()

	# constructing quantum circuit
	for idx in range(len(gates)-1, -1, -1):

		gate = gates[idx]


		if gate.get("type") == "CNOT":
                # adding CNOT gate to the quantum circuit
			circuit.cx(gate.get("control_qbit"), gate.get("target_qbit"))

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
			circuit.u(-gate.get("Theta"), -gate.get("Lambda"), -gate.get("Phi"), gate.get("target_qbit"))

		elif gate.get("type") == "RX":
                # RX gate
			circuit.rx(-gate.get("Theta"), gate.get("target_qbit"))

		elif gate.get("type") == "RY":
                # RY gate
			circuit.ry(-gate.get("Theta"), gate.get("target_qbit"))

		elif gate.get("type") == "RZ" or gate.get("type") == "RZ_P":
                # RZ gate
			circuit.rz(-gate.get("Phi"), gate.get("target_qbit"))
				
		elif gate.get("type") == "H":
                # Hadamard gate
			circuit.h(gate.get("target_qbit"))				

		elif gate.get("type") == "X":
                # X gate
			circuit.x(gate.get("target_qbit"))

		elif gate.get("type") == "SX":
                # SX gate
			circuit.sx(gate.get("target_qbit"))

	return circuit
    
    



##
# @brief Call to import initial quantum circuit in QISKIT format to be further comporessed
# @param qc_in The quantum circuit to be imported
def convert_Qiskit_to_Squander( qc_in ):  

    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterExpression


        

    # get the register of qubits
    q_register = qc_in.qubits

    # get the size of the register
    register_size = qc_in.num_qubits

    # construct the qgd gate structure            
    Circuit_Squander = Circuit(register_size)
    parameters = list()

    for gate in qc_in.data:

        name = gate[0].name            
        #print('Gate name in Qiskit: ', name )


        if name == 'u' or name == 'u3':
            # add u3 gate 
            qubits = gate[1]                
            qubit = q_register.index( qubits[0] )   # qubits[0].index

            params = gate[0].params
            params[0] = params[0]/2 #SQUADER works with theta/2
                    
            for param in params:
                parameters = parameters + [float(param)]

            Circuit_Squander.add_U3( qubit, True, True, True )


        elif name == 'cx':
            # add cx gate 
            qubits = gate[1]
            qubit0 = q_register.index( qubits[0] )
            qubit1 = q_register.index( qubits[1] )
            Circuit_Squander.add_CNOT( qubit1, qubit0 )

        elif name == "cry":

            qubits = gate[1]
            qubit0 = q_register.index( qubits[0] )
            qubit1 = q_register.index( qubits[1] )

            params = gate[0].params
            params[0] = params[0]/2 #SQUADER works with theta/2

            for param in params:
                parameters = parameters + [float(param)]

            Circuit_Squander.add_CRY( qubit1, qubit0 )

        elif name == "cz":
            qubits = gate[1]
            qubit0 = q_register.index( qubits[0] ) 
            qubit1 = q_register.index( qubits[1] )
            Circuit_Squander.add_CZ( qubit1, qubit0 )

        elif name == "ch":
            qubits = gate[1]
            qubit0 = q_register.index( qubits[0] ) 
            qubit1 = q_register.index( qubits[1] )
            Circuit_Squander.add_CH( qubit1, qubit0 )

        elif name == "rx":
            qubits = gate[1]                
            qubit = q_register.index( qubits[0] ) 

            params = gate[0].params
            params[0] = params[0]/2 #SQUADER works with theta/2
                    
            for param in params:
                parameters = parameters + [float(param)]

            Circuit_Squander.add_RX( qubit )

        elif name == "ry":
            qubits = gate[1]                
            qubit = q_register.index( qubits[0] ) 

            params = gate[0].params
            params[0] = params[0]/2 #SQUADER works with theta/2
                    
            for param in params:
                parameters = parameters + [float(param)]

            Circuit_Squander.add_RY( qubit )

        elif name == "rz" :
            qubits = gate[1]                
            qubit = q_register.index( qubits[0] ) 

            params    = gate[0].params
            params[0] = params[0]/2 #SQUADER works with phi/2
                    
            for param in params:
                parameters = parameters + [float(param)]

            Circuit_Squander.add_RZ( qubit )

        elif name == "h":
            qubits = gate[1]                
            qubit = q_register.index( qubits[0] ) 

            Circuit_Squander.add_H( qubit )

        elif name == "x":
            qubits = gate[1]                
            qubit = q_register.index( qubits[0] ) 

            Circuit_Squander.add_X( qubit )

        elif name == "y":
            qubits = gate[1]                
            qubit = q_register.index( qubits[0] ) 

            Circuit_Squander.add_Y( qubit )

        elif name == "z":
            qubits = gate[1]                
            qubit = q_register.index( qubits[0] ) 

            Circuit_Squander.add_Z( qubit )

        elif name == "sx":
            qubits = gate[1]                
            qubit = q_register.index( qubits[0] ) 

            Circuit_Squander.add_SX( qubit )
        else:
            print("convert_Qiskit_to_Squander: Unimplemented gate: ", name )


    parameters = np.asarray(parameters, dtype=np.float64)


    return Circuit_Squander, parameters




