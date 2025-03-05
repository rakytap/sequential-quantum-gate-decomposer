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
from squander import CRY
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
def get_Qiskit_Circuit( Squander_circuit, parameters ):

    from qiskit import QuantumCircuit

    # creating Qiskit quantum circuit  
    circuit = QuantumCircuit(Squander_circuit.get_Qbit_Num() )
    
    gates = Squander_circuit.get_Gates()
    
    # constructing quantum circuit
    for gate in gates:

        if isinstance( gate, CNOT ):
            # adding CNOT gate to the quantum circuit
            circuit.cx( gate.get_Control_Qbit(), gate.get_Target_Qbit() )
            
        elif isinstance( gate, CRY ):
            # adding CNOT gate to the quantum circuit
            parameters_gate = gate.Extract_Parameters( parameters )
            circuits.cry( parameters_gate[0], gate.get_Control_Qbit(), gate.get_Target_Qbit() )
            
        elif isinstance( gate, CZ ):
            # adding CZ gate to the quantum circuit
            circuit.cz( gate.get_Control_Qbit(), gate.get_Target_Qbit() )

        elif isinstance( gate, CH ):
            # adding CZ gate to the quantum circuit
            circuit.ch( gate.get_Control_Qbit(), gate.get_Target_Qbit() )

        elif isinstance( gate, SYC ):
            # Sycamore gate
            print("Unsupported gate in the circuit export: Sycamore gate")
            return None;

        elif isinstance( gate, U3 ):
            # adding U3 gate to the quantum circuit
            parameters_gate = gate.Extract_Parameters( parameters )
            circuit.u( parameters_gate[0], parameters_gate[1], parameters_gate[2], gate.get_Target_Qbit() )    

        elif isinstance( gate, RX ):
            # RX gate
            parameters_gate = gate.Extract_Parameters( parameters )
            circuit.rx( parameters_gate[0], gate.get_Target_Qbit() )    
            
        elif isinstance( gate, RY ):
            # RY gate
            parameters_gate = gate.Extract_Parameters( parameters )
            circuit.ry( parameters_gate[0], gate.get_Target_Qbit() )    

        elif isinstance( gate, RZ ):
            # RZ gate
            parameters_gate = gate.Extract_Parameters( parameters )
            circuit.rz( parameters_gate[0], gate.get_Target_Qbit() )    
            
        elif isinstance( gate, H ):
            # Hadamard gate
            circuit.h( gate.get_Target_Qbit() )    

        elif isinstance( gate, X ):
            # X gate
            circuit.x( gate.get_Target_Qbit() )  

        elif isinstance( gate, Y ):
            # Y gate
            circuit.y( gate.get_Target_Qbit() )  

        elif isinstance( gate, Z ):
            # Z gate
            circuit.z( gate.get_Target_Qbit() )  

        elif isinstance( gate, SX ):
            # SX gate
            circuit.sx( gate.get_Target_Qbit() )  

        elif isinstance( gate, Circuit ):
            # Sub-circuit gate
            raise ValueError("Qiskit export of circuits with subcircuit is not supported. Use Circuit::get_Flat_Circuit prior of exporting circuit.")  
            
        else:
            print(gate)
            raise ValueError("Unsupported gate in the circuit export.")
    
    return( circuit )
    

##
# @brief Export the inverse of a Squsnder circuit into Qiskit circuit.
# @param a Squander object containing a squander circuit (Circuit, any decomposing class, VQE class, etc..)
# @return Return with a Qiskit compatible quantum circuit.
def get_Qiskit_Circuit_inverse( Squander_circuit, parameters ):

    from qiskit import QuantumCircuit

    # creating Qiskit quantum circuit
    circuit = QuantumCircuit(Squander_circuit.get_Qbit_Num() )
    
    gates = Squander_circuit.get_Gates()
    
    # constructing quantum circuit
    for idx in range(len(gates)-1, -1, -1):

        gate = gates[idx]

        if isinstance( gate, CNOT ):
            # adding CNOT gate to the quantum circuit
            circuit.cx( gate.get_Control_Qbit(), gate.get_Target_Qbit() )
            
        elif isinstance( gate, CRY ):
            # adding CNOT gate to the quantum circuit
            parameters_gate = gate.Extract_Parameters( parameters )
            circuit.cry( -parameters_gate[0], gate.get_Control_Qbit(), gate.get_Target_Qbit() )
            
        elif isinstance( gate, CZ ):
            # adding CZ gate to the quantum circuit
            circuit.cz( gate.get_Control_Qbit(), gate.get_Target_Qbit() )

        elif isinstance( gate, CH ):
            # adding CZ gate to the quantum circuit
            circuit.ch( gate.get_Control_Qbit(), gate.get_Target_Qbit() )

        elif isinstance( gate, SYC ):
            # Sycamore gate
            print("Unsupported gate in the circuit export: Sycamore gate")
            return None;

        elif isinstance( gate, U3 ):
            # adding U3 gate to the quantum circuit
            parameters_gate = gate.Extract_Parameters( parameters )
            circuit.u( -parameters_gate[0], -parameters_gate[2], -parameters_gate[1], gate.get_Target_Qbit() )   

        elif isinstance( gate, RX ):
            # RX gate
            parameters_gate = gate.Extract_Parameters( parameters )
            circuit.rx( -parameters_gate[0], gate.get_Target_Qbit() )    
            
        elif isinstance( gate, RY ):
            # RY gate
            parameters_gate = gate.Extract_Parameters( parameters )
            circuit.ry( -parameters_gate[0], gate.get_Target_Qbit() )    

        elif isinstance( gate, RZ ):
            # RZ gate
            parameters_gate = gate.Extract_Parameters( parameters )
            circuit.rz( -parameters_gate[0], gate.get_Target_Qbit() )    
            
        elif isinstance( gate, H ):
            # Hadamard gate
            circuit.h( gate.get_Target_Qbit() )    

        elif isinstance( gate, X ):
            # X gate
            circuit.x( gate.get_Target_Qbit() )  

        elif isinstance( gate, Y ):
            # Y gate
            circuit.y( gate.get_Target_Qbit() )  

        elif isinstance( gate, Z ):
            # Z gate
            circuit.z( gate.get_Target_Qbit() )  

        elif isinstance( gate, SX ):
            # SX gate
            circuit.sx( gate.get_Target_Qbit() )

        elif isinstance( gate, Circuit ):
            # Sub-circuit gate
            raise ValueError("Qiskit export of circuits with subcircuit is not supported. Use Circuit::get_Flat_Circuit prior of exporting circuit.")   
            
        else:
            print(gate)
            raise ValueError("Unsupported gate in the circuit export.")
    #print( circuit )
    return( circuit )


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




