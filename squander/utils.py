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

## \file utils.py
## \brief Generic utility functionalities for SQUANDER 



import numpy as np
from squander.IO_interfaces import Qiskit_IO
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from qiskit import QuantumCircuit

import qiskit
qiskit_version = qiskit.version.get_version_info()

if qiskit_version[0] == '0':
    from qiskit import Aer
    from qiskit import execute
    if int(qiskit_version[2])>3:
        from qiskit.quantum_info import Operator
else:
    import qiskit_aer as Aer
    from qiskit import transpile
    from qiskit.quantum_info import Operator



##
# @brief Call to retrieve the unitary from QISKIT circuit
def get_unitary_from_qiskit_circuit( circuit: QuantumCircuit ):
    """
    Call to extract a unitary from Qiskit circuit 

    Args:

        circuit (QuantumCircuit) A Qiskit circuit

    Return:

        Returns with the generated unitary

    """

        
    if qiskit_version[0] == '0':
        backend = Aer.get_backend('aer_simulator')
        circuit.save_unitary()
        
        # job execution and getting the result as an object
        job = execute(circuit, backend)
        
        # the result of the Qiskit job
        result=job.result()  
 
    else :       
        
        circuit.save_unitary()
        backend = Aer.AerSimulator(method='unitary')
        
        compiled_circuit = transpile(circuit, backend)
        result = backend.run(compiled_circuit).result()
        


    return np.asarray( result.get_unitary(circuit) )        





def get_unitary_from_qiskit_circuit_operator( circuit: QuantumCircuit ):
    """
    Call to extract a unitary from Qiskit circuit using qiskit.quantum_info.Operator support

    Args:

        circuit (QuantumCircuit) A Qiskit circuit

    Return:

        Returns with the generated unitary

    """


    if qiskit_version[0] == '0' and int(qiskit_version[2])<4:
    
        print("Currently installed version of qiskit does not support extracting the unitary of a circuit via Operator. Using get_unitary_from_qiskit_circuit function instead.")        
        return get_unitary_from_qiskit_circuit(circuit)

    return Operator(circuit).to_matrix()



def qasm_to_squander_circuit( filename: str, return_transpiled=False):
    """
    Converts a QASM file to a SQUANDER circuit

    Args:

        filename (str) The path to the QASM file

    Return:

        Returns with the SQUANDER circuit and the array of the corresponding parameters

    """
    
    qc = qiskit.QuantumCircuit.from_qasm_file(filename)
    from squander.gates import gates_Wrapper as gate
    SUPPORTED_GATES_NAMES = {n.lower().replace("cnot", "cx") for n in dir(gate) if not n.startswith("_") and issubclass(getattr(gate, n), gate.Gate) and n != "Gate"}
    if any(gate.operation.name not in SUPPORTED_GATES_NAMES for gate in qc.data):
        qc_transpiled = qiskit.transpile(qc, basis_gates=SUPPORTED_GATES_NAMES, optimization_level=0)
    else:
        qc_transpiled = qc

    circuit_squander, circut_parameters = Qiskit_IO.convert_Qiskit_to_Squander(qc_transpiled)
    
    if return_transpiled: 
        return circuit_squander, circut_parameters, qc_transpiled
    return circuit_squander, circut_parameters




def CompareCircuits( circ1: Circuit, parameters1: np.ndarray, circ2: Circuit, parameters2: np.ndarray, parallel : int = 1, tolerance: float = 1e-5) :
    """
    Call to test if the two circuits give the same state transformation upon a random input state

    
    Args:

        circ1 ( Circuit ) A circuit

        parameters1 ( np.ndarray ) A parameter array associated with the input circuit

        circ2 ( Circuit ) A circuit

        parameters2 ( np.ndarray ) A parameter array associated with the input circuit

        parallel (int, optional) Set 0 for sequential evaluation, 1 for using TBB parallelism or 2 for using openMP

        tolerance ( float, optional) The tolerance of the comparision when the inner product of the resulting states is matched to unity.

    
    Return:

        Returns with True if the two circuits give identical results.
    """ 


    qbit_num1 = circ1.get_Qbit_Num()
    qbit_num2 = circ2.get_Qbit_Num()

    if qbit_num1 != qbit_num2:
        raise Exception( "The two compared circuits should have the same number of qubits." )
    
    matrix_size = 1 << qbit_num1
    initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
    initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
    initial_state = initial_state_real + initial_state_imag*1j
    norm = np.sum(initial_state_real * initial_state_real + initial_state_imag*initial_state_imag)
    initial_state = initial_state/np.sqrt(norm)
    


    transformed_state_1 = initial_state.copy()
    transformed_state_2 = initial_state    
    
    circ1.apply_to( parameters1, transformed_state_1, parallel=parallel )
    circ2.apply_to( parameters2, transformed_state_2, parallel=parallel)    
    
    overlap = np.sum( transformed_state_1.conj() * transformed_state_2 )
    #print( "overlap: ", np.abs(overlap) )

    assert( (np.abs(overlap)-1) < tolerance )


def circuit_to_CNOT_basis( circ: Circuit, parameters: np.ndarray):
    """
    Call to transpile a SQUANDER circuit to CNOT basis

    
    Args:

        circ ( Circuit ) A circuit

        parameters ( np.ndarray ) A parameter array associated with the input circuit

                
    Return:

        Returns with the transpiled circuit and the associated parameters
    """
    from squander.gates.gates_Wrapper import (
        CH, CZ, SYC, CRY, CU, CR, CROT ) #CCX, CSWAP, SWAP, CRX, CRZ, CP
    gates = circ.get_Gates()
    circuit = Circuit( circ.get_Qbit_Num() )
    params = []
    for gate in gates:
        if isinstance(gate, Circuit):
            subcircuit, subparams = circuit_to_CNOT_basis( gate, parameters[ gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index() + gate.get_Parameter_Num() ] )
            circuit.add_Gate( subcircuit )
            params.append( subparams )
        elif isinstance(gate, CH):
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_U1(gate.get_Control_Qbit())
            params.append([-np.pi/2/2, -np.pi/2/2, np.pi/4/2, -np.pi/4/2, np.pi/2, np.pi/2])
        elif isinstance(gate, CZ):
            circuit.add_H(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_H(gate.get_Target_Qbit())
            params.append([])
        elif isinstance(gate, SYC):
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_SX(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_SX(gate.get_Control_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_SX(gate.get_Control_Qbit())
            circuit.add_SX(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_SX(gate.get_Control_Qbit())
            circuit.add_SX(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Control_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_SX(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_SX(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_SX(gate.get_Control_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_SX(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Control_Qbit())
            circuit.add_SX(gate.get_Target_Qbit())
            circuit.add_SX(gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Control_Qbit())
            params.append([-np.pi/3/2, np.pi/6/2, np.pi/2/2, np.pi/2/2, 11*np.pi/12/2, np.pi/2/2, np.pi/2/2, 9*np.pi/4/2, np.pi/2, -3*np.pi/4/2, 5*np.pi/2/2, np.pi/2, -np.pi/2, np.pi/4/2, -3*np.pi/4/2])
            """
            #CZPowGate(t)=CP(sympy.pi*t)
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Control_Qbit())
            #iSWAP-power decomposition
            circuit.add_S(gate.get_Target_Qbit())
            circuit.add_S(gate.get_Control_Qbit())
            circuit.add_H(gate.get_Target_Qbit())
            circuit.add_H(gate.get_Control_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_H(gate.get_Target_Qbit())
            circuit.add_H(gate.get_Control_Qbit())
            circuit.add_Sdg(gate.get_Target_Qbit())
            circuit.add_Sdg(gate.get_Control_Qbit())
            circuit.add_H(gate.get_Target_Qbit())
            circuit.add_H(gate.get_Control_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_H(gate.get_Target_Qbit())
            circuit.add_H(gate.get_Control_Qbit())
            #add_GP(np.pi/24) - global phase
            params.append([-np.pi/12/2, np.pi/12/2, -np.pi/12/2, np.pi/2/2, np.pi/2/2])
            """
        elif isinstance(gate, CRY):
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            theta, = parameters[gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index() + gate.get_Parameter_Num()]
            params.append([-theta/2, theta/2])
        elif isinstance(gate, CU):
            circuit.add_U1(gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            theta, phi, lbda, gamma = parameters[ gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index() + gate.get_Parameter_Num() ]
            params.append([(lbda+phi)/2+gamma, lbda/2, theta/2, -theta/2, -(phi+lbda)/2/2, (phi-lbda)/2/2])
        elif isinstance(gate, CR):
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            theta, phi = parameters[ gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index() + gate.get_Parameter_Num() ]
            params.append([(-phi+np.pi/2)/2, -theta/2, theta/2, (phi-np.pi/2)/2])
        elif isinstance(gate, CROT):
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            theta, phi = parameters[ gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index() + gate.get_Parameter_Num() ]
            params.append([-phi/2, np.pi/2/2, -theta, -np.pi/2/2, phi/2])
        else:
            circuit.add_Gate(gate)
            params.append( parameters[ gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index() + gate.get_Parameter_Num() ] )

    return circuit, np.concatenate(params)

def test_circuit_to_CNOT_basis():
    circ = Circuit(2)
    circ.add_CH(0, 1)
    circ.add_CZ(0, 1)
    circ.add_CRY(0, 1)
    circ.add_SYC(0, 1)
    circ.add_CR(0, 1)
    circ.add_CROT(0, 1)
    circ.add_CU(0, 1)
    paramcount = 0 + 0 + 1 + 0 + 2 + 2 + 4
    params = np.random.rand(paramcount)*2*np.pi
    newcirc, newparams = circuit_to_CNOT_basis(circ, params)
    Umat = np.eye(1<<2, dtype=np.complex128)
    Umatnew = np.eye(1<<2, dtype=np.complex128)
    circ.apply_to(params, Umat)
    newcirc.apply_to(newparams, Umatnew)
    #phase = np.angle(np.linalg.det(Umat @ np.linalg.inv(Umatnew)))
    phase = np.angle((Umatnew @ Umat.conj().T)[0,0])
    # Normalize one matrix
    Umatnew = Umatnew * np.exp(-1j * phase)
    # Check closeness
    assert np.allclose(Umat, Umatnew), (Umat, Umatnew)
#test_circuit_to_CNOT_basis()
