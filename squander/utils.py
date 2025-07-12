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

    allowed_gates = {'u', 'u3', 'cx', 'cry', 'cz', 'ch', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 'sx'}

    if any(gate[0].name not in allowed_gates for gate in qc.data):
        qc_transpiled = qiskit.transpile(qc, basis_gates=allowed_gates, optimization_level=0)
    else: 
        qc_transpiled = qc

    circuit_squander, circut_parameters = Qiskit_IO.convert_Qiskit_to_Squander(qc_transpiled)
    
    if return_transpiled: return circuit_squander, circut_parameters, qc_transpiled
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

                


