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
##    \brief Utility functionalities for SQUANDER python binding



import numpy as np
from squander import Qiskit_IO

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
def get_unitary_from_qiskit_circuit( circuit ):

        
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


##
#@brief Call to extract a unitary from Qiskit circuit
#@param A Qiskit circuit
#@return Returns with the generated unitary
def get_unitary_from_qiskit_circuit_operator(circuit):


    if qiskit_version[0] == '0' and int(qiskit_version[2])<4:
    
        print("Currently installed version of qiskit does not support extracting the unitary of a circuit via Operator. Using get_unitary_from_qiskit_circuit function instead.")
        
        return get_unitary_from_qiskit_circuit(circuit)

    return Operator(circuit).to_matrix()


##
#@brief Converts a QASM file to a SQUANDER circuit
#@param filename The path to the QASM file
#@return Tuple: SQUANDER circuit, List of circuit parameters
def qasm_to_squander_circuit(filename):

    
    qc = qiskit.QuantumCircuit.from_qasm_file(filename)
    circuit_squander, circut_parameters = Qiskit_IO.convert_Qiskit_to_Squander(qc)
    
    return circuit_squander, circut_parameters
                


