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
## \file wide_circuit_optimization.py
## \brief Simple example python code demonstrating a wide circuit optimization

import squander.decomposition.qgd_Wide_Circuit_Optimization as Wide_Circuit_Optimization
from squander import Partition_Aware_Mapping
from squander import utils
from squander import Qiskit_IO
import time
from squander import Circuit
import numpy as np
from qiskit import transpile
def generate_star_topology(num_qubits):
    return [(0, i) for i in range(1, num_qubits)]
def extract_two_qubit_gate_count(gate_nums_dict):

    # List of two-qubit gate names
    two_qubit_gates = ['CNOT', 'CZ', 'CU', 'CH', 'SYC', 'CRY', 'CRZ', 'CRX', 'CP', 'SWAP', 'CSWAP']
    
    total_two_qubit = 0
    for gate_name in two_qubit_gates:
        total_two_qubit += gate_nums_dict.get(gate_name, 0)
    return total_two_qubit
if __name__ == '__main__':

    use_qiskit_sabre = False
    config = {  
            'strategy': "TreeSearch", 
            'test_subcircuits': True,
            'test_final_circuit': True,
            'max_partition_size': 3,
    }

    filename = "benchmarks/qfast/5q/vqe.qasm"
    start_time = time.time()

    # load the circuit from a file
    circ_orig, parameters_orig = utils.qasm_to_squander_circuit(filename)
    N = circ_orig.get_Qbit_Num()
    # instantiate the object for optimizing wide circuits
    wide_circuit_optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization( config )

    # run circuti optimization
    circ_flat, parameters = wide_circuit_optimizer.OptimizeWideCircuit( circ_orig, parameters_orig, True )

    config['topology'] = generate_star_topology(N)
    circo = Qiskit_IO.get_Qiskit_Circuit(circ_flat.get_Flat_Circuit(),parameters)
    if use_qiskit_sabre:
        coupling_map = [[i,j] for i,j in config['topology']]
        circuit_qiskit_sabre = transpile(circo, coupling_map=coupling_map)
        circ, parameters = Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit_sabre)
        config['routed']= True
        wide_circuit_optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization( config )
    else:
        wide_circuit_optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization( config )
        # run circuti optimization
        circ, parameters = Qiskit_IO.convert_Qiskit_to_Squander(circo)
    circ, parameters = wide_circuit_optimizer.OptimizeWideCircuit( circ, parameters, True )
    print(f"Two qubit gate count: {extract_two_qubit_gate_count(circ.get_Gate_Nums())}")
    print("--- %s seconds elapsed during optimization ---" % (time.time() - start_time))


