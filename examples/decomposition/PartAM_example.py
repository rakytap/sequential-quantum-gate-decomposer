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
if __name__ == '__main__':


    config = {
            'strategy': "TreeSearch",
            'test_subcircuits': True,
            'test_final_circuit': True,
            'max_partition_size': 3,
    }

    filename = "benchmarks/qfast/4q/adder_q4.qasm"
    start_time = time.time()

    # load the circuit from a file
    circ_orig, parameters_orig = utils.qasm_to_squander_circuit(filename)
    config['topology'] = [
    (0, 1), (0, 2), (0, 3), 
    ]
    wide_circuit_optimizer = Partition_Aware_Mapping( config )
    circ, params, input_perm,output_perm = wide_circuit_optimizer.Partition_Aware_Mapping( circ_orig, parameters_orig )
    #print(Qiskit_IO.get_Qiskit_Circuit(circ.get_Flat_Circuit(),params))
    num_qubits = circ.get_Qbit_Num() 
    matrix_size = 1 << num_qubits 
    initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
    initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
    initial_state = initial_state_real + initial_state_imag*1j
    initial_state = initial_state/np.linalg.norm(initial_state)
    original_state = initial_state.copy()
    circ_orig.apply_to(parameters_orig,original_state)
    circ_Final = Circuit(circ.get_Qbit_Num() )
    circ_Final.add_Permutation(input_perm)
    circ_Final.add_Circuit(circ)
    output_perm_T = [0]* circ.get_Qbit_Num() 
    for i, j in enumerate(output_perm):
        output_perm_T[j] = i
    circ_Final.add_Permutation(output_perm)
    PartAM_state = initial_state.copy()
    circ_Final.apply_to(params,PartAM_state)
    print(f"Decomposition error on random state: {1-abs(np.vdot(PartAM_state,original_state))}")
    print("--- %s seconds elapsed during optimization ---" % (time.time() - start_time))


