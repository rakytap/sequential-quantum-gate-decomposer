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
            'diagnostics': True,  # Enable diagnostic output
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
    output_perm_T = [0]* circ.get_Qbit_Num() 
    for i, j in enumerate(output_perm):
        output_perm_T[j] = i
    # Validate permutation inverse calculation
    if config.get('diagnostics', False):
        print(f"\n{'='*70}")
        print(f"Permutation Validation")
        print(f"{'='*70}")
        print(f"input_perm (initial pi): {input_perm}")
        print(f"output_perm (final pi): {output_perm}")
        
        # Compute inverse
        output_perm_T = [0] * circ.get_Qbit_Num()
        for i, j in enumerate(output_perm):
            output_perm_T[j] = i
        print(f"output_perm_T (inverse): {output_perm_T}")
        
        # Verify inverse: output_perm_T[output_perm[i]] should equal i
        test_inverse = [output_perm_T[output_perm[i]] for i in range(len(output_perm))]
        if test_inverse != list(range(len(output_perm))):
            print(f"  ERROR: Inverse calculation is WRONG!")
            print(f"  Expected: {list(range(len(output_perm)))}")
            print(f"  Got: {test_inverse}")
        else:
            print(f"  Inverse verified: OK")
    
    if not config.get('diagnostics', False):
        output_perm_T = [0] * circ.get_Qbit_Num()
        for i, j in enumerate(output_perm):
            output_perm_T[j] = i
    
    circ_Final.add_Permutation(input_perm)
    circ_Final.add_Circuit(circ)
    circ_Final.add_Permutation(output_perm_T)
    
    # Additional matrix validation in example
    if config.get('diagnostics', False):
        try:
            print(f"\n{'='*70}")
            print(f"Final Circuit Matrix Validation")
            print(f"{'='*70}")
            orig_matrix = circ_orig.get_Matrix(parameters_orig)
            final_matrix = circ_Final.get_Matrix(params)
            matrix_error = np.linalg.norm(orig_matrix - final_matrix, 'fro')
            print(f"Original vs Final circuit error: {matrix_error:.2e}")
            
            # Test without output permutation
            circ_test = Circuit(circ.get_Qbit_Num())
            circ_test.add_Permutation(input_perm)
            circ_test.add_Circuit(circ)
            test_matrix = circ_test.get_Matrix(params)
            test_error = np.linalg.norm(orig_matrix - test_matrix, 'fro')
            print(f"Without output perm error: {test_error:.2e}")
        except Exception as e:
            print(f"Matrix validation error: {e}")
    
    PartAM_state = initial_state.copy()
    circ_Final.apply_to(params, PartAM_state)
    state_error = 1 - abs(np.vdot(PartAM_state, original_state))
    print(f"\n{'='*70}")
    print(f"State Vector Validation")
    print(f"{'='*70}")
    print(f"Decomposition error on random state: {state_error:.10f}")
    print("--- %s seconds elapsed during optimization ---" % (time.time() - start_time))
    print(f"{'='*70}\n")


