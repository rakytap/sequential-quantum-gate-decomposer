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
def validate_result(circ_orig, parameters_orig, circ, params, input_perm, output_perm):
    """Validate decomposition by applying both circuits to a random state."""
    num_qubits = circ.get_Qbit_Num()
    matrix_size = 1 << num_qubits
    initial_state_real = np.random.uniform(-1.0, 1.0, (matrix_size,))
    initial_state_imag = np.random.uniform(-1.0, 1.0, (matrix_size,))
    initial_state = initial_state_real + initial_state_imag * 1j
    initial_state = initial_state / np.linalg.norm(initial_state)

    original_state = initial_state.copy()
    circ_orig.apply_to(parameters_orig, original_state)

    circ_Final = Circuit(num_qubits)
    output_perm_T = [0] * num_qubits
    for i, j in enumerate(output_perm):
        output_perm_T[j] = i
    input_perm_list = [int(x) for x in input_perm]
    circ_Final.add_Permutation(input_perm_list)
    circ_Final.add_Circuit(circ)
    circ_Final.add_Permutation(output_perm_T)

    PartAM_state = initial_state.copy()
    circ_Final.apply_to(params, PartAM_state)
    state_error = 1 - abs(np.vdot(PartAM_state, original_state))
    return state_error, circ_Final


if __name__ == '__main__':

    filename = "bv_n14.qasm"
    circ_orig, parameters_orig = utils.qasm_to_squander_circuit(filename)
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),(12, 13)]

    # ================================================================
    # Full-circuit mode (default, window_size=0)
    # ================================================================
    print(f"\n{'='*70}")
    print("Full-circuit mode (window_size=0)")
    print(f"{'='*70}")

    config_full = {
        'strategy': "TreeSearch",
        'test_subcircuits': True,
        'test_final_circuit': True,
        'max_partition_size': 4,
        'progressbar': True,
        'topology': topology,
    }

    start_time = time.time()
    pam_full = Partition_Aware_Mapping(config_full)
    circ_full, params_full, input_perm_full, output_perm_full = \
        pam_full.Partition_Aware_Mapping(circ_orig, parameters_orig)
    elapsed_full = time.time() - start_time

    error_full, circ_final_full = validate_result(
        circ_orig, parameters_orig,
        circ_full, params_full, input_perm_full, output_perm_full
    )
    print(f"Decomposition error: {error_full:.10f}")
    print(f"Gate counts: {circ_final_full.get_Gate_Nums()}")
    print(f"Time: {elapsed_full:.2f}s")

    # ================================================================
    # Windowed mode (window_size=3)
    # ================================================================
    print(f"\n{'='*70}")
    print("Windowed mode (window_size=3)")
    print(f"{'='*70}")

    config_windowed = {
        'strategy': "TreeSearch",
        'test_subcircuits': True,
        'test_final_circuit': True,
        'max_partition_size': 4,
        'progressbar': True,
        'topology': topology,
        'window_size': 7,
    }

    start_time = time.time()
    pam_windowed = Partition_Aware_Mapping(config_windowed)
    circ_win, params_win, input_perm_win, output_perm_win = \
        pam_windowed.Partition_Aware_Mapping(circ_orig, parameters_orig)
    elapsed_win = time.time() - start_time

    error_win, circ_final_win = validate_result(
        circ_orig, parameters_orig,
        circ_win, params_win, input_perm_win, output_perm_win
    )
    print(f"Decomposition error: {error_win:.10f}")
    print(f"Gate counts: {circ_final_win.get_Gate_Nums()}")
    print(f"Time: {elapsed_win:.2f}s")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Mode':<20} {'Error':<20} {'Time':<10}")
    print(f"{'Full circuit':<20} {error_full:<20.10f} {elapsed_full:<10.2f}s")
    print(f"{'Windowed (K=3)':<20} {error_win:<20.10f} {elapsed_win:<10.2f}s")
    print(f"{'='*70}\n")



