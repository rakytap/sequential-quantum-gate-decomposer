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
## \file PartAM_example.py
## \brief Example demonstrating Partition Aware Mapping

from squander import Partition_Aware_Mapping
from squander import utils
from squander import Circuit
import numpy as np
import time


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


def run_and_report(label, config, circ_orig, parameters_orig):
    """Run PartAM with the given config and print results."""
    print(f"\n{'='*70}")
    print(label)
    print(f"{'='*70}")

    start_time = time.time()
    pam = Partition_Aware_Mapping(config)
    circ, params, input_perm, output_perm = pam.Partition_Aware_Mapping(circ_orig, parameters_orig)
    elapsed = time.time() - start_time

    error, circ_final = validate_result(
        circ_orig, parameters_orig, circ, params, input_perm, output_perm
    )
    print(f"Decomposition error: {error:.10f}")
    print(f"Gate counts: {circ_final.get_Gate_Nums()}")
    print(f"Time: {elapsed:.2f}s")
    return error, elapsed


if __name__ == '__main__':

    filename = "bv_n14.qasm"
    circ_orig, parameters_orig = utils.qasm_to_squander_circuit(filename)
    topology = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
    ]

    results = {}

    # ================================================================
    # Default: single forward pass (sabre_iterations=0)
    # ================================================================
    results['default'] = run_and_report("Default (single forward pass)", {
        'strategy': "TreeSearch",
        'max_partition_size': 4,
        'progressbar': True,
        'topology': topology,
        'sabre_iterations': 0,
    }, circ_orig, parameters_orig)

    # ================================================================
    # SABRE-style layout refinement (sabre_iterations=3)
    # ================================================================
    results['sabre'] = run_and_report("SABRE iterations=3", {
        'strategy': "TreeSearch",
        'max_partition_size': 4,
        'progressbar': True,
        'topology': topology,
        'sabre_iterations': 3,
    }, circ_orig, parameters_orig)

    # ================================================================
    # Multiple layout trials with SABRE iterations
    # ================================================================
    results['trials'] = run_and_report("SABRE iterations=3, layout trials=5", {
        'strategy': "TreeSearch",
        'max_partition_size': 4,
        'progressbar': True,
        'topology': topology,
        'sabre_iterations': 3,
        'n_layout_trials': 5,
        'random_seed': 42,
    }, circ_orig, parameters_orig)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Mode':<40} {'Error':<20} {'Time':<10}")
    for label, (error, elapsed) in results.items():
        print(f"{label:<40} {error:<20.10f} {elapsed:<10.2f}s")
    print(f"{'='*70}\n")
