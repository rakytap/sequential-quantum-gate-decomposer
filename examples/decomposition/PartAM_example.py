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
## \brief Simple example python code demonstrating Partition Aware Mapping

import time
import numpy as np

from squander import Partition_Aware_Mapping
from squander import utils
from squander import Circuit
from squander.decomposition.qgd_Wide_Circuit_Optimization import (
    qgd_Wide_Circuit_Optimization,
)


def make_linear_topology(n_qubits):
    return [(i, i + 1) for i in range(n_qubits - 1)]


def validate_result(circ_orig, parameters_orig, circ, params, input_perm, output_perm):
    """Apply both circuits to a random state and return ``1 - |<psi|phi>|``."""
    num_qubits = circ.get_Qbit_Num()
    matrix_size = 1 << num_qubits
    rng = np.random.RandomState(0)
    initial_state = (
        rng.uniform(-1, 1, (matrix_size,))
        + 1j * rng.uniform(-1, 1, (matrix_size,))
    )
    initial_state /= np.linalg.norm(initial_state)

    original_state = initial_state.copy()
    circ_orig.apply_to(parameters_orig, original_state)

    circ_Final = Circuit(num_qubits)
    output_perm_T = [0] * num_qubits
    for i, j in enumerate(output_perm):
        output_perm_T[j] = i
    circ_Final.add_Permutation([int(x) for x in input_perm])
    circ_Final.add_Circuit(circ)
    circ_Final.add_Permutation(output_perm_T)

    state = initial_state.copy()
    circ_Final.apply_to(params, state)
    return 1 - abs(np.vdot(state, original_state))


if __name__ == '__main__':

    filename = "bv_n14.qasm"

    # load the circuit from a file
    circ_orig, parameters_orig = utils.qasm_to_squander_circuit(filename)
    N = circ_orig.get_Qbit_Num()
    topology = make_linear_topology(N)

    initial_cnot = circ_orig.get_Gate_Nums().get('CNOT', 0)
    print(f"Qubits: {N}, initial CNOTs: {initial_cnot}")

    start_time = time.time()

    # one-shot WCO pass before PartAM (topology=None, max_partition_size=3,
    # part_size_end=4) to fuse trivially-mergeable blocks
    pre_partam_cleanup_config = {
        'strategy': 'TreeSearch',
        'pre-opt-strategy': 'TreeSearch',
        'partition_strategy': 'ilp',
        'test_subcircuits': False,
        'test_final_circuit': False,
        'max_partition_size': 3,
        'topology': None,
        'verbosity': 0,
        'tolerance': 1e-8,
        'parallel': 0,
        'part_size_end': 4,
    }
    wco = qgd_Wide_Circuit_Optimization(pre_partam_cleanup_config)
    pre_partam_circ, pre_partam_params = wco.OptimizeWideCircuit(
        circ_orig.get_Flat_Circuit(),
        parameters_orig,
    )
    pre_partam_cleanup_cnot = pre_partam_circ.get_Gate_Nums().get('CNOT', 0)
    print(f"PartAM input CNOTs after pre-cleanup: {pre_partam_cleanup_cnot}")

    # PartAM config
    config = {
        'strategy': "TreeSearch",
        'test_subcircuits': False,
        'test_final_circuit': False,
        'max_partition_size': 3,
        'progressbar': False,
        'topology': topology,
        'verbosity': 0,
        'cleanup': True,
        'sabre_iterations': 20,
        'n_layout_trials': 128,
        'random_seed': 42,
        # Cheap candidate prefilter before full A* scoring.
        'prefilter_top_k': 400,
        'prefilter_min_per_partition': 2,
        'prefilter_min_3q': 12,
        # Rank every layout trial by actual constructed routing, not only by
        # the heuristic trial cost.  QFT is sensitive to this cap.
        'actual_routing_rank_top_k': None,
        # Boundary-state beam routing runs in the C++ SABRE router.
        'use_cpp_router': True,
        'layout_boundary_beam_width': 4,
        'layout_boundary_beam_depth': 3,
        'boundary_beam_width': 4,
        'boundary_beam_depth': 3,
        'cnot_cost': 0.5 / 3.0,
        'cleanup_top_k': 3,
        'parallel_layout_trials': True,
        'layout_trial_workers': 0,
        'max_E_size': 40,
        'max_lookahead': 6,
        'E_weight': 0.3,
        'E_alpha': 1.0,  # LightSABRE-style uniform lookahead (no per-depth decay)
        'decay_delta': 0.001,
        'swap_burst_budget': 0,
        'path_tiebreak_weight': 0.2,
        'three_qubit_exit_weight': 1.5,
        'partition_weight_model': 'window_turnover',
        'pack_credit_weight': 1.0,
        'partition_chain_penalty_weight': 2.5,
    }

    # instantiate the object for Partition Aware Mapping
    pam = Partition_Aware_Mapping(config)

    # run Partition Aware Mapping
    circ, params, input_perm, output_perm = pam.Partition_Aware_Mapping(
        pre_partam_circ.get_Flat_Circuit(), pre_partam_params
    )

    elapsed = time.time() - start_time

    error = validate_result(
        circ_orig, parameters_orig, circ, params, input_perm, output_perm
    )

    print(f"CNOTs pre-cleanup: {pam._cnot_pre_cleanup}")
    print(f"CNOTs post-cleanup: {circ.get_Gate_Nums().get('CNOT', 0)}")
    print(f"Decomposition error: {error:.10f}")
    print("--- %s seconds elapsed during optimization ---" % elapsed)
