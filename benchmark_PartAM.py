"""
Benchmark PartAM cleanup phase per circuit.

Runs each circuit 5 times with PartAM (cleanup=True) and records:
  - qubit count
  - initial CNOT count (original QASM circuit)
  - CNOT count before cleanup (post-synthesis, pre-cleanup)
  - CNOT count after cleanup (final)
  - decomposition error
  - compilation time (seconds)

Results are exported to benchmark_PartAM.csv.

Usage:
    conda activate qgd
    python benchmark_PartAM.py
"""

import numpy as np
import time
import os
import glob
import csv
import random

from squander import Partition_Aware_Mapping
from squander import utils
from squander import Circuit

N_RUNS = 3
OUTPUT_CSV = "benchmark_PartAM_layout.csv"


def validate_result(circ_orig, parameters_orig, circ, params, input_perm, output_perm):
    num_qubits = circ.get_Qbit_Num()
    matrix_size = 1 << num_qubits
    rng = np.random.RandomState(0)
    initial_state = rng.uniform(-1, 1, (matrix_size,)) + 1j * rng.uniform(-1, 1, (matrix_size,))
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


def make_linear_topology(n_qubits):
    return [(i, i + 1) for i in range(n_qubits - 1)]


def run_once(circ_orig, parameters_orig, topology):
    config = {
        'strategy': "TreeSearch",
        'test_subcircuits': False,
        'test_final_circuit': False,
        'max_partition_size': 3,
        'progressbar': False,
        'topology': topology,
        'verbosity': 0,
        'cleanup': True,
        'sabre_iterations':20,
        'n_layout_trials':128,
        'random_seed':random.randint(1,100),
        # Cheap candidate prefilter before full A* scoring.
        'prefilter_top_k': 50,
        # Rank every layout trial by actual constructed routing, not only by
        # the heuristic trial cost.
        'actual_routing_rank_top_k': None,
        'top_k_pi': 1,
        'cnot_cost': 0.5 / 3.0,  # old: swap_cost=15, local_cost_weight=1.0 -> 15:1 swap:cnot
        'cleanup_top_k': 3,
        "parallel_layout_trials": True,
        "layout_trial_workers": 0,
        'max_E_size': 40,
        'max_lookahead': 6,
        'E_weight': 0.3,
        'E_alpha': 1.0,        # LightSABRE-style uniform lookahead (no per-depth decay)
        # Disable extra routing heuristics while diagnosing 3-qubit partition
        # quality.
        'decay_delta': 0.0,
        'swap_burst_budget': 5,
        'path_tiebreak_weight': 0.0,
        'three_qubit_exit_weight': 1.5,
    }

    # Clean the initial circuit using the same config pattern as in PartAM.py
    from squander.decomposition.qgd_Wide_Circuit_Optimization import qgd_Wide_Circuit_Optimization
    cleanup_config = dict(config)
    cleanup_config['topology'] = None
    cleanup_config['routed'] = False
    cleanup_config['test_subcircuits'] = False
    cleanup_config['test_final_circuit'] = False
    cleanup_config['global_min'] = True
    cleanup_config['pre-opt-strategy'] = 'TreeSearch'

    wco = qgd_Wide_Circuit_Optimization(cleanup_config)
    #circ_orig, parameters_orig = wco.OptimizeWideCircuit(circ_orig.get_Flat_Circuit(), parameters_orig)

    start = time.time()
    pam = Partition_Aware_Mapping(config)
    circ, params, pi_in, pi_out = pam.Partition_Aware_Mapping(circ_orig.get_Flat_Circuit(), parameters_orig)
    elapsed = time.time() - start
    routing_time = pam._routing_time
    cnot_before_cleanup = pam._cnot_pre_cleanup
    cnot_after_cleanup = circ.get_Gate_Nums().get('CNOT', 0)
    error = validate_result(circ_orig, parameters_orig, circ, params, pi_in, pi_out)

    return cnot_before_cleanup, cnot_after_cleanup, error, elapsed, routing_time


if __name__ == '__main__':
    circs_dir = "circs"
    qasm_files = sorted(glob.glob(os.path.join(circs_dir, "*.qasm")))

    if not qasm_files:
        print(f"No .qasm files found in {circs_dir}/")
        exit(1)

    print(f"Found {len(qasm_files)} circuits in {circs_dir}/")
    print(f"Running {N_RUNS} times per circuit (cleanup=True)\n")

    fieldnames = [
        'circuit', 'n_qubits', 'run',
        'initial_cnot', 'cnot_pre_cleanup', 'cnot_post_cleanup',
        'error', 'time_s','routing_time_s'
    ]

    # Open CSV once and flush after each circuit so partial results are never lost
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for filepath in qasm_files:
            name = os.path.basename(filepath)
            print(f"{'='*70}")
            print(f"Circuit: {name}")

            circ_orig, parameters_orig, _ = utils.qasm_to_squander_circuit(filepath)
            n_qubits = circ_orig.get_Qbit_Num()
            topology = make_linear_topology(n_qubits)

            initial_cnot = circ_orig.get_Gate_Nums().get('CNOT', 0)
            print(f"Qubits: {n_qubits}, Initial CNOTs: {initial_cnot}")
            print(f"{'Run':>4} {'Pre-cleanup':>12} {'Post-cleanup':>12} {'Error':>12} {'Time(s)':>10} {'Routing time(s)':>10}")

            for run_idx in range(N_RUNS):
                cnot_pre, cnot_post, error, elapsed, routing_time = run_once(circ_orig, parameters_orig, topology)
                print(f"{run_idx:>4} {cnot_pre:>12} {cnot_post:>12} {error:>12.2e} {elapsed:>10.1f} {routing_time:>10.1f}")
                writer.writerow({
                    'circuit': name,
                    'n_qubits': n_qubits,
                    'run': run_idx,
                    'initial_cnot': initial_cnot,
                    'cnot_pre_cleanup': cnot_pre,
                    'cnot_post_cleanup': cnot_post,
                    'error': error,
                    'time_s': round(elapsed, 3),
                    'routing_time_s': round(routing_time,3)
                })
                f.flush()

            print()

    print(f"Results saved to {OUTPUT_CSV}")
