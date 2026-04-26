"""
Benchmark PartAM per circuit without the cleanup phase.

Runs each circuit N_RUNS times with PartAM (cleanup=False) and records:
  - qubit count
  - initial CNOT count (original QASM circuit)
  - routed CNOT count (final, no cleanup)
  - decomposition error
  - compilation time (seconds)
  - routing time (seconds)

Results are exported to benchmark_PartAM_no_cleanup_layout.csv.

Usage:
    conda activate qgd
    python benchmark_PartAM_no_cleanup.py
"""

import csv
import glob
import os
import random
import time

import numpy as np

from squander import Circuit
from squander import Partition_Aware_Mapping
from squander import utils

N_RUNS = 3
OUTPUT_CSV = "benchmark_PartAM_no_cleanup_layout.csv"


def validate_result(circ_orig, parameters_orig, circ, params, input_perm, output_perm):
    num_qubits = circ.get_Qbit_Num()
    matrix_size = 1 << num_qubits
    rng = np.random.RandomState(0)
    initial_state = rng.uniform(-1, 1, (matrix_size,)) + 1j * rng.uniform(
        -1, 1, (matrix_size,)
    )
    initial_state /= np.linalg.norm(initial_state)

    original_state = initial_state.copy()
    circ_orig.apply_to(parameters_orig, original_state)

    circ_final = Circuit(num_qubits)
    output_perm_t = [0] * num_qubits
    for i, j in enumerate(output_perm):
        output_perm_t[j] = i
    circ_final.add_Permutation([int(x) for x in input_perm])
    circ_final.add_Circuit(circ)
    circ_final.add_Permutation(output_perm_t)

    state = initial_state.copy()
    circ_final.apply_to(params, state)
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
        'cleanup': False,
        'sabre_iterations': 20,
        'n_layout_trials': 256,
        'random_seed': random.randint(1, 100),
        'prefilter_top_k': 5000,
        'top_k_pi': 1,
        'cnot_cost': 1.0 / 3.0,  # old: swap_cost=15, local_cost_weight=0.1 -> 150:1 swap:cnot
        "parallel_layout_trials": True,
        "layout_trial_workers": 0,
        'max_E_size': 40,
        'max_lookahead': 6,
        'E_weight': 0.7,
        'E_alpha': 1.0,        # LightSABRE-style uniform lookahead (no per-depth decay)
        'decay_delta': 0.001,  # Qiskit LightSABRE DECAY_RATE
        'swap_burst_budget': 5,
        'three_qubit_exit_weight': 0.5,
        'log_schedule': True,
    }

    start = time.time()
    pam = Partition_Aware_Mapping(config)
    circ, params, pi_in, pi_out = pam.Partition_Aware_Mapping(
        circ_orig, parameters_orig
    )
    elapsed = time.time() - start
    routing_time = pam._routing_time
    cnot_final = circ.get_Gate_Nums().get('CNOT', 0)
    error = validate_result(circ_orig, parameters_orig, circ, params, pi_in, pi_out)

    return cnot_final, error, elapsed, routing_time


if __name__ == '__main__':
    circs_dir = "circs"
    qasm_files = sorted(glob.glob(os.path.join(circs_dir, "*.qasm")))

    if not qasm_files:
        print(f"No .qasm files found in {circs_dir}/")
        exit(1)

    print(f"Found {len(qasm_files)} circuits in {circs_dir}/")
    print(f"Running {N_RUNS} times per circuit (cleanup=False)\n")

    fieldnames = [
        'circuit',
        'n_qubits',
        'run',
        'initial_cnot',
        'cnot_final',
        'error',
        'time_s',
        'routing_time_s',
    ]

    # Open CSV once and flush after each circuit so partial results are never lost.
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for filepath in qasm_files:
            name = os.path.basename(filepath)
            print(f"{'=' * 70}")
            print(f"Circuit: {name}")

            circ_orig, parameters_orig, _ = utils.qasm_to_squander_circuit(filepath)
            n_qubits = circ_orig.get_Qbit_Num()
            topology = make_linear_topology(n_qubits)

            initial_cnot = circ_orig.get_Gate_Nums().get('CNOT', 0)
            print(f"Qubits: {n_qubits}, Initial CNOTs: {initial_cnot}")
            print(
                f"{'Run':>4} {'Final CNOT':>12} {'Error':>12} "
                f"{'Time(s)':>10} {'Routing time(s)':>15}"
            )

            for run_idx in range(N_RUNS):
                cnot_final, error, elapsed, routing_time = run_once(
                    circ_orig, parameters_orig, topology
                )
                print(
                    f"{run_idx:>4} {cnot_final:>12} {error:>12.2e} "
                    f"{elapsed:>10.1f} {routing_time:>15.1f}"
                )
                writer.writerow(
                    {
                        'circuit': name,
                        'n_qubits': n_qubits,
                        'run': run_idx,
                        'initial_cnot': initial_cnot,
                        'cnot_final': cnot_final,
                        'error': error,
                        'time_s': round(elapsed, 3),
                        'routing_time_s': round(routing_time, 3),
                    }
                )
                f.flush()

            print()

    print(f"Results saved to {OUTPUT_CSV}")
