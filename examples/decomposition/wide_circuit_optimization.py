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
from squander.decomposition.qgd_Wide_Circuit_Optimization import (
    CNOTGateCount, SingleQubitGateCount, TotalRawGateCount, CircuitGateStats,
)
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander import utils
from squander import Qiskit_IO
import time, requests, os, zipfile, tempfile, json, numpy as np
from pathlib import Path
from collections import Counter

# IBM Eagle native gate set (QMill benchmark basis)
IBM_EAGLE_BASIS = ["cx", "rz", "sx", "x"]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARTITIONING_BENCHMARK_ROOT = REPOSITORY_ROOT / "benchmarks" / "partitioning"
BENCHMARK_DATASETS = {
    "IBMEagle": PARTITIONING_BENCHMARK_ROOT / "IBMEagle",
    "QASMBenchmarks": PARTITIONING_BENCHMARK_ROOT / "QASMBenchmarks",
}


def transpile_to_ibm_eagle(qasm_path_or_circ, parameters=None):
    """Transpile a circuit to IBM Eagle's native gate set and return consistent
    gate stats via Squander's CircuitGateStats.

    Args:
        qasm_path_or_circ: Either a path to a .qasm file, or a Squander Circuit.
        parameters: Required if passing a Squander Circuit.

    Returns:
        dict from CircuitGateStats, using the same decomposition as the main
        benchmark pipeline.
    """
    from qiskit import QuantumCircuit, transpile

    if isinstance(qasm_path_or_circ, (str, os.PathLike)):
        qc = QuantumCircuit.from_qasm_file(str(qasm_path_or_circ))
    else:
        qc = Qiskit_IO.get_Qiskit_Circuit(
            qasm_path_or_circ,
            np.asarray(parameters if parameters is not None else [], dtype=np.float64),
        )

    transpiled = transpile(qc, basis_gates=IBM_EAGLE_BASIS, optimization_level=0)
    eagle_circ, eagle_params = Qiskit_IO.convert_Qiskit_to_Squander(transpiled)
    return CircuitGateStats(eagle_circ)


def save_qasm2(circuit, parameters, output_path):
    """Atomically export a Squander circuit as OpenQASM 2 via Qiskit I/O."""
    from qiskit import qasm2

    qiskit_circuit = Qiskit_IO.get_Qiskit_Circuit(
        circuit, np.asarray(parameters, dtype=np.float64)
    )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        qasm2.dump(qiskit_circuit, str(temporary_path))
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def result_paths(max_partition_size):
    """Return result directories and JSON path for a 3- or 4-qubit run."""
    if max_partition_size not in (3, 4):
        raise ValueError(
            "Archived wide-circuit results support max_partition_size 3 or 4, "
            f"not {max_partition_size}"
        )
    suffix = f"{max_partition_size}qbit"
    result_directories = {
        dataset: PARTITIONING_BENCHMARK_ROOT / f"{dataset}_results_{suffix}"
        for dataset in BENCHMARK_DATASETS
    }
    for result_directory in result_directories.values():
        result_directory.mkdir(parents=True, exist_ok=True)
    return result_directories, REPOSITORY_ROOT / f"results_{suffix}.json"


if __name__ == "__main__":

    config = {
        "strategy": "TreeSearch",  # possible values: "TreeSearch", "qiskit", "bqskit", "TabuSearch"
        "test_subcircuits": False,
        "test_final_circuit": False,
        "max_partition_size": 3,
        "beam": None,
        "use_osr": True,
        "use_graph_search": True,
        "auto_expand_partition_size": False,
        "pre-opt-strategy": "TreeSearch",  # possible values: "TreeSearch", "qiskit", "bqskit", "TabuSearch"
        "routing-strategy": "seqpam-ilp",  # possible values: "sabre", "light-sabre", "bqskit-sabre", "seqpam-quick", "seqpam-ilp"
        #"tolerance": 1e-8, #1e-8 for use_float and 1e-10 if not are sensible but leave to default
        "use_float": True,  # whether to use single precision for the optimization (experimental, may cause instability in some cases, but can significantly reduce optimization time and memory usage for large circuits)
        # **{'use_basin_hopping': True, 'bh_T': 1.1822334624366124, 'bh_stepsize': 0.9020671823381502, 'bh_interval': 165, 'bh_target_accept_rate': 0.7037812116166546, 'bh_stepwise_factor': 0.8254028860713254}
    }
    result_directories, RESULTS_FILE = result_paths(config["max_partition_size"])

    def get_circuit_stats(filepath):
        """Return (cnot_count, qubit_count) for a QASM file via squander parsing."""
        circ, _, _ = utils.qasm_to_squander_circuit(str(filepath))
        return CNOTGateCount(circ, 0), circ.get_Qbit_Num()

    # Inputs are curated in-repository, so no generated-file or reset filtering
    # is needed here. Each final circuit is written under the same filename in
    # the dataset's corresponding result directory.
    files = []
    for dataset, input_directory in BENCHMARK_DATASETS.items():
        result_directory = result_directories[dataset]
        for filepath in input_directory.glob("*.qasm"):
            output_path = result_directory / filepath.name
            files.append(
                (get_circuit_stats(filepath), dataset, filepath, output_path)
            )
    files.sort(key=lambda item: item[0])

    if len(files) != 77:
        raise RuntimeError(
            f"Expected 77 archived benchmark circuits, found {len(files)}"
        )

    print("=== Running circuits in order of (CNOT_count, qubit_count) ===")
    for (cnots, qubits), dataset, filepath, _ in files:
        print(
            f"  {cnots:>8} CNOT, {qubits:>4} qubits  "
            f"{dataset}/{filepath.name}"
        )
    print("=" * 60)


    # JSON results file with resume support
    import json

    # Load existing results to skip already-processed circuits
    existing_results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            existing_results = json.load(f)
        for entry in existing_results.values():
            if isinstance(entry, dict):
                entry.pop("init_ibm_eagle", None)
                entry.pop("final_ibm_eagle", None)
        print(f"Loaded {len(existing_results)} existing results; will skip already-processed circuits.")

    results = existing_results  # merge new results into existing
    completed_count = sum(
        1
        for _, _, filename, output_path in files
        if filename.name in results and output_path.is_file()
    )
    for _, dataset, filename, output_path in files:
        fname = filename.name
        if fname in results and output_path.is_file():
            print(f"Skipping already processed: {fname}")
            continue

        print(f"executing optimization of circuit: {filename}")
        #if not filename.endswith("_n140.qasm"): continue

        # load the circuit from a file
        circ, parameters, _ = utils.qasm_to_squander_circuit(str(filename))
        config["topology"] = (
            Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization.linear_topology(
                circ.get_Qbit_Num()
            )
        )

        # pre-optimization stats
        init_stats = CircuitGateStats(circ)

        # run circuit optimization
        wide_circuit_optimizer = (
            Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization({**config})
        )
        start_time = time.time()
        optcirc, optparameters = wide_circuit_optimizer.OptimizeWideCircuit(
            circ, parameters
        )
        elapsed = time.time() - start_time

        # post-optimization stats
        opt_stats = CircuitGateStats(optcirc)
        opt_time = wide_circuit_optimizer.config.get("optimization_time", None)

        # routing stats (if routing was needed)
        a2a_stats = None
        routed_stats = None
        a2a_time = routing_time = None
        if wide_circuit_optimizer.config.get("routed_circuit", None) is not None:
            a2acirc = wide_circuit_optimizer.config["all_to_all_circuit"]
            routedcirc = wide_circuit_optimizer.config["routed_circuit"]
            a2a_stats = CircuitGateStats(a2acirc)
            routed_stats = CircuitGateStats(routedcirc)
            a2a_time = wide_circuit_optimizer.config.get("all_to_all_optimization_time", None)
            routing_time = wide_circuit_optimizer.config.get("routing_time", None)

        result_entry = {
            "file": fname,
            "dataset": dataset,
            "output_file": str(output_path.relative_to(REPOSITORY_ROOT)),
            "strategy": config["strategy"],
            "pre_opt_strategy": config["pre-opt-strategy"],
            "routing_strategy": config["routing-strategy"],
            "init": init_stats,
            "final": opt_stats,
            "timing": {
                "a2a": round(a2a_time, 2) if a2a_time else None,
                "routing": round(routing_time, 2) if routing_time else None,
                "optimization": round(opt_time, 2) if opt_time else None,
                "total": round(elapsed, 2),
            },
        }
        if a2a_stats:
            result_entry["all_to_all"] = a2a_stats
        if routed_stats:
            result_entry["routed"] = routed_stats

        wide_circuit_optimizer.check_compare_circuits(
            circ,
            parameters,
            optcirc,
            optparameters,
            routing=wide_circuit_optimizer.config.get("routed_circuit", None) is not None,
            label="example final original-to-output",
        )

        save_qasm2(optcirc, optparameters, output_path)
        print(f"  wrote verified result: {output_path}")

        results[fname] = result_entry
        completed_count += 1

        # Save after each verified circuit so interrupted runs can resume.
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  init: {init_stats['cnot_equiv']} CNOT, {init_stats['single_qubit']} 1q, "
              f"{init_stats['total_raw']} total, {init_stats['qubits']}q")
        print(f"  final: {opt_stats['cnot_equiv']} CNOT, {opt_stats['single_qubit']} 1q, "
              f"{opt_stats['total_raw']} total")
        print(f"  time: {elapsed:.2f}s")
        print(f"--- {completed_count}/{len(files)} circuits processed ---")

        print("--- %s seconds elapsed during optimization ---" % elapsed)
