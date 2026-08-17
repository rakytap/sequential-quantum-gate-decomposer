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
import squander.partitioning.routing as Exact_Routing
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import signal
import time
import traceback
import numpy as np
from pathlib import Path

# IBM Eagle native gate set (QMill benchmark basis)
IBM_EAGLE_BASIS = ["cx", "rz", "sx", "x"]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARTITIONING_BENCHMARK_ROOT = REPOSITORY_ROOT / "benchmarks" / "partitioning"
BENCHMARK_DATASETS = {
    "IBMEagle": PARTITIONING_BENCHMARK_ROOT / "IBMEagle",
    "QASMBenchmarks": PARTITIONING_BENCHMARK_ROOT / "QASMBenchmarks",
}
DATASET_STATS_CACHE = PARTITIONING_BENCHMARK_ROOT / "dataset_stats.json"
DATASET_STATS_SCHEMA_VERSION = 1


def runtime_source_fingerprint():
    """Identify the exact Python implementation loaded by a benchmark run."""
    digest = hashlib.sha256()
    source_paths = (
        Path(__file__).resolve(),
        Path(Wide_Circuit_Optimization.__file__).resolve(),
        Path(Exact_Routing.__file__).resolve(),
    )
    for source_path in source_paths:
        digest.update(str(source_path).encode("utf-8"))
        digest.update(b"\0")
        with source_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


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


def audit_paths(output_path):
    """Return the live JSONL and compressed audit paths beside a result QASM."""
    output_path = Path(output_path)
    stem = output_path.with_suffix("")
    return Path(f"{stem}.audit.jsonl"), Path(f"{stem}.audit.json.gz")


def routing_catalog_path(output_path):
    """Return the reusable, non-proof routing catalog beside a result QASM."""
    output_path = Path(output_path)
    return Path(f"{output_path.with_suffix('')}.routing-catalog.json.gz")


def clear_invalidated_result_artifacts(output_path):
    """Remove generated relics when a circuit has no ``results.json`` entry.

    The result directory is separate from the archived benchmark inputs.  A
    basename-qualified match therefore removes the output QASM, live/compressed
    audits, routing catalog, temporary files, and diagnostics without touching
    any other circuit or its source QASM.
    """
    output_path = Path(output_path)
    artifact_prefix = f"{output_path.with_suffix('').name}."
    removed = []
    if not output_path.parent.is_dir():
        return removed
    for artifact in output_path.parent.iterdir():
        if (
            artifact.name.startswith(artifact_prefix)
            and (artifact.is_file() or artifact.is_symlink())
        ):
            artifact.unlink(missing_ok=True)
            removed.append(artifact)
    return removed


def file_sha256(path):
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset_stats_cache(cache_path=DATASET_STATS_CACHE):
    """Load content-addressed benchmark sorting statistics."""
    try:
        with Path(cache_path).open() as stream:
            cache = json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if (
        not isinstance(cache, dict)
        or cache.get("schema_version") != DATASET_STATS_SCHEMA_VERSION
        or not isinstance(cache.get("files"), dict)
    ):
        return {}
    return cache["files"]


def cached_circuit_stats(filepath, cache):
    """Return sortable circuit stats, parsing only on a content cache miss."""
    filepath = Path(filepath)
    relative_path = filepath.relative_to(REPOSITORY_ROOT).as_posix()
    digest = file_sha256(filepath)
    cached = cache.get(relative_path)
    if (
        isinstance(cached, dict)
        and cached.get("sha256") == digest
        and isinstance(cached.get("cnot_count"), int)
        and isinstance(cached.get("qubit_count"), int)
    ):
        return (
            (cached["cnot_count"], cached["qubit_count"]),
            cached,
            True,
        )

    circ, _, _ = utils.qasm_to_squander_circuit(str(filepath))
    entry = {
        "sha256": digest,
        "cnot_count": int(CNOTGateCount(circ, 0)),
        "qubit_count": int(circ.get_Qbit_Num()),
    }
    return (entry["cnot_count"], entry["qubit_count"]), entry, False


def save_dataset_stats_cache(entries, cache_path=DATASET_STATS_CACHE):
    """Atomically store only the currently archived benchmark inputs."""
    save_results(
        Path(cache_path),
        {
            "schema_version": DATASET_STATS_SCHEMA_VERSION,
            "files": dict(sorted(entries.items())),
        },
    )


def result_paths(max_partition_size, strategy):
    """Return strategy-specific result directories and their JSON files."""
    if max_partition_size not in (3, 4):
        raise ValueError(
            "Archived wide-circuit results support max_partition_size 3 or 4, "
            f"not {max_partition_size}"
        )
    if (
        not isinstance(strategy, str)
        or not strategy
        or not all(character.isalnum() or character in "-_" for character in strategy)
    ):
        raise ValueError(f"Invalid strategy name for result paths: {strategy!r}")
    suffix = f"{max_partition_size}qbit_{strategy}"
    result_directories = {
        dataset: PARTITIONING_BENCHMARK_ROOT / f"{dataset}_results_{suffix}"
        for dataset in BENCHMARK_DATASETS
    }
    for result_directory in result_directories.values():
        result_directory.mkdir(parents=True, exist_ok=True)
    result_files = {
        dataset: result_directory / "results.json"
        for dataset, result_directory in result_directories.items()
    }
    return result_directories, result_files


def optimize_circuit_worker(
    config,
    dataset,
    filename,
    output_path,
    expected_source_fingerprint,
    result_queue,
):
    """Optimize and archive one circuit in an isolated process."""
    old_audit_path = os.environ.get("SQUANDER_REWRITE_AUDIT_JSONL")
    try:
        # Protect the machine from parent-side Python/PuLP/Gurobi allocations.
        # Native routing children receive their own, smaller limits separately.
        Exact_Routing._set_process_address_space_limit(
            config, "circuit_worker_memory_limit_gib", 64.0
        )
        actual_source_fingerprint = runtime_source_fingerprint()
        if actual_source_fingerprint != expected_source_fingerprint:
            raise RuntimeError(
                "Benchmark source changed after this run started; restart the "
                "driver so the worker imports one consistent implementation. "
                f"started={expected_source_fingerprint[:16]}, "
                f"current={actual_source_fingerprint[:16]}."
            )
        # Give BQSKit and multiprocessing descendants a dedicated process group
        # that the parent can terminate together on timeout.
        if hasattr(os, "setsid"):
            os.setsid()

        filename = Path(filename)
        output_path = Path(output_path)
        audit_jsonl_path, audit_gzip_path = audit_paths(output_path)
        catalog_path = routing_catalog_path(output_path)
        audit_jsonl_path.unlink(missing_ok=True)
        os.environ["SQUANDER_REWRITE_AUDIT_JSONL"] = str(audit_jsonl_path)
        fname = filename.name
        circ, parameters, _ = utils.qasm_to_squander_circuit(str(filename))
        worker_config = dict(config)
        worker_config["topology"] = (
            Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization.linear_topology(
                circ.get_Qbit_Num()
            )
        )
        worker_config["exact_routing_catalog_output_path"] = str(catalog_path)

        init_stats = CircuitGateStats(circ)
        optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization(
            worker_config
        )
        start_time = time.monotonic()
        optcirc, optparameters = optimizer.OptimizeWideCircuit(circ, parameters)
        elapsed = time.monotonic() - start_time

        opt_stats = CircuitGateStats(optcirc)
        opt_time = optimizer.config.get("optimization_time")
        a2a_stats = None
        routed_stats = None
        a2a_time = None
        routing_time = None
        routed = optimizer.config.get("routed_circuit")
        if routed is not None:
            a2a_stats = CircuitGateStats(optimizer.config["all_to_all_circuit"])
            routed_stats = CircuitGateStats(routed)
            a2a_time = optimizer.config.get("all_to_all_optimization_time")
            routing_time = optimizer.config.get("routing_time")

        result_entry = {
            "file": fname,
            "dataset": dataset,
            "output_file": str(output_path.relative_to(REPOSITORY_ROOT)),
            "status": "completed",
            "strategy": config["strategy"],
            "pre_opt_strategy": config["pre-opt-strategy"],
            "routing_strategy": config["routing-strategy"],
            "source_fingerprint": expected_source_fingerprint,
            "configuration": result_configuration(
                optimizer.config, circ.get_Qbit_Num()
            ),
            "init": init_stats,
            "final": opt_stats,
            "timing": {
                "a2a": round(a2a_time, 2) if a2a_time is not None else None,
                "routing": (
                    round(routing_time, 2) if routing_time is not None else None
                ),
                "optimization": round(opt_time, 2) if opt_time is not None else None,
                "total": round(elapsed, 2),
            },
        }
        if a2a_stats is not None:
            result_entry["all_to_all"] = a2a_stats
        if routed_stats is not None:
            result_entry["routed"] = routed_stats

        optimizer.check_compare_circuits(
            circ,
            parameters,
            optcirc,
            optparameters,
            routing=routed is not None,
        )
        save_qasm2(optcirc, optparameters, output_path)
        input_representation = Wide_Circuit_Optimization._squander_audit_representation(
            circ, parameters, range(circ.get_Qbit_Num())
        )
        output_representation = Wide_Circuit_Optimization._squander_audit_representation(
            optcirc, optparameters, range(optcirc.get_Qbit_Num())
        )
        run_metadata = {
            "input_file": str(filename.relative_to(REPOSITORY_ROOT)),
            "output_file": str(output_path.relative_to(REPOSITORY_ROOT)),
            "dataset": dataset,
            "strategy": config["strategy"],
            "pre_opt_strategy": config["pre-opt-strategy"],
            "routing_strategy": config["routing-strategy"],
            "configuration": result_entry["configuration"],
            "source_fingerprint": expected_source_fingerprint,
            "initial_mapping": optimizer.config.get("initial_mapping"),
            "final_mapping": optimizer.config.get("final_mapping"),
            "input_circuit": input_representation,
            "output_circuit": output_representation,
            "input_circuit_sha256": (
                Wide_Circuit_Optimization._exact_state_sha256(
                    Wide_Circuit_Optimization._qasm_exact_state(
                        input_representation
                    )
                )
            ),
            "output_circuit_sha256": (
                Wide_Circuit_Optimization._exact_state_sha256(
                    Wide_Circuit_Optimization._qasm_exact_state(
                        output_representation
                    )
                )
            ),
            "input_file_sha256": file_sha256(filename),
            "output_file_sha256": file_sha256(output_path),
        }
        routing_catalog = optimizer.config.get("routing_osr_catalog")
        if routing_catalog is not None:
            routing_catalog = dict(routing_catalog)
            routing_catalog["file"] = str(
                Path(routing_catalog["file"]).relative_to(REPOSITORY_ROOT)
            )
            result_entry["routing_osr_catalog"] = routing_catalog
            run_metadata["routing_osr_catalog"] = routing_catalog
        rewrite_audit, audit_sha256 = optimizer.save_rewrite_audit(
            audit_jsonl_path, audit_gzip_path, run_metadata
        )
        audit_verification = optimizer.verify_rewrite_audit(
            audit_gzip_path, expected_sha256=audit_sha256
        )
        result_entry["rewrite_audit"] = {
            "schema_version": rewrite_audit["schema_version"],
            "file": str(audit_gzip_path.relative_to(REPOSITORY_ROOT)),
            "sha256": audit_sha256,
            "summary": rewrite_audit["summary"],
            **audit_verification,
        }
        audit_jsonl_path.unlink(missing_ok=True)
        result_queue.put({"ok": True, "entry": result_entry})
    except Exception:
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})
    finally:
        if old_audit_path is None:
            os.environ.pop("SQUANDER_REWRITE_AUDIT_JSONL", None)
        else:
            os.environ["SQUANDER_REWRITE_AUDIT_JSONL"] = old_audit_path


def terminate_circuit_process(process, grace_seconds=5.0):
    """Terminate a circuit worker and descendants without signaling this driver."""
    if not process.is_alive():
        process.join()
        return

    try:
        process_group = os.getpgid(process.pid)
    except (AttributeError, ProcessLookupError):
        process_group = None

    if process_group == process.pid:
        os.killpg(process_group, signal.SIGTERM)
    else:
        process.terminate()
    process.join(grace_seconds)

    if process.is_alive():
        try:
            process_group = os.getpgid(process.pid)
        except (AttributeError, ProcessLookupError):
            process_group = None
        if process_group == process.pid:
            os.killpg(process_group, signal.SIGKILL)
        else:
            process.kill()
        process.join()


def save_results(results_file, results):
    """Atomically save resumable benchmark metadata."""
    temporary_path = results_file.with_suffix(results_file.suffix + ".tmp")
    try:
        with open(temporary_path, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(temporary_path, results_file)
    finally:
        temporary_path.unlink(missing_ok=True)


def result_configuration(config, qubit_num):
    """Return the resolved optimization settings needed to reproduce a result."""
    keys = (
        "max_partition_size",
        "tolerance",
        "osr_optimization_tolerance",
        "synthesis_acceptance_tolerance",
        "circuit_validation_tolerance",
        "bqskit_synthesis_epsilon",
        "use_float",
        "use_osr",
        "use_graph_search",
        "auto_expand_partition_size",
        "partition_strategy",
        "routing_partition_strategy",
        "pam_swap_cnot_cost",
        "partition_workers",
        "routing_synthesis_workers",
        "routing_column_max_iteration_loops",
        "routing_column_synthesis_mode",
        "max_equal_cnot_optimization_rounds",
        "exact_routing_fallback_retry_count",
        "exact_routing_catalog_progress",
        "exact_routing_catalog_progress_interval",
        "exact_routing_light_sabre_seed_count",
        "exact_routing_light_sabre_trials_per_seed",
        "exact_routing_light_guided_cover_count",
        "exact_routing_minimum_cover_seed_count",
        "exact_routing_post_catalog_pam_seed_count",
        "exact_routing_pam_layout_passes",
        "exact_routing_layout_seed_count",
        "exact_routing_layout_total_passes",
        "exact_routing_layout_candidate_limit",
        "exact_routing_layout_portfolio_timeout_seconds",
        "exact_routing_pam_swap_cnot_costs",
        "exact_routing_pam_cover_strategies",
        "exact_routing_lazy_osr",
        "exact_routing_cover_pool_timeout_seconds",
        "exact_routing_token_seed_timeout_seconds",
        "exact_routing_token_seed_cover_strategies",
        "routing_synthesis_worker_memory_limit_gib",
        "routing_minimum_available_memory_fraction",
        "circuit_worker_memory_limit_gib",
        "exact_routing_flow_seed_max_terms",
        "exact_routing_fixed_cover_max_triangle_constraints",
        "exact_routing_fixed_cover_backend",
        "exact_routing_sat_solver",
        "exact_routing_sat_max_estimated_clauses",
        "beam",
    )
    snapshot = {key: config.get(key) for key in keys}
    partition_size = int(config["max_partition_size"])
    partition_size_end = partition_size
    if (
        config.get("strategy") not in ("bqskit", "qiskit")
        and config.get("auto_expand_partition_size", False)
        and (config.get("use_osr", False) or config.get("use_graph_search", False))
    ):
        partition_size_end = min(4, qubit_num)
    snapshot["partition_size_schedule"] = list(
        range(partition_size, partition_size_end + 1)
    )
    return snapshot


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=24.0,
        help="Wall-clock limit per circuit; use 0 to disable (default: 24).",
    )
    parser.add_argument(
        "--retry-timeouts",
        action="store_true",
        help="Retry circuits already recorded with status=timeout.",
    )
    args = parser.parse_args()
    if args.timeout_hours < 0:
        parser.error("--timeout-hours must be nonnegative")
    circuit_timeout = (
        None if args.timeout_hours == 0 else args.timeout_hours * 60.0 * 60.0
    )
    source_fingerprint = runtime_source_fingerprint()
    print(
        f"Benchmark source fingerprint: {source_fingerprint[:16]}",
        flush=True,
    )

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
        "routing-strategy": "exact-osr",  # all-partition OSR + exact SAT/Gurobi mapping flow; no post-routing synthesis
        # "tolerance": 1e-14,  # Squander Hilbert-Schmidt optimization target
        # "osr_optimization_tolerance": 1e-6,  # squared OSR tail cost; rank cutoff is its square root (1e-3)
        # "synthesis_acceptance_tolerance": 1e-10,  # Common block budget
        "use_float": True,  # whether to use single precision for the optimization (experimental, may cause instability in some cases, but can significantly reduce optimization time and memory usage for large circuits)
        # Minimize the complete exact-CNOT rank profile in one smooth OSR
        # optimization.  BFGS2's small basin sample is deterministic per seed
        # and avoids the unstable successive single-cut objective.
        "osr_profile_temperature": 0.1,
        # Use one smooth all-cuts objective for A2A, routing, and post-routing
        # synthesis; no stage-specific tuning is required.
        "osr_cut_smoothmax_temperature": 0.1,
        "optimizer": "BFGS2",
        "use_basin_hopping": True,
        "use_differential_evolution": False,
        "use_dual_annealing": False,
        "max_iteration_loops": 8,
        "max_inner_iterations_bfgs2": 1000,
    }
    result_directories, result_files = result_paths(
        config["max_partition_size"], config["strategy"]
    )

    # Inputs are curated in-repository, so no generated-file or reset filtering
    # is needed here. Each final circuit is written under the same filename in
    # the dataset's corresponding result directory.
    cached_stats = load_dataset_stats_cache()
    current_stats = {}
    cache_hits = 0
    cache_misses = 0
    files = []
    for dataset, input_directory in BENCHMARK_DATASETS.items():
        result_directory = result_directories[dataset]
        for filepath in input_directory.glob("*.qasm"):
            output_path = result_directory / filepath.name
            stats, cache_entry, cache_hit = cached_circuit_stats(
                filepath, cached_stats
            )
            relative_path = filepath.relative_to(REPOSITORY_ROOT).as_posix()
            current_stats[relative_path] = cache_entry
            cache_hits += int(cache_hit)
            cache_misses += int(not cache_hit)
            files.append(
                (stats, dataset, filepath, output_path)
            )
    files.sort(key=lambda item: item[0])

    if len(files) != 77:
        raise RuntimeError(
            f"Expected 77 archived benchmark circuits, found {len(files)}"
        )
    if cache_misses or current_stats != cached_stats:
        save_dataset_stats_cache(current_stats)
    print(
        f"Dataset statistics: {cache_hits} cached, {cache_misses} parsed; "
        f"cache={DATASET_STATS_CACHE.relative_to(REPOSITORY_ROOT)}"
    )

    print("=== Running circuits in order of (CNOT_count, qubit_count) ===")
    for (cnots, qubits), dataset, filepath, _ in files:
        print(
            f"  {cnots:>8} CNOT, {qubits:>4} qubits  "
            f"{dataset}/{filepath.name}"
        )
    print("=" * 60)


    # Each dataset directory is a self-contained archive: result circuits,
    # rewrite audits, and its own resumable results.json.
    results_by_dataset = {}
    loaded_count = 0
    for dataset, results_file in result_files.items():
        dataset_results = {}
        if results_file.exists():
            with results_file.open() as stream:
                dataset_results = json.load(stream)
            if not isinstance(dataset_results, dict):
                raise RuntimeError(f"{results_file} must contain a JSON object")
            for entry in dataset_results.values():
                if isinstance(entry, dict):
                    entry.pop("init_ibm_eagle", None)
                    entry.pop("final_ibm_eagle", None)
        results_by_dataset[dataset] = dataset_results
        loaded_count += len(dataset_results)
    if loaded_count:
        print(
            f"Loaded {loaded_count} existing results; "
            "will skip already-processed circuits."
        )

    def result_is_finished(dataset, fname, output_path):
        results = results_by_dataset[dataset]
        entry = results.get(fname)
        if not isinstance(entry, dict):
            return False
        _, audit_gzip_path = audit_paths(output_path)
        audit_entry = entry.get("rewrite_audit")
        has_verified_audit = (
            isinstance(audit_entry, dict)
            and audit_gzip_path.is_file()
            and audit_entry.get("file")
            and audit_entry.get("schema_version")
            == Wide_Circuit_Optimization.REWRITE_AUDIT_SCHEMA_VERSION
        )
        if (
            output_path.is_file()
            and has_verified_audit
            and entry.get("status", "completed") == "completed"
        ):
            return True
        return entry.get("status") == "timeout" and not args.retry_timeouts

    completed_count = sum(
        result_is_finished(dataset, filename.name, output_path)
        for _, dataset, filename, output_path in files
    )
    for _, dataset, filename, output_path in files:
        current_source_fingerprint = runtime_source_fingerprint()
        if current_source_fingerprint != source_fingerprint:
            raise RuntimeError(
                "Benchmark source changed while this driver was running. "
                "Restart it before processing another circuit: "
                f"started={source_fingerprint[:16]}, "
                f"current={current_source_fingerprint[:16]}."
            )
        fname = filename.name
        results = results_by_dataset[dataset]
        results_file = result_files[dataset]
        if result_is_finished(dataset, fname, output_path):
            print(
                f"Skipping already processed: {fname} "
                f"({results[fname].get('status', 'completed')})"
            )
            continue

        # Deleting one results.json entry is the deliberate invalidation API.
        # Do not let an output QASM, audit, or catalog from that old run leak
        # into the replacement run or block it on a source-QASM mismatch.
        if fname not in results:
            removed_artifacts = clear_invalidated_result_artifacts(output_path)
            if removed_artifacts:
                print(
                    f"Cleared {len(removed_artifacts)} invalidated result "
                    f"artifacts for {fname}."
                )

        print(f"executing optimization of circuit: {filename}")
        #if not filename.endswith("_n140.qasm"): continue

        context = mp.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=optimize_circuit_worker,
            args=(
                config,
                dataset,
                str(filename),
                str(output_path),
                source_fingerprint,
                result_queue,
            ),
        )
        process.start()
        try:
            process.join(circuit_timeout)
        except KeyboardInterrupt:
            print(f"\nInterrupted by user while processing {fname}; not recording a timeout.")
            terminate_circuit_process(process)
            raise

        if process.is_alive():
            terminate_circuit_process(process)
            output_path.with_suffix(output_path.suffix + ".tmp").unlink(
                missing_ok=True
            )
            timeout_circuit, _, _ = utils.qasm_to_squander_circuit(str(filename))
            timeout_config = dict(config)
            timeout_config["topology"] = (
                Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization.linear_topology(
                    timeout_circuit.get_Qbit_Num()
                )
            )
            resolved_timeout_config = (
                Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization(
                    timeout_config
                ).config
            )
            result_entry = {
                "file": fname,
                "dataset": dataset,
                "output_file": str(output_path.relative_to(REPOSITORY_ROOT)),
                "status": "timeout",
                "timeout_seconds": round(circuit_timeout, 2),
                "strategy": config["strategy"],
                "pre_opt_strategy": config["pre-opt-strategy"],
                "routing_strategy": config["routing-strategy"],
                "source_fingerprint": source_fingerprint,
                "configuration": result_configuration(
                    resolved_timeout_config, timeout_circuit.get_Qbit_Num()
                ),
                "init": CircuitGateStats(timeout_circuit),
                "timing": {
                    "a2a": None,
                    "routing": None,
                    "optimization": None,
                    "total": round(circuit_timeout, 2),
                },
            }
            results[fname] = result_entry
            save_results(results_file, results)
            completed_count += 1
            print(
                f"  timed out after {circuit_timeout / 3600.0:.2f} hours; "
                "recorded timeout and continuing"
            )
            print(f"--- {completed_count}/{len(files)} circuits processed ---")
            continue

        try:
            worker_result = result_queue.get(timeout=5.0)
        except queue.Empty:
            worker_result = {
                "ok": False,
                "traceback": (
                    f"Circuit worker exited with code {process.exitcode} "
                    "without returning a result."
                ),
                "exitcode": process.exitcode,
            }
        finally:
            result_queue.close()
            result_queue.join_thread()

        if not worker_result["ok"]:
            output_path.with_suffix(output_path.suffix + ".tmp").unlink(
                missing_ok=True
            )
            failed_circuit, _, _ = utils.qasm_to_squander_circuit(
                str(filename)
            )
            failed_config = dict(config)
            failed_config["topology"] = (
                Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization.linear_topology(
                    failed_circuit.get_Qbit_Num()
                )
            )
            resolved_failed_config = (
                Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization(
                    failed_config
                ).config
            )
            failure = worker_result["traceback"]
            results[fname] = {
                "file": fname,
                "dataset": dataset,
                "output_file": str(output_path.relative_to(REPOSITORY_ROOT)),
                "status": "failed",
                "strategy": config["strategy"],
                "pre_opt_strategy": config["pre-opt-strategy"],
                "routing_strategy": config["routing-strategy"],
                "source_fingerprint": source_fingerprint,
                "configuration": result_configuration(
                    resolved_failed_config,
                    failed_circuit.get_Qbit_Num(),
                ),
                "init": CircuitGateStats(failed_circuit),
                "worker_exitcode": worker_result.get("exitcode"),
                "failure": failure,
                "timing": {
                    "a2a": None,
                    "routing": None,
                    "optimization": None,
                    "total": None,
                },
            }
            save_results(results_file, results)
            completed_count += 1
            print(
                f"  FAILED {fname}; recorded diagnostic and continuing:\n"
                f"{failure}",
                flush=True,
            )
            print(f"--- {completed_count}/{len(files)} circuits attempted ---")
            continue

        result_entry = worker_result["entry"]
        results[fname] = result_entry
        save_results(results_file, results)
        completed_count += 1
        init_stats = result_entry["init"]
        opt_stats = result_entry["final"]
        elapsed = result_entry["timing"]["total"]
        print(f"  wrote verified result: {output_path}")
        print(f"  init: {init_stats['cnot_equiv']} CNOT, {init_stats['single_qubit']} 1q, "
              f"{init_stats['total_raw']} total, {init_stats['qubits']}q")
        print(f"  final: {opt_stats['cnot_equiv']} CNOT, {opt_stats['single_qubit']} 1q, "
              f"{opt_stats['total_raw']} total")
        print(f"  time: {elapsed:.2f}s")
        print(f"--- {completed_count}/{len(files)} circuits processed ---")

        print("--- %s seconds elapsed during optimization ---" % elapsed)
