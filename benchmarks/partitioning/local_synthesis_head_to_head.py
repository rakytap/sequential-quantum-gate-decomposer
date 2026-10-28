#!/usr/bin/env python3
"""Paired, independently certified 3q/4q OSR versus BQSKit synthesis.

Targets come from both methods' rewrite audits, OSR routing catalogs, and
seeded random circuits. A frozen manifest and append-only JSONL make runs
reproducible and resumable. Both solvers receive the same unitary and graph.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import subprocess
from threading import Lock
import time
import traceback

ROOT = Path(__file__).resolve().parent
DATASETS = ("IBMEagle", "QASMBenchmarks")
METHODS = ("osr", "bqskit")
# Attached BQSKit runtimes share fixed listener ports until their READY
# handshake. Serialize only that startup, never the optimization itself.
_BQSKIT_STARTUP_LOCK = Lock()


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def complete_graph(width):
    return [[i, j] for i in range(width) for j in range(i + 1, width)]


def connected_graph(width, edges):
    graph = sorted({tuple(sorted(map(int, edge))) for edge in edges})
    if not graph or any(a < 0 or b >= width or a == b for a, b in graph):
        return None
    reached = {0}
    while True:
        new = reached | {b for a, b in graph if a in reached} | {
            a for a, b in graph if b in reached
        }
        if new == reached:
            break
        reached = new
    return [list(edge) for edge in graph] if len(reached) == width else None


def induced_graph(qubits, edges):
    positions = {int(q): i for i, q in enumerate(qubits)}
    return connected_graph(
        len(qubits),
        [(positions[int(a)], positions[int(b)]) for a, b in edges
         if int(a) in positions and int(b) in positions],
    )


def load_archive(path, max_bytes, exclusions):
    if path.stat().st_size > max_bytes:
        exclusions.append(f"oversize:{path}")
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError, MemoryError) as exc:
        exclusions.append(f"unreadable:{path}:{type(exc).__name__}")
        return None


def sampled_file_candidates(candidates, seed, per_file):
    return sorted(
        candidates,
        key=lambda c: digest(f"{seed}|{c['source_file']}|{c['source_index']}"),
    )[:per_file]


def audit_event_candidate(
    event, *, dataset, origin, source_circuit, source_file, source_index,
    full_graph, exclusions=None,
):
    """Import a local target, including failed Native synthesis attempts."""
    kind = event.get("kind")
    stage = event.get("stage")
    if kind not in ("rewrite", "synthesis_attempt") or stage not in (
            "all_to_all", "topology_optimization"):
        return None
    before = event.get("before" if kind == "rewrite" else "target") or {}
    qubits = event.get("qubits") or event.get("location") or []
    try:
        width = int(before.get("qubits") or len(qubits))
    except (TypeError, ValueError):
        return None
    if width not in (3, 4):
        return None

    def reject(reason):
        if exclusions is not None:
            exclusions.append(f"{reason}:{source_file}:{source_index}")
        return None

    qasm = before.get("qasm")
    target = {}
    if isinstance(qasm, str):
        target["qasm"] = qasm
    elif before.get("format") == "unitary" and isinstance(before.get("unitary"), dict):
        # The Native audit writes row-major matrices in the same little-endian
        # basis as Qiskit's Operator. Do not guess an unknown producer's basis.
        if origin != "TreeSearch" and not str(event.get("component", "")).startswith(
                "squander"):
            return reject("unknown-unitary-basis")
        target.update(target_unitary=before["unitary"], target_basis="little_endian")
    else:
        return reject("missing-target-representation")

    if stage == "all_to_all":
        graph = complete_graph(width)
    elif event.get("subtopology") is not None:
        graph = connected_graph(width, event["subtopology"])
    elif len(qubits) == width:
        graph = induced_graph(qubits, full_graph)
    else:
        return reject("missing-local-topology")
    if graph is None:
        return reject("disconnected-local-topology")

    input_cnot = (before.get("gate_counts") or {}).get("CNOT")
    if input_cnot is None and isinstance(qasm, str):
        input_cnot = qasm.lower().count("\ncx ")
    after_counts = (event.get("after") or {}).get("gate_counts") or {}
    known_upper = (after_counts.get("CNOT")
                   if event.get("within_tolerance") is True else None)
    return {
        "dataset": dataset, "source_circuit": source_circuit,
        "source_file": source_file, "source_index": source_index,
        "source_event_kind": kind, "origin": origin,
        "context": "a2a" if stage == "all_to_all" else "topology",
        "width": width, "graph": graph, **target,
        # A matrix does not specify an input circuit. A feasible candidate's
        # gate count is a separate upper bound, not an invented source count.
        "input_cnot": None if input_cnot is None else int(input_cnot),
        "known_feasible_cnot_upper_bound": (
            None if known_upper is None else int(known_upper)),
    }


def archived_candidates(root, seed, per_file, max_archive_mb):
    candidates, exclusions = [], []
    max_bytes = int(max_archive_mb * 1024 * 1024)
    for dataset in DATASETS:
        for partition_width in (3, 4):
            for origin in ("TreeSearch", "bqskit"):
                directory = root / f"{dataset}_results_{partition_width}qbit_{origin}"
                if not directory.is_dir():
                    continue
                for path in sorted(directory.glob("*.audit.json.gz")):
                    archive = load_archive(path, max_bytes, exclusions)
                    if archive is None:
                        continue
                    events = archive.get("events", [])
                    route = next(
                        (e for e in events if e.get("kind") == "exact_osr_routing"),
                        None,
                    )
                    input_width = archive.get("run", {}).get(
                        "input_circuit", {}).get("qubits", 0)
                    if not isinstance(input_width, int):
                        input_width = 0
                    full_graph = (route.get("topology") if route else None) or [
                        [i, i + 1] for i in range(max(input_width - 1, 0))
                    ]
                    found = defaultdict(list)
                    for index, event in enumerate(events):
                        candidate = audit_event_candidate(
                            event, dataset=dataset, origin=origin,
                            source_circuit=path.name.removesuffix(".audit.json.gz"),
                            source_file=str(path.relative_to(root)), source_index=index,
                            full_graph=full_graph, exclusions=exclusions,
                        )
                        if candidate is not None:
                            found[(candidate["width"], candidate["context"])].append(candidate)
                    for group in found.values():
                        candidates.extend(sampled_file_candidates(
                            group, seed, per_file))

            # Production routing catalogs contain 3q targets even in 4q runs.
            # Genuine topology-aware 4q targets are in the post-routing audits.
            if partition_width != 3:
                continue
            directory = root / f"{dataset}_results_3qbit_TreeSearch"
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.routing-catalog.json.gz")):
                archive = load_archive(path, max_bytes, exclusions)
                if archive is None:
                    continue
                found = []
                for index, payload in enumerate(archive.get("payloads", [])):
                    qasm = payload.get("source_qasm")
                    if not isinstance(qasm, str):
                        continue
                    width = len(payload.get("input_assignment", []))
                    if width != 3:
                        continue
                    graph = connected_graph(width, payload.get("topology", []))
                    if graph is None:
                        continue
                    found.append({
                        "dataset": dataset,
                        "source_circuit": path.name.removesuffix(
                            ".routing-catalog.json.gz"),
                        "source_file": str(path.relative_to(root)),
                        "source_index": index,
                        "origin": "TreeSearch_catalog",
                        "context": "routing_catalog",
                        "width": width,
                        "graph": graph,
                        "qasm": qasm,
                        "input_cnot": qasm.lower().count("\ncx "),
                    })
                candidates.extend(sampled_file_candidates(found, seed, per_file))
    return candidates, exclusions


def generated_candidates(seed, per_bin):
    import numpy as np
    from qiskit import QuantumCircuit, qasm2

    rng = np.random.default_rng(seed)
    result = []
    for width in (3, 4):
        graphs = {
            "a2a": complete_graph(width),
            "path": [[i, i + 1] for i in range(width - 1)],
        }
        if width == 4:
            graphs["star"] = [[0, i] for i in range(1, width)]
            graphs["cycle"] = [[0, 1], [1, 2], [2, 3], [0, 3]]
        for name, graph in graphs.items():
            for cnots in (2, 4, 7, 10):
                for index in range(per_bin):
                    circuit = QuantumCircuit(width)
                    for q in range(width):
                        circuit.u(*rng.uniform(-math.pi, math.pi, 3), q)
                    for _ in range(cnots):
                        edge = graph[int(rng.integers(len(graph)))]
                        control, target = (
                            edge if int(rng.integers(2)) else edge[::-1]
                        )
                        circuit.cx(control, target)
                        for q in edge:
                            circuit.u(*rng.uniform(-math.pi, math.pi, 3), q)
                    result.append({
                        "dataset": "random",
                        "source_circuit": f"{width}q_{name}_{cnots}_{index}",
                        "source_file": "generated",
                        "source_index": index,
                        "origin": "random",
                        "context": f"random_{name}",
                        "width": width,
                        "graph": graph,
                        "qasm": qasm2.dumps(circuit),
                        "input_cnot": cnots,
                    })
    return result


def matrix_from_qasm(qasm):
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    return np.asarray(
        Operator(QuantumCircuit.from_qasm_str(qasm)).data,
        dtype=np.complex128,
    )


def matrix_from_case(case):
    """Decode the shared target without changing its unitary or basis."""
    import numpy as np

    encoded = case.get("target_unitary")
    if encoded is None:
        matrix = matrix_from_qasm(case["qasm"])
    else:
        if case.get("target_basis", "little_endian") != "little_endian":
            raise ValueError("Unknown serialized target basis")
        dimension = int(encoded["dimension"])
        real = np.asarray(encoded["real"], dtype=np.float64)
        imag = np.asarray(encoded["imag"], dtype=np.float64)
        if real.size != dimension * dimension or imag.size != real.size:
            raise ValueError("Malformed serialized target matrix")
        matrix = (real + 1j * imag).reshape(dimension, dimension)
    dimension = 1 << int(case["width"])
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise ValueError("Target matrix has an invalid width or nonfinite entries")
    if np.linalg.norm(
            matrix.conj().T @ matrix - np.eye(dimension), ord="fro") > 1e-8:
        raise ValueError("Target matrix is not unitary; projection is not allowed")
    return matrix


def unitary_digest(matrix):
    import numpy as np

    nonzero = matrix.flat[np.flatnonzero(abs(matrix) > 1e-12)[0]]
    normalized = matrix * (np.conj(nonzero) / abs(nonzero))
    packed = np.stack((normalized.real, normalized.imag), axis=-1)
    return hashlib.sha256(np.round(packed, 12).tobytes()).hexdigest()


def choose_cases(candidates, seed, max_cases, per_circuit, exclusions=None):
    groups = defaultdict(list)
    for candidate in candidates:
        cnots = candidate.get("input_cnot")
        complexity = ("unknown" if cnots is None else
                      "0-2" if cnots <= 2 else "3-5" if cnots <= 5 else
                      "6-9" if cnots <= 9 else "10+")
        group = (candidate["width"], candidate["context"],
                 candidate["dataset"], candidate["origin"], complexity)
        groups[group].append(candidate)
    for group in groups.values():
        group.sort(key=lambda c: digest(
            f"{seed}|{c['source_file']}|{c['source_index']}"))
    keys = sorted(groups, key=lambda key: digest(f"{seed}|{key}"))
    offsets, circuit_counts = defaultdict(int), Counter()
    seen, selected = set(), []
    while len(selected) < max_cases:
        advanced = False
        for key in keys:
            group = groups[key]
            if offsets[key] >= len(group):
                continue
            advanced = True
            case = dict(group[offsets[key]])
            offsets[key] += 1
            source = (case["dataset"], case["source_circuit"])
            if circuit_counts[source] >= per_circuit:
                continue
            try:
                target_hash = unitary_digest(matrix_from_case(case))
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                if exclusions is not None:
                    exclusions.append(
                        f"invalid-target:{case['source_file']}:{case['source_index']}:"
                        f"{type(exc).__name__}:{exc}")
                continue
            identity = (target_hash, tuple(map(tuple, case["graph"])))
            if identity in seen:
                continue
            seen.add(identity)
            circuit_counts[source] += 1
            case["target_sha256"] = target_hash
            case["case_id"] = digest(
                f"{case['width']}|{target_hash}|{case['graph']}")[:24]
            selected.append(case)
            if len(selected) >= max_cases:
                break
        if not advanced:
            break
    return selected


def process_infidelity(target, candidate):
    import numpy as np

    if (target.shape != candidate.shape or not np.isfinite(target).all()
            or not np.isfinite(candidate).all()):
        return math.inf
    overlap = np.trace(target.conj().T @ candidate) / target.shape[0]
    return max(0.0, 1.0 - float(abs(overlap) ** 2))


def reverse_qubit_order(matrix, width):
    """Convert between BQSKit's MSB-first and Qiskit's LSB-first matrices."""
    import numpy as np

    indices = [int(f"{index:0{width}b}"[::-1], 2)
               for index in range(1 << width)]
    return np.asarray(matrix)[np.ix_(indices, indices)]


def effective_coupling_graph(case):
    """Resolve legacy unrestricted A2A graphs without weakening topologies.

    Older frozen manifests use an empty (or null) graph for all-to-all
    synthesis. Resolve that shorthand at execution, not by rewriting the
    archived target, graph or case ID. An explicitly empty topology still
    permits no two-qubit interactions.
    """
    graph = case.get("graph")
    if not graph and case.get("context") == "a2a":
        return complete_graph(case["width"])
    if graph is None:
        raise ValueError("A topology case must specify its coupling graph")
    return [list(edge) for edge in graph]


def certify_output(case, target, native_matrix, output_qasm, claimed_cnots,
                   tolerance):
    """Check the emitted gate sequence with an independent Qiskit simulator."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    circuit = QuantumCircuit.from_qasm_str(output_qasm)
    if circuit.num_qubits != case["width"] or circuit.num_clbits:
        raise ValueError("exported circuit has the wrong quantum/classical width")
    legal_edges = {tuple(sorted(edge)) for edge in effective_coupling_graph(case)}
    cnots = 0
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        if operation.name == "cx" and len(qubits) == 2:
            if tuple(sorted(qubits)) not in legal_edges:
                raise ValueError(f"off-topology CNOT on {qubits}")
            cnots += 1
        elif len(qubits) != 1 or instruction.clbits:
            raise ValueError(f"unsupported non-CNOT operation: {operation.name}")
    if cnots != claimed_cnots:
        raise ValueError(f"native CNOT count {claimed_cnots} != exported {cnots}")
    exported_matrix = np.asarray(Operator(circuit).data, dtype=np.complex128)
    target_error = process_infidelity(target, exported_matrix)
    native_error = process_infidelity(native_matrix, exported_matrix)
    if target_error > tolerance:
        raise ValueError(f"exported circuit differs from target: {target_error}")
    if native_error > tolerance:
        raise ValueError(f"QASM/native matrix disagreement: {native_error}")
    return {
        "status": "verified", "cnots": cnots,
        "process_infidelity": target_error,
        "native_export_process_infidelity": native_error,
        "topology_verified": True, "output_qasm": output_qasm,
        "output_qasm_sha256": digest(output_qasm),
    }


def solve_worker(case, method, tolerance, seed, out_queue, osr_config=None,
                 cpu_id=None, compiler_ready=None):
    try:
        # Descendants inherit this private group, so timeout cleanup cannot
        # leave compiler workers running on a CPU reassigned to another solve.
        if hasattr(os, "setsid"):
            os.setsid()
        if cpu_id is not None:
            os.sched_setaffinity(0, {cpu_id})
        import numpy as np

        np.random.seed(seed)
        target = matrix_from_case(case)
        graph = [tuple(edge) for edge in effective_coupling_graph(case)]
        if method == "osr":
            from squander.decomposition.qgd_Wide_Circuit_Optimization import (
                qgd_Wide_Circuit_Optimization as WCO,
            )

            config = {
                "strategy": "TreeSearch", "topology": None,
                "use_osr": True, "use_graph_search": True,
                "max_partition_size": case["width"],
                "synthesis_acceptance_tolerance": tolerance,
                "verbosity": 0, "parallel": 0,
            }
            config.update(osr_config or {})
            config["random_seed"] = seed
            if (config.get("strategy") != "TreeSearch"
                    or config.get("use_osr") is not True
                    or config.get("use_graph_search") is not True):
                raise ValueError("This benchmark requires OSR graph synthesis")
            optimizer = WCO(config)
            started = time.monotonic()
            solutions = optimizer.DecomposePartition(
                target, optimizer.config, mini_topology=graph)
            seconds = time.monotonic() - started
            valid = []
            for circuit, params in solutions:
                matrix = circuit.get_Matrix(np.asarray(params, dtype=np.float64))
                error = process_infidelity(target, matrix)
                if error <= tolerance:
                    valid.append((int(circuit.get_Gate_Nums().get("CNOT", 0)),
                                  error, circuit, params, matrix))
            if valid:
                from qiskit import qasm2
                from squander.IO_interfaces import Qiskit_IO

                result = None
                failures = []
                for cnots, _, circuit, params, matrix in sorted(
                        valid, key=lambda entry: entry[:2]):
                    try:
                        output_qasm = qasm2.dumps(
                            Qiskit_IO.get_Qiskit_Circuit(circuit, params))
                        result = certify_output(
                            case, target, matrix, output_qasm, cnots, tolerance)
                        break
                    except Exception as exc:
                        failures.append(f"{type(exc).__name__}: {exc}")
                if result is None:
                    result = {"status": "invalid", "error": failures[:3]}
                result["seconds"] = seconds
            else:
                result = {"status": "no_solution", "seconds": seconds}
            result["effective_config"] = optimizer.config
        else:
            from bqskit import MachineModel, compile as bq_compile
            from bqskit.compiler import Compiler
            from bqskit.ir.gates import CNOTGate, U3Gate
            from bqskit.ir.lang.qasm2 import OPENQASM2Language
            from bqskit.qis.unitary import UnitaryMatrix

            model = MachineModel(
                case["width"], coupling_graph=graph,
                gate_set={CNOTGate(), U3Gate()},
            )
            epsilon = tolerance / (1.0 + math.sqrt(1.0 - tolerance))
            with Compiler(num_workers=1, num_blas_threads=1) as compiler:
                # READY means both fixed listeners have closed and the
                # attached compiler owns its server and worker connection.
                if compiler_ready is not None:
                    compiler_ready.set()
                started = time.monotonic()
                circuit = bq_compile(
                    UnitaryMatrix(reverse_qubit_order(target, case["width"])),
                    model=model, compiler=compiler,
                    optimization_level=1, max_synthesis_size=case["width"],
                    synthesis_epsilon=epsilon, seed=seed,
                )
                seconds = time.monotonic() - started
            native_matrix = reverse_qubit_order(
                np.asarray(circuit.get_unitary(), dtype=np.complex128),
                case["width"],
            )
            unfolded = circuit.copy()
            unfolded.unfold_all()
            output_qasm = OPENQASM2Language().encode(unfolded)
            try:
                result = certify_output(
                    case, target, native_matrix, output_qasm,
                    int(unfolded.count(CNOTGate())), tolerance)
            except (ValueError, AssertionError) as exc:
                result = {"status": "invalid", "error": str(exc)}
            result["seconds"] = seconds
        out_queue.put(result)
    except BaseException:
        out_queue.put({"status": "failed", "error": traceback.format_exc()[-2500:]})


def kill_process_tree(process):
    import psutil

    # Never signal a group inherited from the benchmark controller. Spawned
    # solve_worker establishes its own session before starting descendants.
    if hasattr(os, "killpg"):
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
                process.join(timeout=2)
                return
        except (ProcessLookupError, PermissionError):
            pass
    try:
        parent = psutil.Process(process.pid)
        for child in reversed(parent.children(recursive=True)):
            try:
                child.kill()
            except psutil.Error:
                pass
        parent.kill()
    except psutil.Error:
        pass
    process.join(timeout=2)


def run_one(case, method, args, cpu_id=None):
    import psutil

    context = mp.get_context("spawn")
    out_queue = context.Queue(maxsize=1)
    compiler_ready = context.Event() if method == "bqskit" else None
    worker_args = (case, method, args.tolerance, args.seed, out_queue,
                   getattr(args, "osr_config_values", None), cpu_id)
    if compiler_ready is not None:
        worker_args += (compiler_ready,)
    process = context.Process(target=solve_worker, args=worker_args)
    startup_locked = False
    if compiler_ready is not None:
        # Waiting for scheduling is outside a solve's resource budget, just
        # like waiting for an available CPU. Compiler initialization remains
        # inside the measured deadline and wall time.
        _BQSKIT_STARTUP_LOCK.acquire()
        startup_locked = True
    started = time.monotonic()
    peak_mb, forced, result = 0.0, None, None
    try:
        process.start()
        while process.is_alive():
            if startup_locked and compiler_ready.is_set():
                _BQSKIT_STARTUP_LOCK.release()
                startup_locked = False
            # Drain before joining: a large circuit can fill Queue's pipe,
            # preventing the producer's feeder thread (and worker) from exiting.
            try:
                result = out_queue.get(timeout=0.1)
                break
            except queue.Empty:
                pass
            try:
                parent = psutil.Process(process.pid)
                rss = sum(p.memory_info().rss for p in
                          [parent, *parent.children(recursive=True)])
                peak_mb = max(peak_mb, rss / (1024 * 1024))
            except psutil.Error:
                pass
            if peak_mb > args.memory_mb:
                forced = "memory_limit"
                break
            if time.monotonic() - started > args.timeout_seconds:
                forced = "timeout"
                break
        if forced:
            kill_process_tree(process)
            result = {"status": forced}
        elif result is None:
            try:
                result = out_queue.get(timeout=2)
            except queue.Empty:
                result = {"status": "failed", "error":
                          f"worker exited {process.exitcode} without a result"}
    except KeyboardInterrupt:
        kill_process_tree(process)
        raise
    finally:
        try:
            if process.pid is not None:
                process.join(timeout=2)
                if process.is_alive():
                    kill_process_tree(process)
            out_queue.close()
            out_queue.join_thread()
        finally:
            if startup_locked:
                _BQSKIT_STARTUP_LOCK.release()
    result.update({
        "wall_seconds": time.monotonic() - started,
        "peak_rss_mb": round(peak_mb, 1),
        "bqskit_startup_serialized_until_ready": compiler_ready is not None,
        "limits": {"timeout_seconds": args.timeout_seconds,
                   "memory_mb": args.memory_mb, "threads": args.threads,
                   "cpu_id": cpu_id},
    })
    return result


def _clustered_partition_mean_interval(clusters, seed):
    """Bootstrap source circuits, retaining the partition-weighted estimand."""
    import numpy as np

    values = list(clusters.values())
    if len(values) < 2:
        return None
    counts = np.asarray([len(value) for value in values], dtype=np.int64)
    totals = np.asarray([sum(value) for value in values], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(2000, len(values)))
    # Sampling an entire circuit preserves dependence among its partitions.
    # Weight each resampled partition equally, just as in the displayed mean.
    bootstrap = (
        totals[indices].sum(axis=1) / counts[indices].sum(axis=1)
    )
    return [float(value) for value in np.percentile(bootstrap, [2.5, 97.5])]


def summarize(rows, cases, seed, tolerance):
    import numpy as np
    from scipy.stats import binomtest

    outcomes = {(r["case_id"], r["method"]): r for r in rows}
    groups = defaultdict(list)
    for case in cases:
        groups[f"{case['width']}q/{case['context']}"].append(case)
    summary = {}
    for label, group in sorted(groups.items()):
        statuses = {method: Counter() for method in METHODS}
        times = {method: [] for method in METHODS}
        pairs, pair_errors, hybrid_covered = [], [], 0
        osr_only, bqskit_only, osr_timeout_losses = 0, 0, 0
        paired_attempted = 0
        unpaired_verified = {method: 0 for method in METHODS}
        for case in group:
            results = {method: outcomes.get((case["case_id"], method), {})
                       for method in METHODS}
            attempted = {
                method: result.get("status", "not_run") != "not_run"
                for method, result in results.items()
            }
            paired_attempted += all(attempted.values())
            for method, result in results.items():
                status = result.get("status", "not_run")
                statuses[method][status] += 1
                if status == "verified":
                    times[method].append(result["seconds"])
            valid = [r for r in results.values()
                     if r.get("status") == "verified"]
            hybrid_covered += bool(valid)
            if results["osr"].get("status") == "verified":
                if not attempted["bqskit"]:
                    unpaired_verified["osr"] += 1
                else:
                    osr_only += results["bqskit"].get("status") != "verified"
            if results["bqskit"].get("status") == "verified":
                if not attempted["osr"]:
                    unpaired_verified["bqskit"] += 1
                else:
                    bqskit_only += results["osr"].get("status") != "verified"
                osr_timeout_losses += results["osr"].get("status") == "timeout"
            if len(valid) == 2:
                for result in valid:
                    if (result.get("topology_verified") is not True
                            or not result.get("output_qasm")):
                        raise RuntimeError(
                            "A verified row lacks its circuit certificate; "
                            "rerun the frozen manifest with this harness")
                pair_error = process_infidelity(
                    matrix_from_qasm(results["osr"]["output_qasm"]),
                    matrix_from_qasm(results["bqskit"]["output_qasm"]),
                )
                if pair_error > 4.1 * tolerance:
                    raise RuntimeError(
                        f"Paired circuits disagree for {case['case_id']}: "
                        f"{pair_error}")
                pair_errors.append(pair_error)
                pairs.append((case, results["osr"], results["bqskit"]))
        deltas = [o["cnots"] - b["cnots"] for _, o, b in pairs]
        wins = sum(d < 0 for d in deltas)
        losses = sum(d > 0 for d in deltas)
        ties = sum(d == 0 for d in deltas)
        test = binomtest(wins, wins + losses) if wins + losses else None
        clusters = defaultdict(list)
        for (case, _, _), delta in zip(pairs, deltas):
            clusters[f"{case['dataset']}/{case['source_circuit']}"].append(delta)
        interval = _clustered_partition_mean_interval(clusters, seed)
        paired_times = {
            method: [pair[index]["seconds"] for pair in pairs]
            for method, index in (("osr", 1), ("bqskit", 2))
        }
        wall_pairs = [
            (osr["wall_seconds"], bqskit["wall_seconds"])
            for _, osr, bqskit in pairs
            if all(
                method.get("wall_seconds") is not None
                and math.isfinite(method["wall_seconds"])
                and method["wall_seconds"] >= 0
                for method in (osr, bqskit)
            )
        ]
        solver_ratios = [
            bqskit["seconds"] / osr["seconds"]
            for _, osr, bqskit in pairs
            if all(
                math.isfinite(method["seconds"]) and method["seconds"] > 0
                for method in (osr, bqskit)
            )
        ]
        wall_ratios = [
            bqskit / osr for osr, bqskit in wall_pairs
            if osr > 0 and bqskit > 0
        ]
        summary[label] = {
            "cases": len(group),
            "paired_attempted": paired_attempted,
            "paired_verified": len(pairs),
            "osr_wins": wins, "bqskit_wins": losses, "ties": ties,
            "osr_only_verified": osr_only,
            "bqskit_only_verified": bqskit_only,
            "osr_timeouts_against_verified_bqskit": osr_timeout_losses,
            "completion_aware_osr_wins": wins + osr_only,
            "completion_aware_osr_losses": losses + bqskit_only,
            "unpaired_verified": unpaired_verified,
            "unrun_counts": {
                method: statuses[method].get("not_run", 0)
                for method in METHODS
            },
            "status_counts": {m: dict(statuses[m]) for m in METHODS},
            "mean_cnot_delta_osr_minus_bqskit":
                float(np.mean(deltas)) if deltas else None,
            "median_cnot_delta_osr_minus_bqskit":
                float(np.median(deltas)) if deltas else None,
            "circuit_cluster_mean_delta_95pct_ci": interval,
            "two_sided_sign_test_p": test.pvalue if test else None,
            "median_seconds_on_verified": {
                m: float(np.median(times[m])) if times[m] else None
                for m in METHODS
            },
            "cnot_delta_ci_estimand": "partition_weighted_mean",
            "median_seconds_on_paired_verified": {
                method: float(np.median(values)) if values else None
                for method, values in paired_times.items()
            },
            "paired_wall_timing_count": len(wall_pairs),
            "median_wall_seconds_on_paired_verified": {
                method: (
                    float(np.median([pair[index] for pair in wall_pairs]))
                    if wall_pairs else None
                )
                for method, index in (("osr", 0), ("bqskit", 1))
            },
            "paired_solver_speedup_count": len(solver_ratios),
            "median_paired_solver_speedup_bqskit_over_osr": (
                float(np.median(solver_ratios)) if solver_ratios else None
            ),
            "paired_wall_speedup_count": len(wall_ratios),
            "median_paired_wall_speedup_bqskit_over_osr": (
                float(np.median(wall_ratios)) if wall_ratios else None
            ),
            "hybrid_oracle_verified_cases": hybrid_covered,
            "max_pairwise_process_infidelity": (
                max(pair_errors) if pair_errors else None),
        }
    return summary


def regression_audit(rows, cases):
    outcomes = {(row["case_id"], row["method"]): row for row in rows}
    audit = []
    for case in cases:
        osr = outcomes.get((case["case_id"], "osr"), {})
        bqskit = outcomes.get((case["case_id"], "bqskit"), {})
        osr_status = osr.get("status", "not_run")
        bqskit_status = bqskit.get("status", "not_run")
        missing_methods = [
            method for method, status in (
                ("osr", osr_status), ("bqskit", bqskit_status)
            ) if status == "not_run"
        ]
        if missing_methods:
            kind = "missing_comparison_result"
        elif osr_status == bqskit_status == "verified":
            if osr["cnots"] <= bqskit["cnots"]:
                continue
            kind = "osr_cnot_loss"
        elif bqskit_status == "verified":
            kind = "osr_timeout_loss" if osr_status == "timeout" else "osr_failure_loss"
        elif osr_status == "verified":
            kind = "bqskit_failure_osr_verified"
        else:
            kind = "neither_verified"
        audit.append({
            "kind": kind, "case_id": case["case_id"],
            "width": case["width"], "context": case["context"],
            "source_circuit": case["source_circuit"],
            "target_sha256": case["target_sha256"], "graph": case["graph"],
            "osr_status": osr_status, "bqskit_status": bqskit_status,
            "missing_methods": missing_methods,
            "osr_cnots": osr.get("cnots"), "bqskit_cnots": bqskit.get("cnots"),
            "osr_wall_seconds": osr.get("wall_seconds"),
            "bqskit_wall_seconds": bqskit.get("wall_seconds"),
        })
    return audit


def _tex_number(value, significant_digits=3):
    """Render numeric values without treating C-style exponents as algebra."""
    literal = f"{value:.{significant_digits}g}"
    if "e" not in literal:
        return literal
    coefficient, exponent = literal.split("e")
    power = f"10^{{{int(exponent)}}}"
    return power if coefficient == "1" else f"{coefficient}\\times {power}"


def _tex_seconds(value):
    return f"{value:.2f}" if value == 0 or value >= 0.01 else _tex_number(value)


def write_tex_table(summary, path, tolerance, audit=(), *,
                    table_label="tab:local-osr-bqskit"):
    lines = [
        "% Generated by local_synthesis_head_to_head.py; do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrccc}",
        "\\hline",
        "Context & Targets & Paired & OSR valid & BQSKit valid & "
        "OSR fail/BQ valid & OSR W/T/L & $\\Delta$CX (95\\% CI) & "
        "Paired time OSR/BQSKit (s) \\\\ ",
        "\\hline",
    ]
    for label, row in sorted(summary.items()):
        valid = [row["status_counts"][method].get("verified", 0)
                 for method in METHODS]
        interval = row["circuit_cluster_mean_delta_95pct_ci"]
        mean = row["mean_cnot_delta_osr_minus_bqskit"]
        delta = (f"${mean:+.2f}\\,[{interval[0]:+.2f},{interval[1]:+.2f}]$"
                 if mean is not None and interval is not None else "--")
        medians = row["median_seconds_on_paired_verified"]
        timing = (f"${_tex_seconds(medians['osr'])}/{_tex_seconds(medians['bqskit'])}$"
                  if all(medians[m] is not None for m in METHODS) else "--")
        wins = (f"{row['osr_wins']}/{row['ties']}/{row['bqskit_wins']}")
        context = label.replace("_", "\\_")
        lines.append(
            f"{context} & {row['cases']} & {row['paired_verified']} & "
            f"{valid[0]} & {valid[1]} & {row['bqskit_only_verified']} & "
            f"{wins} & {delta} & {timing} \\\\ "
        )
    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\caption{Paired local synthesis of 3- and 4-qubit targets. "
        "Each verified exported circuit reproduces the same target up to "
        f"global phase with process infidelity at most ${_tex_number(tolerance, 12)}$ "
        "and obeys the stated coupling graph. Wins/ties/losses and mean "
        "CNOT differences use only jointly verified pairs; confidence "
        "intervals use a source-circuit cluster bootstrap of the same "
        "partition-weighted mean. Completion counts are per method; median "
        "solver times use only jointly verified targets. "
        "OSR fail/BQ valid counts OSR timeouts or failures when BQSKit "
        "produces a verified circuit. Unrun methods are reported as missing "
        "comparisons, not synthesis failures or wins.}",
        "\\label{" + table_label + "}",
        "\\end{table*}",
        "",
    ]
    lines.append("% BEGIN LOCAL SYNTHESIS REGRESSION AUDIT")
    for entry in audit:
        lines.append(
            "% " + entry["kind"] + ": " + entry["case_id"] + " "
            + f"{entry['width']}q/{entry['context']} "
            + entry["source_circuit"] + " "
            + f"OSR={entry['osr_status']}:{entry['osr_cnots']} "
            + f"BQSKit={entry['bqskit_status']}:{entry['bqskit_cnots']}"
        )
    lines.append("% END LOCAL SYNTHESIS REGRESSION AUDIT")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        help="Replay an existing frozen target manifest")
    parser.add_argument("--case-id", action="append",
                        help="Replay selected case IDs; may be repeated")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--jobs", type=int, default=1,
                        help="Independent solves, each pinned to one Linux CPU")
    parser.add_argument("--osr-config", type=Path,
                        help="JSON object of OSR configuration overrides")
    parser.add_argument("--max-cases", type=int, default=1000)
    parser.add_argument("--per-file", type=int, default=32)
    parser.add_argument("--per-circuit", type=int, default=12)
    parser.add_argument("--random-per-bin", type=int, default=8)
    parser.add_argument("--max-archive-mb", type=float, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--memory-mb", type=float, default=4096)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if (args.max_cases < 1 or args.per_file < 1 or args.per_circuit < 1
            or args.random_per_bin < 0 or args.max_archive_mb <= 0
            or args.timeout_seconds <= 0 or args.memory_mb <= 0
            or args.threads < 1 or args.jobs < 1 or not 0 < args.tolerance < 1):
        parser.error("sample sizes and budgets must be positive; tolerance in (0,1)")
    return args


def main():
    args = parse_args()
    args.osr_config_values = (
        json.loads(args.osr_config.read_text()) if args.osr_config else {}
    )
    if not isinstance(args.osr_config_values, dict):
        raise ValueError("--osr-config must contain a JSON object")
    execution = {"timeout_seconds": args.timeout_seconds,
                 "memory_mb": args.memory_mb, "threads": args.threads,
                 "methods": args.methods, "osr_config": args.osr_config_values}
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    manifest_path = args.output.with_suffix(".manifest.json")
    if manifest_path.exists():
        with manifest_path.open() as stream:
            manifest = json.load(stream)
        cases = manifest["cases"]
        print(f"Resuming frozen manifest: {len(cases)} targets", flush=True)
    else:
        if args.output.exists():
            raise RuntimeError("Result JSONL exists without its frozen manifest")
        if args.manifest:
            source_text = args.manifest.read_text()
            manifest = json.loads(source_text)
            manifest["source_manifest_sha256"] = digest(source_text)
            cases = manifest["cases"]
        else:
            candidates, skipped = archived_candidates(
                args.root, args.seed, args.per_file, args.max_archive_mb)
            candidates += generated_candidates(args.seed, args.random_per_bin)
            cases = choose_cases(
                candidates, args.seed, args.max_cases, args.per_circuit,
                exclusions=skipped)
            manifest = {
                "schema": 1, "seed": args.seed, "tolerance": args.tolerance,
                "max_archive_mb": args.max_archive_mb,
                "skipped_archives": skipped,
            }
        if args.case_id:
            requested = set(args.case_id)
            unknown = requested - {case["case_id"] for case in cases}
            if unknown:
                raise ValueError(f"Unknown case IDs: {sorted(unknown)}")
            cases = [case for case in cases if case["case_id"] in requested]
        manifest["cases"] = cases
        manifest["execution"] = execution
        repo_root = ROOT.parents[1]
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        diff = subprocess.check_output(["git", "diff"], cwd=repo_root, text=True)
        manifest["code_provenance"] = {
            "git_revision": revision, "tracked_diff_sha256": digest(diff),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as stream:
            json.dump(manifest, stream, separators=(",", ":"))
        print(f"Froze {len(cases)} targets", flush=True)
    if manifest["tolerance"] != args.tolerance or manifest["seed"] != args.seed:
        raise RuntimeError("Resume must use manifest tolerance and seed")
    if (not args.summary_only and manifest.get("execution", execution) != execution):
        raise RuntimeError("Resume must use the frozen solver settings and resource limits")
    rows = []
    if args.output.exists():
        with args.output.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    completed = {(r["case_id"], r["method"]) for r in rows}
    if not args.summary_only:
        pending = [(index, case, method)
                   for index, case in enumerate(cases, 1)
                   for method in args.methods
                   if (case["case_id"], method) not in completed]
        cpus = queue.Queue()
        allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        if allowed and args.jobs > len(allowed):
            raise ValueError("--jobs exceeds the available CPU affinity mask")
        for cpu_id in (allowed[:args.jobs] if allowed else [None] * args.jobs):
            cpus.put(cpu_id)

        def run_task(task):
            index, case, method = task
            cpu_id = cpus.get()
            try:
                return index, case, method, run_one(case, method, args, cpu_id)
            finally:
                cpus.put(cpu_id)

        with args.output.open("a", buffering=1) as stream:
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                futures = [executor.submit(run_task, task) for task in pending]
                for future in as_completed(futures):
                    index, case, method, result = future.result()
                    row = {
                        "case_id": case["case_id"], "method": method,
                        "width": case["width"], "context": case["context"],
                        "dataset": case["dataset"],
                        "source_circuit": case["source_circuit"],
                        "input_cnot": case["input_cnot"], **result,
                    }
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                    rows.append(row)
                    completed.add((case["case_id"], method))
                    print(
                        f"{index}/{len(cases)} {case['case_id']} {method}: "
                        f"{result['status']} {result.get('cnots', '-')} CX; "
                        f"{result['wall_seconds']:.2f}s wall", flush=True,
                    )
    summary = summarize(rows, cases, args.seed, args.tolerance)
    audit = regression_audit(rows, cases)
    with args.output.with_suffix(".regressions.json").open("w") as stream:
        json.dump(audit, stream, indent=2)
    with args.output.with_suffix(".summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    tex_path = args.output.with_suffix(".tex")
    write_tex_table(summary, tex_path, args.tolerance, audit)
    print(f"Wrote LaTeX table: {tex_path}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
