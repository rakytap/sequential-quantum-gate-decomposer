#!/usr/bin/env python3
"""Paired, independently certified 3q/4q OSR versus BQSKit synthesis.

Targets come from both methods' rewrite audits, OSR routing catalogs, and
seeded random circuits. A frozen manifest and append-only JSONL make runs
reproducible and resumable. Both solvers receive the same unitary and graph.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback

ROOT = Path(__file__).resolve().parent
DATASETS = ("IBMEagle", "QASMBenchmarks")
METHODS = ("osr", "bqskit")


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
                        if event.get("kind") != "rewrite":
                            continue
                        stage = event.get("stage")
                        if stage not in ("all_to_all", "topology_optimization"):
                            continue
                        before = event.get("before") or {}
                        qubits = event.get("qubits") or event.get("location") or []
                        width = int(before.get("qubits") or len(qubits))
                        if width not in (3, 4):
                            continue
                        if len(qubits) != width:
                            continue
                        qasm = before.get("qasm")
                        if not isinstance(qasm, str):
                            continue
                        graph = (complete_graph(width) if stage == "all_to_all"
                                 else induced_graph(qubits, full_graph))
                        if graph is None:
                            continue
                        found[(width, stage)].append({
                            "dataset": dataset,
                            "source_circuit": path.name.removesuffix(".audit.json.gz"),
                            "source_file": str(path.relative_to(root)),
                            "source_index": index,
                            "origin": origin,
                            "context": "a2a" if stage == "all_to_all" else "topology",
                            "width": width,
                            "graph": graph,
                            "qasm": qasm,
                            "input_cnot": int((before.get("gate_counts") or {}).get(
                                "CNOT", qasm.lower().count("\ncx "))),
                        })
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


def unitary_digest(matrix):
    import numpy as np

    nonzero = matrix.flat[np.flatnonzero(abs(matrix) > 1e-12)[0]]
    normalized = matrix * (np.conj(nonzero) / abs(nonzero))
    packed = np.stack((normalized.real, normalized.imag), axis=-1)
    return hashlib.sha256(np.round(packed, 12).tobytes()).hexdigest()


def choose_cases(candidates, seed, max_cases, per_circuit):
    groups = defaultdict(list)
    for candidate in candidates:
        cnots = candidate["input_cnot"]
        complexity = ("0-2" if cnots <= 2 else "3-5" if cnots <= 5 else
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
                target_hash = unitary_digest(matrix_from_qasm(case["qasm"]))
            except (ValueError, IndexError):
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

    overlap = np.trace(target.conj().T @ candidate) / target.shape[0]
    return max(0.0, 1.0 - float(abs(overlap) ** 2))


def reverse_qubit_order(matrix, width):
    """Convert between BQSKit's MSB-first and Qiskit's LSB-first matrices."""
    import numpy as np

    indices = [int(f"{index:0{width}b}"[::-1], 2)
               for index in range(1 << width)]
    return np.asarray(matrix)[np.ix_(indices, indices)]


def certify_output(case, target, native_matrix, output_qasm, claimed_cnots,
                   tolerance):
    """Check the emitted gate sequence with an independent Qiskit simulator."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    circuit = QuantumCircuit.from_qasm_str(output_qasm)
    if circuit.num_qubits != case["width"] or circuit.num_clbits:
        raise ValueError("exported circuit has the wrong quantum/classical width")
    legal_edges = {tuple(sorted(edge)) for edge in case["graph"]}
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


def solve_worker(case, method, tolerance, seed, out_queue):
    try:
        import numpy as np

        target = matrix_from_qasm(case["qasm"])
        graph = [tuple(edge) for edge in case["graph"]]
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

    try:
        parent = psutil.Process(process.pid)
        for child in reversed(parent.children(recursive=True)):
            child.kill()
        parent.kill()
    except psutil.Error:
        pass
    process.join(timeout=2)


def run_one(case, method, args):
    import psutil

    context = mp.get_context("spawn")
    out_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=solve_worker,
        args=(case, method, args.tolerance, args.seed, out_queue),
    )
    started = time.monotonic()
    process.start()
    peak_mb, forced = 0.0, None
    try:
        while process.is_alive():
            process.join(timeout=0.1)
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
        else:
            try:
                result = out_queue.get(timeout=2)
            except queue.Empty:
                result = {"status": "failed", "error":
                          f"worker exited {process.exitcode} without a result"}
    except KeyboardInterrupt:
        kill_process_tree(process)
        raise
    finally:
        out_queue.close()
        out_queue.join_thread()
    result.update({
        "wall_seconds": time.monotonic() - started,
        "peak_rss_mb": round(peak_mb, 1),
    })
    return result


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
        for case in group:
            results = {method: outcomes.get((case["case_id"], method), {})
                       for method in METHODS}
            for method, result in results.items():
                status = result.get("status", "not_run")
                statuses[method][status] += 1
                if status == "verified":
                    times[method].append(result["seconds"])
            valid = [r for r in results.values()
                     if r.get("status") == "verified"]
            hybrid_covered += bool(valid)
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
        means = np.array([np.mean(v) for v in clusters.values()])
        interval = None
        if len(means) >= 2:
            rng = np.random.default_rng(seed)
            bootstrap = rng.choice(
                means, size=(2000, len(means)), replace=True).mean(axis=1)
            interval = [float(v) for v in np.percentile(bootstrap, [2.5, 97.5])]
        summary[label] = {
            "cases": len(group),
            "paired_verified": len(pairs),
            "osr_wins": wins, "bqskit_wins": losses, "ties": ties,
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
            "hybrid_oracle_verified_cases": hybrid_covered,
            "max_pairwise_process_infidelity": (
                max(pair_errors) if pair_errors else None),
        }
    return summary


def write_tex_table(summary, path, tolerance):
    lines = [
        "% Generated by local_synthesis_head_to_head.py; do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrccc}",
        "\\hline",
        "Context & Targets & Paired & OSR valid & BQSKit valid & "
        "OSR W/T/L & $\\Delta$CX (95\\% CI) & "
        "Median time OSR/BQSKit (s) \\\\ ",
        "\\hline",
    ]
    for label, row in sorted(summary.items()):
        valid = [row["status_counts"][method].get("verified", 0)
                 for method in METHODS]
        interval = row["circuit_cluster_mean_delta_95pct_ci"]
        mean = row["mean_cnot_delta_osr_minus_bqskit"]
        delta = (f"${mean:+.2f}\\,[{interval[0]:+.2f},{interval[1]:+.2f}]$"
                 if mean is not None and interval is not None else "--")
        medians = row["median_seconds_on_verified"]
        timing = (f"${medians['osr']:.2f}/{medians['bqskit']:.2f}$"
                  if all(medians[m] is not None for m in METHODS) else "--")
        wins = (f"{row['osr_wins']}/{row['ties']}/{row['bqskit_wins']}")
        context = label.replace("_", "\\_")
        lines.append(
            f"{context} & {row['cases']} & {row['paired_verified']} & "
            f"{valid[0]} & {valid[1]} & {wins} & {delta} & {timing} \\\\ "
        )
    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\caption{Paired local synthesis of 3- and 4-qubit targets. "
        "Each verified exported circuit reproduces the same target up to "
        f"global phase with process infidelity at most ${tolerance:.0e}$ "
        "and obeys the stated coupling graph. Wins/ties/losses and mean "
        "CNOT differences use only jointly verified pairs; confidence "
        "intervals bootstrap source circuits. Completion counts and median "
        "times are per method, with times conditioned on verification.}",
        "\\label{tab:local-osr-bqskit}",
        "\\end{table*}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
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
            or args.threads < 1 or not 0 < args.tolerance < 1):
        parser.error("sample sizes and budgets must be positive; tolerance in (0,1)")
    return args


def main():
    args = parse_args()
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
        candidates, skipped = archived_candidates(
            args.root, args.seed, args.per_file, args.max_archive_mb)
        candidates += generated_candidates(args.seed, args.random_per_bin)
        cases = choose_cases(
            candidates, args.seed, args.max_cases, args.per_circuit)
        manifest = {
            "schema": 1, "seed": args.seed, "tolerance": args.tolerance,
            "max_archive_mb": args.max_archive_mb,
            "skipped_archives": skipped, "cases": cases,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as stream:
            json.dump(manifest, stream, separators=(",", ":"))
        print(f"Froze {len(cases)} targets; skipped {len(skipped)} archives",
              flush=True)
    if manifest["tolerance"] != args.tolerance or manifest["seed"] != args.seed:
        raise RuntimeError("Resume must use manifest tolerance and seed")
    rows = []
    if args.output.exists():
        with args.output.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    completed = {(r["case_id"], r["method"]) for r in rows}
    if not args.summary_only:
        with args.output.open("a", buffering=1) as stream:
            for index, case in enumerate(cases, 1):
                for method in METHODS:
                    key = case["case_id"], method
                    if key in completed:
                        continue
                    result = run_one(case, method, args)
                    row = {
                        "case_id": case["case_id"], "method": method,
                        "width": case["width"], "context": case["context"],
                        "dataset": case["dataset"],
                        "source_circuit": case["source_circuit"],
                        "input_cnot": case["input_cnot"], **result,
                    }
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                    rows.append(row)
                    completed.add(key)
                    print(
                        f"{index}/{len(cases)} {case['case_id']} {method}: "
                        f"{result['status']} {result.get('cnots', '-')} CX; "
                        f"{result['wall_seconds']:.2f}s wall", flush=True,
                    )
    summary = summarize(rows, cases, args.seed, args.tolerance)
    with args.output.with_suffix(".summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    tex_path = args.output.with_suffix(".tex")
    write_tex_table(summary, tex_path, args.tolerance)
    print(f"Wrote LaTeX table: {tex_path}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
