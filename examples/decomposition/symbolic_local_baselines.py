#!/usr/bin/env python3
"""Normalize SDK one-qubit blocks in their native alphabet and replay them."""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"squander/decomposition"))


def normalize_baseline(row):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Optimize1qGatesDecomposition
    import numpy as np
    from symbolic_references import QISKIT_BASES
    if row["status"] != "verified_numeric":
        return copy.deepcopy(row)
    circuit = QuantumCircuit(row["num_qubits"], global_phase=row["global_phase"])
    for op in row["operations"]:
        getattr(circuit, op["gate"])(*op["parameters"], *op["wires"])
    allowed = QISKIT_BASES[row["basis"]]
    optimized = PassManager(Optimize1qGatesDecomposition(basis=list(allowed))).run(circuit)
    operations = [dict(gate=i.operation.name, parameters=[float(p) for p in i.operation.params],
        wires=[optimized.find_bit(q).index for q in i.qubits]) for i in optimized.data]
    if any(op["gate"] not in allowed for op in operations):
        raise ValueError("Local cleanup changed the native basis")
    count = sum(len(op["wires"]) == 2 for op in operations)
    if count != row["entanglers"]:
        raise ValueError("Local cleanup changed the entangler count")
    error = float(np.max(np.abs(Operator(optimized).data-Operator(circuit).data)))
    if not error < 2e-10:
        raise ValueError("Local cleanup failed phase-sensitive operator replay")
    target_error_bound = error + row["max_error"]
    if not target_error_bound < 2e-10:
        raise ValueError("Combined target replay error exceeds the comparison contract")
    # A cleanup can choose a poor native Euler convention. Keep the original
    # if it has fewer locals, while retaining the verified attempted result.
    before = sum(len(op["wires"]) == 1 for op in row["operations"])
    after = sum(len(op["wires"]) == 1 for op in operations)
    result = copy.deepcopy(row)
    if after <= before:
        result.update(operations=operations, global_phase=float(optimized.global_phase),
                      max_error=target_error_bound,
                      parameter_slots=sum(len(op["parameters"]) for op in operations))
    result["native_local_cleanup"] = dict(method="Qiskit Optimize1qGatesDecomposition",
        raw_local_count=before, attempted_local_count=after, retained_local_count=min(before, after),
        phase_sensitive_replay_error=error, entangler_count_unchanged=True,
        removes_adapter_identity_rotations=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.input, "rt") as stream:
        source = json.load(stream)
    report = copy.deepcopy(source)
    report["rows"] = [normalize_baseline(row) for row in source["rows"]]
    report["local_normalization_provenance"] = dict(
        input=str(args.input), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        scope="Numerical fixed bindings, native one-qubit cleanup only; not symbolic-family proofs")
    with gzip.open(args.output, "wt") as stream:
        json.dump(report, stream, separators=(",", ":"), allow_nan=False)
    changed = sum(row.get("native_local_cleanup", {}).get("retained_local_count", 0) <
                  row.get("native_local_cleanup", {}).get("raw_local_count", 0) for row in report["rows"])
    print(json.dumps(dict(rows=len(report["rows"]), cleaned_local_counts=changed)))


if __name__ == "__main__":
    main()
