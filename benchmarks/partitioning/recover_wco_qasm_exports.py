#!/usr/bin/env python3
"""Recover completed WCO synthesis rejected solely by lossy QASM export.

No synthesis is rerun and no verifier is weakened. The retained file must have
its recorded byte hash and exactly match Qiskit's legacy formatted parameter
stream. The audit's lossless output is staged and its complete certificate is
verified before any archive is changed. Original artifacts are backed up.
Original benchmark timings were not retained on failure and remain unknown.
"""

import argparse
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile


EXPORT_FAILURE = (
    "Archived output QASM is not the exact replayed gate and parameter stream."
)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def gate_stats(backend, representation):
    from qiskit import QuantumCircuit
    from squander import Qiskit_IO

    qasm = representation["qasm"] if isinstance(representation, dict) else representation
    circuit, _ = Qiskit_IO.convert_Qiskit_to_Squander(
        QuantumCircuit.from_qasm_str(qasm)
    )
    return backend.CircuitGateStats(circuit)


def recovered_entry(backend, entry, audit, audit_path, digest, verification, old_digest):
    result = dict(entry)
    result["status"] = "completed"
    result["final"] = gate_stats(backend, audit["run"]["output_circuit"])
    routing = [event for event in audit["events"]
               if event.get("kind") == "exact_osr_routing"]
    if routing:
        result["all_to_all"] = gate_stats(backend, routing[0]["input"])
        result["routed"] = gate_stats(backend, routing[-1]["output"])
    if audit["run"].get("routing_osr_catalog") is not None:
        result["routing_osr_catalog"] = audit["run"]["routing_osr_catalog"]
    output_relative = Path(audit["run"]["output_file"])
    result["rewrite_audit"] = {
        "schema_version": audit["schema_version"],
        "file": str(output_relative.with_name(audit_path.name)),
        "sha256": digest,
        "summary": audit["summary"],
        **verification,
    }
    result["export_recovery"] = {
        "reason": "Lossy Qiskit formatting replaced by the exact audited output.",
        "original_failure": result.pop("failure", None),
        "original_audit_sha256": old_digest,
        "original_output_file_sha256": entry.get("_old_output_sha256"),
        "synthesis_rerun": False,
        "original_timings_available": False,
    }
    result.pop("worker_exitcode", None)
    result.pop("_old_output_sha256", None)
    # Failure entries do not contain the completed optimizer's timing values.
    result["timing"] = {"a2a": None, "routing": None,
                        "optimization": None, "total": None}
    return result


def recover_one(results_path, name, *, apply=False):
    """Verify one saved export and optionally commit its guarded recovery."""
    from qiskit import QuantumCircuit, qasm2

    backend = importlib.import_module(
        "squander.decomposition.qgd_Wide_Circuit_Optimization"
    )
    results_path = Path(results_path).resolve()
    original_results = results_path.read_bytes()
    results = json.loads(original_results)
    entry = results[name]
    if entry.get("status") != "failed" or EXPORT_FAILURE not in entry.get("failure", ""):
        raise ValueError("Recovery requires the specific final-export failure.")
    if Path(name).name != name or not name.endswith(".qasm"):
        raise ValueError("Result keys must be local QASM basenames.")
    output_path = results_path.parent / name
    audit_path = output_path.with_suffix(".audit.json.gz")
    original_digest = backend._file_sha256_streaming(audit_path)
    with gzip.open(audit_path, "rt", encoding="utf-8") as stream:
        audit = json.load(stream)
    run = audit["run"]
    if (run.get("output_file") != entry.get("output_file")
            or Path(run["output_file"]).name != name):
        raise AssertionError("Audit and failed result identify different outputs.")
    original_output = output_path.read_bytes()
    original_output_digest = sha256_bytes(original_output)
    if original_output_digest != run.get("output_file_sha256"):
        raise AssertionError("Retained output does not have its recorded byte hash.")
    exact_representation = run["output_circuit"]
    exact_qasm = (exact_representation["qasm"]
                  if isinstance(exact_representation, dict) else exact_representation)
    expected = QuantumCircuit.from_qasm_str(exact_qasm)
    legacy_state = backend._qasm_exact_state(qasm2.dumps(expected))
    retained_state = backend._qasm_exact_state(original_output.decode("utf-8"))
    if retained_state != legacy_state:
        raise AssertionError(
            "Retained output is not the legacy formatter's exact gate stream."
        )
    repaired = dict(audit)
    repaired["run"] = dict(run)
    repaired_output = exact_qasm.encode("utf-8")
    repaired["run"]["output_file_sha256"] = sha256_bytes(repaired_output)
    with tempfile.TemporaryDirectory(
        prefix=".audit-export-recovery-", dir=results_path.parent
    ) as staging:
        staging = Path(staging)
        staged_output = staging / output_path.name
        staged_audit = staging / audit_path.name
        staged_results = staging / results_path.name
        staged_output.write_bytes(repaired_output)
        with gzip.open(staged_audit, "wt", encoding="utf-8") as stream:
            json.dump(repaired, stream, separators=(",", ":"), allow_nan=False)
        digest = backend._file_sha256_streaming(staged_audit)
        verification = backend.verify_rewrite_audit(
            repaired, expected_sha256=digest, audit_path=staged_audit
        )
        updated_entry = dict(entry)
        updated_entry["_old_output_sha256"] = original_output_digest
        updated_entry = recovered_entry(
            backend, updated_entry, repaired, audit_path, digest,
            verification, original_digest,
        )
        results[name] = updated_entry
        staged_results.write_text(json.dumps(results, indent=2), encoding="utf-8")
        report = {"file": name, "status": "verified", "applied": False,
                  "final_cnots": updated_entry["final"]["cnot_equiv"],
                  "verified_rewrites": verification["verified_rewrites"]}
        if not apply:
            return report
        # A running driver must not have changed any artifact while replay ran.
        if (results_path.read_bytes() != original_results
                or output_path.read_bytes() != original_output
                or backend._file_sha256_streaming(audit_path) != original_digest):
            raise RuntimeError("Archive changed during recovery; no files replaced.")
        backup = Path(tempfile.mkdtemp(
            prefix="_export-recovery-backup-", dir=results_path.parent
        ))
        for path in (results_path, output_path, audit_path):
            shutil.copy2(path, backup / path.name)
        try:
            os.replace(staged_output, output_path)
            os.replace(staged_audit, audit_path)
            os.replace(staged_results, results_path)
        except BaseException:
            for path in (output_path, audit_path, results_path):
                shutil.copy2(backup / path.name, path)
            raise
        report.update(applied=True, backup=str(backup))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--circuit", action="append", default=[])
    parser.add_argument("--apply", action="store_true",
                        help="Commit verified recoveries; otherwise only verify.")
    args = parser.parse_args()
    failures = 0
    for results_path in args.results:
        entries = json.loads(results_path.read_text(encoding="utf-8"))
        names = [name for name, row in entries.items()
                 if isinstance(row, dict) and row.get("status") == "failed"
                 and EXPORT_FAILURE in row.get("failure", "")
                 and (not args.circuit or name in args.circuit)]
        for name in names:
            print(json.dumps({"file": name, "status": "verifying"}), flush=True)
            try:
                report = recover_one(results_path, name, apply=args.apply)
            except Exception as error:
                failures += 1
                report = {"file": name, "status": "not_recovered",
                          "error": f"{type(error).__name__}: {error}"}
            print(json.dumps(report), flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
