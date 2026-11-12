"""Validate the full refreshed cohort and regenerate its tables and CSV."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def audit_run(path):
    from squander.decomposition._rewrite_audit_stream import _JSONReader
    with gzip.open(path, "rt", encoding="utf-8") as source:
        reader = _JSONReader(source)
        reader.expect("{")
        while reader.peek() != "}":
            key = reader.value()
            reader.expect(":")
            if key == "run":
                return reader.value()
            if key == "events":
                raise ValueError("Audit run metadata must precede the event stream.")
            reader.value()
            if reader.peek() != "}":
                reader.expect(",")
    raise ValueError("Audit has no run metadata.")


def outcome(left, right):
    return "win" if left < right else "tie" if left == right else "loss"


def summarize(state_dir, output_dir):
    from benchmarks.partitioning.compare_osr_bqskit import synthesis_process_tolerance
    from squander import Qiskit_IO
    from squander.decomposition.qgd_Wide_Circuit_Optimization import CircuitGateStats
    from qiskit import qasm2

    inventory = json.loads((state_dir / "inventory.json").read_text())
    rows, groups = [], {}
    for index, row in enumerate(inventory["completed"]):
        state = json.loads((state_dir / f"case-{index:03d}" / "status.json").read_text())
        if state["status"] != "completed":
            raise ValueError(f"Case {index} is {state['status']}; no partial tables are published.")
        old, entry = row["entry"], state["entry"]
        metadata = json.loads((ROOT / row["directory"] / "results.json").read_text())
        if entry != metadata[row["file"]]:
            raise ValueError(f"Refreshed metadata was not installed for case {index}.")
        for stage in ("init", "all_to_all"):
            if entry.get(stage) != old.get(stage):
                raise AssertionError(f"Reused {stage} changed for case {index}.")
        for stage in ("routed", "final"):
            if stage in old and entry[stage]["cnot_equiv"] > old[stage]["cnot_equiv"]:
                raise AssertionError(f"CNOT regression in {stage}, case {index}.")
        proof = entry["rewrite_audit"]
        proof_path = ROOT / proof["file"]
        if digest(proof_path) != proof["sha256"]:
            raise AssertionError(f"Final proof digest mismatch, case {index}.")
        run = audit_run(proof_path)
        qasm_path = ROOT / entry["output_file"]
        if digest(qasm_path) != run["output_file_sha256"]:
            raise AssertionError(f"Final QASM no longer matches its proof, case {index}.")
        if digest(ROOT / run["input_file"]) != run["input_file_sha256"]:
            raise AssertionError(f"Source circuit changed, case {index}.")
        circuit, _ = Qiskit_IO.convert_Qiskit_to_Squander(qasm2.loads(
            qasm_path.read_text(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))
        if CircuitGateStats(circuit) != entry["final"]:
            raise AssertionError(f"Final QASM counts do not match metadata, case {index}.")
        replay = entry.get("routing_replay", {})
        if "routed" in old:
            if not replay.get("a2a_reused") or not replay.get("catalogue_synthesis_reused"):
                raise AssertionError(f"Missing upstream reuse provenance, case {index}.")
            if replay.get("new_routing_synthesis_calls") != 0:
                raise AssertionError(f"Unexpected new routing synthesis, case {index}.")
            if replay["route_bit_identical"] != (replay["final_seconds"] == 0):
                raise AssertionError(f"Invalid final-stage shortcut, case {index}.")
            candidate = replay.get("candidate_audit")
            if candidate:
                path = ROOT / candidate["file"]
                if digest(path) != candidate["sha256"]:
                    raise AssertionError(f"Candidate proof digest mismatch, case {index}.")
                candidate_run = audit_run(path)
                if digest(ROOT / replay["candidate_qasm"]) != candidate_run["output_file_sha256"]:
                    raise AssertionError(f"Candidate QASM mismatch, case {index}.")
        bq_path = ROOT / "benchmarks/partitioning" / f"{row['dataset']}_results_{row['size']}qbit_bqskit/results.json"
        bq = json.loads(bq_path.read_text()).get(row["file"], {})
        key = f"{row['size']}q-{row['dataset']}"
        group = groups.setdefault(key, dict(completed=0, routing_changed=0, routing_unchanged=0,
                                           already_valid=0, all_to_all=dict(win=0, tie=0, loss=0),
                                           routed=dict(win=0, tie=0, loss=0),
                                           final=dict(win=0, tie=0, loss=0)))
        group["completed"] += 1
        group["routing_unchanged" if replay.get("route_bit_identical") else
              "routing_changed" if replay else "already_valid"] += 1
        record = dict(configuration=row["size"], dataset=row["dataset"], circuit=row["file"],
                      old_routed=old.get("routed", {}).get("cnot_equiv"),
                      routed=entry.get("routed", {}).get("cnot_equiv"),
                      old_final=old["final"]["cnot_equiv"], final=entry["final"]["cnot_equiv"],
                      routing_seconds=replay.get("routing_seconds", 0),
                      final_seconds=replay.get("final_seconds", 0),
                      route_bit_identical=replay.get("route_bit_identical"),
                      bqskit_routed=bq.get("routed", {}).get("cnot_equiv"),
                      bqskit_final=bq.get("final", {}).get("cnot_equiv"))
        if bq.get("status") == "completed":
            if entry["init"]["qubits"] != bq["init"]["qubits"] or entry["init"]["cnot_equiv"] != bq["init"]["cnot_equiv"]:
                raise AssertionError(f"Unmatched source circuit in case {index}.")
            if synthesis_process_tolerance(entry) != synthesis_process_tolerance(bq):
                raise AssertionError(f"Unmatched synthesis tolerances in case {index}.")
            for stage in ("all_to_all", "routed", "final"):
                if stage in entry and stage in bq:
                    result = outcome(entry[stage]["cnot_equiv"], bq[stage]["cnot_equiv"])
                    group[stage][result] += 1
                    record[stage + "_outcome"] = result
        rows.append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "routing-replay.csv").open("w") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(policy=inventory["policy"], completed=len(rows),
                   skipped_incomplete=len(inventory["unavailable"]), groups=groups,
                   cnot_regressions=0, timing_basis="archived-complete-run-plus-measured-extension",
                   all_a2a_counts_unchanged=True, all_final_files_bound_to_verified_audits=True)
    (output_dir / "routing-replay-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "benchmarks/partitioning/compare_osr_bqskit.py"),
                    "--no-fetch", "--include-4q", "--output", str(output_dir / "comparison.tex")], check=True)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=ROOT / ".routing-refresh")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "benchmarks/partitioning/results_archive_20261111")
    arguments = parser.parse_args()
    summarize(arguments.state_dir, arguments.output_dir)
