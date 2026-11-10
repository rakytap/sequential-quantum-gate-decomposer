#!/usr/bin/env python3
"""Prove exact SDK proposals for fixed-target local-count regressions."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"squander/decomposition"))
from symbolic_archive import comparison_contract, wall_time_limit, SymbolicDeadline
from symbolic_contract import retain_phase_results
from symbolic_gates import make_gate
from symbolic_sdk_lift import lift_sdk_candidate
from symbolic_synthesis_archive import source_hashes, write_report


def projective_result(row):
    return row.get("projective") or (row.get("phase_exact") if row.get("projective_omitted_as_no_better") else None)


def refine(job):
    row, baseline, seconds = job
    results = {}
    for mode in ("phase_exact", "projective"):
        start = time.monotonic()
        try:
            with wall_time_limit(seconds):
                result = lift_sdk_candidate(make_gate(row["gate"]), baseline, row["basis"],
                                             phase_exact=mode == "phase_exact")
        except SymbolicDeadline:
            result = dict(status="timeout", reason="Exact SDK proposal proof budget")
        except Exception as error:
            result = dict(status="compiler_error", error_type=type(error).__name__, reason=str(error))
        results[mode] = dict(result, phase_exact=mode == "phase_exact", elapsed_seconds=time.monotonic()-start)
    return dict(gate=row["gate"], parameters=row["parameters"], target_scope=row["target_scope"],
                basis=row["basis"], phase_statuses={mode:result["status"] for mode,result in results.items()},
                **retain_phase_results(results["phase_exact"], results["projective"]))


def load(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seconds-per-mode", type=float, default=45)
    args = parser.parse_args()
    if args.workers < 1 or args.seconds_per_mode <= 0:
        parser.error("Positive worker count and budget required")
    archive, references = load(args.input), load(args.baselines)
    if archive["comparison_contract"] != comparison_contract():
        parser.error("Archive contract differs")
    sdk = {(r["gate"], r["basis"]):r for r in references["rows"] if r["status"] == "verified_numeric"}
    jobs = []
    for row in archive["rows"]:
        candidate = projective_result(row)
        baseline = sdk.get((row["gate"], row["basis"]))
        if row["target_scope"] != "fixed_gate" or not candidate or not baseline:
            continue
        if candidate["status"] != "verified_symbolic" or candidate["cost"]["entanglers"] != baseline["entanglers"]:
            continue
        local = sum(len(op["wires"]) == 1 for op in baseline["operations"])
        if candidate["cost"]["single_qubit_gates"] > local:
            jobs.append((row, baseline, args.seconds_per_mode))
    hashes = source_hashes()
    hashes[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
        git_revision=revision, source_sha256=hashes, versions=references["versions"],
        comparison_contract=comparison_contract(), budget_seconds_per_mode=args.seconds_per_mode,
        selection=dict(kind="fixed_target_equal_entanglers_higher_local_count",
            archive=str(args.input), archive_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
            baselines=str(args.baselines), baseline_sha256=hashlib.sha256(args.baselines.read_bytes()).hexdigest()),
        rows=[], clifford_certificates={})
    checkpoint = args.output.with_name(args.output.name+".partial.json.gz")
    with mp.get_context("fork").Pool(args.workers) as pool:
        for result in pool.imap_unordered(refine, jobs):
            report["rows"].append(result)
            if len(report["rows"]) % 10 == 0:
                write_report(checkpoint, report)
                print(f"{len(report['rows'])}/{len(jobs)} exact-lift cells", flush=True)
    report["rows"].sort(key=lambda row:(row["gate"].casefold(), row["basis"]))
    report["summary"] = {mode:dict(Counter(r["phase_statuses"][mode] for r in report["rows"]))
                           for mode in ("phase_exact", "projective")}
    report["summary"]["cells"] = len(report["rows"])
    report["complete_selected_cells"] = len(report["rows"]) == len(jobs)
    write_report(args.output, report)
    checkpoint.unlink(missing_ok=True)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
