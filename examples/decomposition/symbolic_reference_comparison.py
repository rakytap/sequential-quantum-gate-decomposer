#!/usr/bin/env python3
"""Generate phase-verified Qiskit upper bounds for the exact symbolic catalog."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "squander/decomposition"))
from symbolic_bases import BASIS_SETS
from symbolic_references import QISKIT_BASES, catalog_cases, qiskit_baseline, reference_ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", action="append", default=[],
                        help="Preset; repeatable. 'sdk' selects supported presets, 'all' records unsupported cells too")
    parser.add_argument("--gate", action="append", default=[], help="Catalog gate; repeatable; default is every definition")
    parser.add_argument("--seed", type=int, default=219)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bases = args.basis or ["cx_u"]
    if "all" in bases:
        bases = list(BASIS_SETS)
    elif "sdk" in bases:
        bases = list(QISKIT_BASES)
    unknown = set(bases)-set(BASIS_SETS)
    if unknown:
        parser.error(f"Unknown basis: {sorted(unknown)}")
    cases = list(catalog_cases())
    if args.gate:
        requested = set(args.gate)
        cases = [case for case in cases if case[0] in requested]
        if requested-{case[0] for case in cases}:
            parser.error("Select canonical catalog names listed by symbolic_gate_catalog.py")
    ledger = reference_ledger()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                         text=True, capture_output=True)
    report = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
                  git_revision=git.stdout.strip() if git.returncode == 0 else None,
                  versions={p: importlib.metadata.version(p) for p in ("qiskit", "sympy", "numpy")},
                  comparison_contract=ledger["comparison_contract"],
                  cases=len(cases), bases=bases, rows=[])
    source_dir = Path(__file__).resolve().parents[2] / "squander/decomposition"
    source_files = [Path(__file__), *(source_dir / filename for filename in (
        "symbolic_gates.py", "symbolic_bases.py", "symbolic_references.py", "symbolic_reference_ledger.json"))]
    report["source_sha256"] = {str(p.relative_to(source_dir.parents[1])):
                               hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    total = len(cases)*len(bases)
    checkpoint = args.output.with_suffix(".partial.json")
    for name, parameters in cases:
        for basis in bases:
            try:
                row = qiskit_baseline(name, parameters, basis, seed=args.seed)
            except Exception as error:
                row = dict(gate=name, parameters=[str(p) for p in parameters], basis=basis,
                           status="compiler_failed", failures=[dict(input="input_preparation", error=str(error))])
            references = [r for r in ledger["records"]
                          if r["gate"] == name and r["basis"] == basis
                          and r["parameter_domain"] == "fixed gate"]
            row["literature_references"] = [r["id"] for r in references]
            upper = min((r["upper_bound"] for r in references if r["upper_bound"] is not None), default=None)
            row["literature_upper_bound"] = upper
            row["meets_literature_upper_bound"] = (
                None if upper is None or row["status"] != "verified_numeric"
                else row["entanglers"] <= upper)
            report["rows"].append(row)
        if len(report["rows"]) % 25 < len(bases):
            print(f"{len(report['rows'])}/{total} comparisons completed", file=sys.stderr, flush=True)
        if len(report["rows"]) % 100 < len(bases):
            checkpoint.write_text(json.dumps(report, allow_nan=False)+"\n")
    report["summary"] = dict(Counter(row["status"] for row in report["rows"]))
    report["summary"]["literature_count_regressions"] = sum(
        row["meets_literature_upper_bound"] is False for row in report["rows"])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    checkpoint.unlink(missing_ok=True)
    print(json.dumps(report["summary"], sort_keys=True))
    if (report["summary"]["literature_count_regressions"]
            or any(row["status"] == "compiler_failed" for row in report["rows"])):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
