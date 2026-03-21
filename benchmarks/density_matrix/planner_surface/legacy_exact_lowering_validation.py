#!/usr/bin/env python3
"""Validation: Phase 3 Task 1 Story 4 legacy exact lowering.

Builds representative legacy `qgd_Circuit` inputs and verifies that exact
in-bounds cases lower into the canonical planner surface while unsupported gate
families and malformed attached noise schedules fail before planner entry.

Run with:
    python benchmarks/density_matrix/planner_surface/legacy_exact_lowering_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_surface.common import build_software_metadata
from squander import Circuit
from squander.partitioning.noisy_planner import (
    build_canonical_planner_surface_from_qgd_circuit,
    build_planner_audit_record,
)

SUITE_NAME = "phase3_planner_surface_legacy_exact_lowering"
ARTIFACT_FILENAME = "legacy_exact_lowering_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "planner_surface"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "schema_version",
    "software",
    "summary",
    "cases",
)


def positive_case_without_noise() -> dict:
    circuit = Circuit(2)
    circuit.add_U3(0)
    circuit.add_U3(1)
    circuit.add_CNOT(1, 0)
    circuit.add_U3(0)

    surface = build_canonical_planner_surface_from_qgd_circuit(
        circuit,
        workload_id="legacy_manual_u3_cnot",
    )
    record = build_planner_audit_record(surface, metadata={"case_kind": "positive"})
    record.update({"case_name": surface.workload_id, "status": "pass"})
    return record


def positive_case_with_noise() -> dict:
    circuit = Circuit(2)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)

    surface = build_canonical_planner_surface_from_qgd_circuit(
        circuit,
        workload_id="legacy_manual_with_noise",
        density_noise=[
            {
                "channel": "local_depolarizing",
                "target": 1,
                "after_gate_index": 1,
                "error_rate": 0.1,
            }
        ],
    )
    record = build_planner_audit_record(
        surface, metadata={"case_kind": "positive", "has_attached_noise": True}
    )
    record.update({"case_name": surface.workload_id, "status": "pass"})
    return record


def negative_case_unsupported_gate() -> dict:
    circuit = Circuit(1)
    circuit.add_H(0)
    try:
        build_canonical_planner_surface_from_qgd_circuit(
            circuit,
            workload_id="legacy_manual_with_h",
        )
    except Exception as err:
        return {
            "case_name": "legacy_manual_with_h",
            "status": "unsupported",
            "metadata": {"case_kind": "negative"},
            "error": err.to_dict() if hasattr(err, "to_dict") else {"reason": str(err)},
        }
    raise RuntimeError("Unsupported legacy gate case unexpectedly passed")


def negative_case_invalid_noise_index() -> dict:
    circuit = Circuit(2)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)
    try:
        build_canonical_planner_surface_from_qgd_circuit(
            circuit,
            workload_id="legacy_manual_bad_noise_index",
            density_noise=[
                {
                    "channel": "phase_damping",
                    "target": 0,
                    "after_gate_index": 99,
                    "lambda": 0.07,
                }
            ],
        )
    except Exception as err:
        return {
            "case_name": "legacy_manual_bad_noise_index",
            "status": "unsupported",
            "metadata": {"case_kind": "negative"},
            "error": err.to_dict() if hasattr(err, "to_dict") else {"reason": str(err)},
        }
    raise RuntimeError("Invalid legacy noise schedule case unexpectedly passed")


def build_cases() -> list[dict]:
    return [
        positive_case_without_noise(),
        positive_case_with_noise(),
        negative_case_unsupported_gate(),
        negative_case_invalid_noise_index(),
    ]


def build_artifact_bundle(cases: list[dict]) -> dict:
    positive_cases = [case for case in cases if case["status"] == "pass"]
    negative_cases = [case for case in cases if case["status"] == "unsupported"]
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if len(positive_cases) == 2 and len(negative_cases) == 2
        else "fail",
        "schema_version": "phase3_canonical_noisy_planner_v1",
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "positive_cases": len(positive_cases),
            "negative_cases": len(negative_cases),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Legacy exact lowering bundle missing required fields: {}".format(
                ", ".join(missing)
            )
        )
    return bundle


def write_artifact_bundle(bundle: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ARTIFACT_FILENAME
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the legacy exact lowering bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case console output.",
    )
    args = parser.parse_args(argv)

    cases = build_cases()
    bundle = build_artifact_bundle(cases)
    output_path = write_artifact_bundle(bundle, output_dir=args.output_dir)

    if not args.quiet:
        for case in cases:
            print("{case_name}: {status}".format(**case))
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
