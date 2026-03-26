#!/usr/bin/env python3
"""Partition-descriptor audit surface validation.

Emits audit-oriented descriptor records for one continuity case, one microcase,
one structured-family case, and one supported legacy exact-lowering case.
Validates that the shared descriptor audit surface carries stable provenance
and summary metadata across supported workload classes.

Run with:
    python benchmarks/density_matrix/planner_surface/descriptor_audit_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_surface.common import (
    build_phase2_continuity_vqe,
    build_software_metadata,
)
from benchmarks.density_matrix.planner_surface.workloads import (
    iter_microcase_descriptor_sets,
    iter_structured_descriptor_sets,
)
from squander import Circuit
from squander.partitioning.noisy_planner import (
    DESCRIPTOR_SCHEMA_VERSION,
    build_canonical_planner_surface_from_qgd_circuit,
    build_descriptor_audit_record,
    build_partition_descriptor_set,
    build_phase3_continuity_partition_descriptor_set,
)

SUITE_NAME = "phase3_planner_surface_descriptor_audit"
ARTIFACT_FILENAME = "descriptor_audit_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_surface"
    / "descriptor_audit"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "schema_version",
    "software",
    "summary",
    "cases",
)


def build_cases() -> list[dict]:
    continuity_vqe, _, _ = build_phase2_continuity_vqe(4)
    continuity_audit = build_descriptor_audit_record(
        build_phase3_continuity_partition_descriptor_set(continuity_vqe),
        metadata={"case_kind": "continuity"},
    )
    continuity_audit["case_name"] = continuity_audit["provenance"]["workload_id"]

    micro_metadata, micro_descriptor_set = next(iter(iter_microcase_descriptor_sets()))
    micro_audit = build_descriptor_audit_record(
        micro_descriptor_set, metadata=micro_metadata
    )
    micro_audit["case_name"] = micro_audit["provenance"]["workload_id"]

    structured_metadata, structured_descriptor_set = next(
        iter(iter_structured_descriptor_sets())
    )
    structured_audit = build_descriptor_audit_record(
        structured_descriptor_set, metadata=structured_metadata
    )
    structured_audit["case_name"] = structured_audit["provenance"]["workload_id"]

    legacy_circuit = Circuit(2)
    legacy_circuit.add_U3(0)
    legacy_circuit.add_CNOT(1, 0)
    legacy_descriptor_set = build_partition_descriptor_set(
        build_canonical_planner_surface_from_qgd_circuit(
            legacy_circuit, workload_id="legacy_descriptor_audit"
        )
    )
    legacy_audit = build_descriptor_audit_record(
        legacy_descriptor_set, metadata={"case_kind": "legacy_exact"}
    )
    legacy_audit["case_name"] = legacy_audit["provenance"]["workload_id"]

    return [continuity_audit, micro_audit, structured_audit, legacy_audit]


def build_artifact_bundle(cases: list[dict]) -> dict:
    summary_key_sets = {frozenset(case["summary"].keys()) for case in cases}
    partition_key_sets = {
        frozenset(case["partitions"][0].keys()) for case in cases if case["partitions"]
    }
    passed_cases = sum(case["summary"]["partition_count"] > 0 for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if passed_cases == len(cases)
        and len(summary_key_sets) == 1
        and len(partition_key_sets) == 1
        else "fail",
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "summary_schema_count": len(summary_key_sets),
            "partition_schema_count": len(partition_key_sets),
            "continuity_cases": sum(
                case["metadata"].get("case_kind") == "continuity" for case in cases
            ),
            "methods_cases": sum(
                case["metadata"].get("case_kind") in {"microcase", "structured_family"}
                for case in cases
            ),
            "legacy_cases": sum(
                case["metadata"].get("case_kind") == "legacy_exact" for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Descriptor audit bundle missing required fields: {}".format(
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
        help="Directory to write the descriptor audit bundle into.",
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
            print(
                "{case_name}: partitions={partition_count}, source_type={source_type}".format(
                    case_name=case["case_name"],
                    partition_count=case["summary"]["partition_count"],
                    source_type=case["provenance"]["source_type"],
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
