#!/usr/bin/env python3
"""Validation: Phase 3 planner-candidate audit surface.

Emits one auditable candidate surface for the current noisy planner by showing
that each supported planner candidate can build representative continuity,
microcase, and structured descriptor sets on the frozen Phase 3 workload
surface.

Run with:
    python benchmarks/density_matrix/planner_calibration/planner_candidate_audit_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_calibration.common import (
    PLANNER_CANDIDATE_SCHEMA_VERSION,
    build_planner_candidates,
)
from benchmarks.density_matrix.planner_surface.common import (
    build_phase2_continuity_vqe,
    build_software_metadata,
)
from benchmarks.density_matrix.planner_surface.workloads import (
    iter_microcase_descriptor_sets,
    iter_structured_descriptor_sets,
)
from squander.partitioning.noisy_planner import build_phase3_continuity_partition_descriptor_set

SUITE_NAME = "phase3_planner_calibration_candidate_audit"
ARTIFACT_FILENAME = "planner_candidate_audit_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_calibration"
    / "planner_candidate_audit"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "candidate_schema_version",
    "software",
    "summary",
    "cases",
)
REQUIRED_CASE_KINDS = ("continuity", "microcase", "structured_family")


def _descriptor_summary(case_kind: str, metadata: dict, descriptor_set) -> dict:
    payload = descriptor_set.to_dict()
    return {
        "case_kind": case_kind,
        "workload_id": payload["workload_id"],
        "qbit_num": payload["qbit_num"],
        "max_partition_qubits": payload["max_partition_qubits"],
        "partition_count": payload["partition_count"],
        "descriptor_member_count": payload["descriptor_member_count"],
        "max_partition_span": payload["max_partition_span"],
        "partition_member_counts": payload["partition_member_counts"],
        "family_name": metadata.get("family_name"),
        "noise_pattern": metadata.get("noise_pattern"),
        "source_type": payload["source_type"],
    }


def _representative_continuity_case(candidate) -> dict:
    qbit_num = 4
    vqe, _, topology = build_phase2_continuity_vqe(qbit_num)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(
        vqe,
        workload_id=f"phase2_xxz_hea_q{qbit_num}_continuity",
        max_partition_qubits=candidate.max_partition_qubits,
    )
    return _descriptor_summary(
        "continuity",
        {"topology": topology},
        descriptor_set,
    )


def _representative_microcase(candidate) -> dict:
    metadata, descriptor_set = next(
        iter(
            iter_microcase_descriptor_sets(
                max_partition_qubits=candidate.max_partition_qubits
            )
        )
    )
    return _descriptor_summary("microcase", metadata, descriptor_set)


def _representative_structured_case(candidate) -> dict:
    metadata, descriptor_set = next(
        iter(
            iter_structured_descriptor_sets(
                max_partition_qubits=candidate.max_partition_qubits
            )
        )
    )
    return _descriptor_summary("structured_family", metadata, descriptor_set)


def _candidate_case(candidate) -> dict:
    representative_cases = [
        _representative_continuity_case(candidate),
        _representative_microcase(candidate),
        _representative_structured_case(candidate),
    ]
    supported_case_kinds = [case["case_kind"] for case in representative_cases]
    representative_workload_ids = [case["workload_id"] for case in representative_cases]
    candidate_payload = candidate.to_dict()
    return {
        **candidate_payload,
        "supported_case_kinds": supported_case_kinds,
        "representative_workload_ids": representative_workload_ids,
        "representative_cases": representative_cases,
        "candidate_surface_pass": (
            candidate_payload["schema_version"] == PLANNER_CANDIDATE_SCHEMA_VERSION
            and supported_case_kinds == list(REQUIRED_CASE_KINDS)
            and len(set(representative_workload_ids)) == len(REQUIRED_CASE_KINDS)
            and all(
                case["max_partition_qubits"] == candidate.max_partition_qubits
                for case in representative_cases
            )
            and all(case["partition_count"] > 0 for case in representative_cases)
        ),
    }


def build_cases() -> list[dict]:
    return [_candidate_case(candidate) for candidate in build_planner_candidates()]


def build_artifact_bundle(cases: list[dict]) -> dict:
    candidate_ids = [case["candidate_id"] for case in cases]
    candidate_passes = sum(case["candidate_surface_pass"] for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if candidate_passes == len(cases) and len(set(candidate_ids)) == len(candidate_ids)
        else "fail",
        "candidate_schema_version": PLANNER_CANDIDATE_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_candidates": len(cases),
            "candidate_passes": candidate_passes,
            "candidate_ids": candidate_ids,
            "planner_families": sorted({case["planner_family"] for case in cases}),
            "max_partition_qubits_values": [
                case["max_partition_qubits"] for case in cases
            ],
            "representative_case_kinds": list(REQUIRED_CASE_KINDS),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Planner-candidate audit bundle missing required fields: {}".format(
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
        help="Directory to write the planner-candidate audit bundle into.",
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
                "{candidate_id}: max_partition_qubits={max_partition_qubits}, cases={supported_case_kinds}".format(
                    **case
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
