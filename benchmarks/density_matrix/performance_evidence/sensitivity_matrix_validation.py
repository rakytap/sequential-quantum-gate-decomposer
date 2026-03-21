#!/usr/bin/env python3
"""Validation: Phase 3 Task 7 Story 4 sensitivity surface.

Records bounded sensitivity across planner-setting identity, noise placement,
and workload identity through one machine-reviewable benchmark surface.

Run with:
    python benchmarks/density_matrix/performance_evidence/sensitivity_matrix_validation.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
    build_performance_evidence_selected_candidate,
    build_performance_evidence_software_metadata,
    performance_evidence_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_performance_evidence_core_benchmark_records,
)

SUITE_NAME = "phase3_performance_evidence_sensitivity_matrix"
ARTIFACT_FILENAME = "sensitivity_matrix_bundle.json"
DEFAULT_OUTPUT_DIR = performance_evidence_output_dir("sensitivity_matrix")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "record_schema_version",
    "software",
    "selected_candidate",
    "summary",
    "cases",
)


def build_cases() -> list[dict]:
    return [
        case
        for case in build_performance_evidence_core_benchmark_records()
        if case["case_kind"] == "structured_family"
    ]


def build_artifact_bundle(cases: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        grouped[case["review_group_id"]].append(case)

    full_noise_groups = sum(
        {entry["noise_pattern"] for entry in entries} == {"sparse", "periodic", "dense"}
        for entries in grouped.values()
    )
    full_seed_groups = sum(
        len({entry["seed"] for entry in entries}) >= 3 for entries in grouped.values()
    )
    sensitivity_surface_pass = all(
        len({entry["seed"] for entry in entries}) >= 3
        and {entry["noise_pattern"] for entry in entries}
        == {"sparse", "periodic", "dense"}
        for entries in grouped.values()
    )
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if sensitivity_surface_pass and len(cases) > 0 else "fail",
        "record_schema_version": PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
        "software": build_performance_evidence_software_metadata(),
        "selected_candidate": build_performance_evidence_selected_candidate(),
        "summary": {
            "total_cases": len(cases),
            "review_groups": sorted(grouped),
            "full_noise_groups": full_noise_groups,
            "full_seed_groups": full_seed_groups,
            "family_names": sorted({case["family_name"] for case in cases}),
            "structured_qbits": sorted({case["qbit_num"] for case in cases}),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 7 Story 4 bundle missing required fields: {}".format(
                ", ".join(missing)
            )
        )
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the Task 7 Story 4 bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output.",
    )
    args = parser.parse_args(argv)

    cases = build_cases()
    bundle = build_artifact_bundle(cases)
    output_path = write_artifact_bundle(bundle, args.output_dir, ARTIFACT_FILENAME)

    if not args.quiet:
        print(
            "full_noise_groups={full_noise_groups}, full_seed_groups={full_seed_groups}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
