#!/usr/bin/env python3
"""Sensitivity matrix validation for performance evidence.

Records bounded sensitivity across planner-setting identity, noise placement, and
workload identity through one machine-reviewable benchmark surface.

Run with:
    python benchmarks/density_matrix/performance_evidence/sensitivity_matrix_validation.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.common import (
    performance_evidence_output_dir,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_performance_evidence_core_benchmark_records,
)
from benchmarks.density_matrix.performance_evidence.validation_support import (
    assemble_record_schema_case_bundle,
)
from benchmarks.density_matrix.validation_scaffold import (
    require_bundle_fields,
    run_case_slice_cli,
)

SUITE_NAME = "performance_evidence_sensitivity_matrix"
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


def build_sensitivity_matrix_cases() -> list[dict]:
    return [
        case
        for case in build_performance_evidence_core_benchmark_records()
        if case["case_kind"] == "structured_family"
    ]


def build_sensitivity_matrix_bundle(cases: list[dict]) -> dict:
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
        and {entry["noise_pattern"] for entry in entries} == {"sparse", "periodic", "dense"}
        for entries in grouped.values()
    )
    status = "pass" if sensitivity_surface_pass and len(cases) > 0 else "fail"
    summary = {
        "total_cases": len(cases),
        "review_groups": sorted(grouped),
        "full_noise_groups": full_noise_groups,
        "full_seed_groups": full_seed_groups,
        "family_names": sorted({case["family_name"] for case in cases}),
        "structured_qbits": sorted({case["qbit_num"] for case in cases}),
    }
    bundle = assemble_record_schema_case_bundle(SUITE_NAME, status, summary, cases)
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Sensitivity matrix bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_sensitivity_matrix_cases,
        build_artifact_bundle=build_sensitivity_matrix_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the sensitivity matrix bundle into.",
        quiet_report=lambda b: print(
            "full_noise_groups={full_noise_groups}, full_seed_groups={full_seed_groups}".format(
                **b["summary"]
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
