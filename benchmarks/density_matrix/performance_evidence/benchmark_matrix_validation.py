#!/usr/bin/env python3
"""Benchmark matrix validation for performance evidence.

Freezes the dual-anchor benchmark matrix and representative review-set membership
through one stable machine-reviewable surface.

Run with:
    python benchmarks/density_matrix/performance_evidence/benchmark_matrix_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_CONTINUITY,
    PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_STRUCTURED,
    performance_evidence_output_dir,
)
from benchmarks.density_matrix.performance_evidence.case_selection import (
    build_performance_evidence_inventory_cases,
)
from benchmarks.density_matrix.performance_evidence.validation_support import (
    assemble_benchmark_matrix_bundle,
)
from benchmarks.density_matrix.validation_scaffold import (
    require_bundle_fields,
    run_case_slice_cli,
)

SUITE_NAME = "performance_evidence_benchmark_matrix"
ARTIFACT_FILENAME = "benchmark_matrix_bundle.json"
DEFAULT_OUTPUT_DIR = performance_evidence_output_dir("benchmark_matrix")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "selected_candidate",
    "software",
    "summary",
    "cases",
)


def build_benchmark_matrix_cases() -> list[dict]:
    return build_performance_evidence_inventory_cases()


def build_benchmark_matrix_bundle(cases: list[dict]) -> dict:
    summary = {
        "total_cases": len(cases),
        "continuity_cases": sum(
            case["benchmark_slice"] == PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_CONTINUITY
            for case in cases
        ),
        "structured_cases": sum(
            case["benchmark_slice"] == PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_STRUCTURED
            for case in cases
        ),
        "representative_review_cases": sum(case["representative_review_case"] for case in cases),
        "continuity_qbits": sorted(
            {case["qbit_num"] for case in cases if case["case_kind"] == "continuity"}
        ),
        "structured_qbits": sorted(
            {
                case["qbit_num"]
                for case in cases
                if case["case_kind"] == "structured_family"
            }
        ),
        "structured_seed_count": len(
            {
                case["seed"]
                for case in cases
                if case["case_kind"] == "structured_family"
            }
        ),
        "noise_patterns": sorted(
            {
                case["noise_pattern"]
                for case in cases
                if case["noise_pattern"] is not None
            }
        ),
    }
    bundle = assemble_benchmark_matrix_bundle(SUITE_NAME, "pass", summary, cases)
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Benchmark matrix bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_benchmark_matrix_cases,
        build_artifact_bundle=build_benchmark_matrix_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the benchmark matrix bundle into.",
        quiet_report=lambda b: print(
            "total_cases={total_cases}, continuity_cases={continuity_cases}, structured_cases={structured_cases}, representative_review_cases={representative_review_cases}".format(
                **b["summary"]
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
