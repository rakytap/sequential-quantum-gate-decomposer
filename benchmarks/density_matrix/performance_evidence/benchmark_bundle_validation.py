#!/usr/bin/env python3
"""Validation: Phase 3 Task 7 Story 7 shared benchmark package.

Packages counted, diagnosis-only, and excluded Task 7 evidence together with the
explicit boundary evidence later publication consumers need.

Run with:
    python benchmarks/density_matrix/performance_evidence/benchmark_bundle_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.bundle import (
    build_task7_benchmark_package_payload,
)
from benchmarks.density_matrix.performance_evidence.common import (
    TASK7_BENCHMARK_PACKAGE_SCHEMA_VERSION,
    build_task7_software_metadata,
    task7_story_output_dir,
    write_artifact_bundle,
)

SUITE_NAME = "phase3_task7_story7_benchmark_package"
ARTIFACT_FILENAME = "benchmark_package_bundle.json"
DEFAULT_OUTPUT_DIR = task7_story_output_dir("story7_benchmark_package")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "schema_version",
    "software",
    "selected_candidate",
    "summary",
    "required_artifacts",
    "cases",
    "negative_cases",
)


def build_artifact_bundle() -> dict:
    payload = build_task7_benchmark_package_payload()
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if payload["summary"]["total_cases"] == len(payload["cases"])
        and payload["summary"]["task6_boundary_cases"] == len(payload["negative_cases"])
        else "fail",
        "schema_version": TASK7_BENCHMARK_PACKAGE_SCHEMA_VERSION,
        "software": build_task7_software_metadata(),
        "selected_candidate": payload["selected_candidate"],
        "summary": dict(payload["summary"]),
        "required_artifacts": list(payload["required_artifacts"]),
        "cases": payload["cases"],
        "negative_cases": payload["negative_cases"],
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 7 Story 7 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 7 Story 7 bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output.",
    )
    args = parser.parse_args(argv)

    bundle = build_artifact_bundle()
    output_path = write_artifact_bundle(bundle, args.output_dir, ARTIFACT_FILENAME)

    if not args.quiet:
        print(
            "counted_supported_cases={counted_supported_cases}, diagnosis_only_cases={diagnosis_only_cases}, excluded_cases={excluded_cases}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
