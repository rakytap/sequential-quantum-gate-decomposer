#!/usr/bin/env python3
"""Shared benchmark package validation for performance evidence.

Packages counted, diagnosis-only, and excluded evidence together with explicit
boundary records for downstream publication consumers.

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
    build_performance_evidence_benchmark_package_payload,
)
from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_SCHEMA_VERSION,
    build_performance_evidence_software_metadata,
    performance_evidence_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.validation_scaffold import require_bundle_fields

SUITE_NAME = "performance_evidence_benchmark_package"
ARTIFACT_FILENAME = "benchmark_package_bundle.json"
DEFAULT_OUTPUT_DIR = performance_evidence_output_dir("benchmark_package")
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


def build_performance_evidence_benchmark_package() -> dict:
    payload = build_performance_evidence_benchmark_package_payload()
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if payload["summary"]["total_cases"] == len(payload["cases"])
        and payload["summary"]["correctness_evidence_boundary_cases"] == len(payload["negative_cases"])
        else "fail",
        "schema_version": PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_SCHEMA_VERSION,
        "software": build_performance_evidence_software_metadata(),
        "selected_candidate": payload["selected_candidate"],
        "summary": dict(payload["summary"]),
        "required_artifacts": list(payload["required_artifacts"]),
        "cases": payload["cases"],
        "negative_cases": payload["negative_cases"],
    }
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Performance evidence benchmark package")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the benchmark package bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output.",
    )
    args = parser.parse_args(argv)

    bundle = build_performance_evidence_benchmark_package()
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
