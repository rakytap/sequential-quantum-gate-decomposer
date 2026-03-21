#!/usr/bin/env python3
"""Run and emit all density-matrix correctness-evidence validation bundles in one process.

Run with:
    python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.correctness_evidence import (
    correctness_bundle_validation as correctness_package,
)
from benchmarks.density_matrix.correctness_evidence import (
    correctness_matrix_validation as correctness_matrix,
)
from benchmarks.density_matrix.correctness_evidence import (
    external_correctness_validation as external_correctness,
)
from benchmarks.density_matrix.correctness_evidence import (
    output_integrity_validation as output_integrity,
)
from benchmarks.density_matrix.correctness_evidence import (
    runtime_classification_validation as runtime_classification,
)
from benchmarks.density_matrix.correctness_evidence import (
    sequential_correctness_validation as sequential_correctness,
)
from benchmarks.density_matrix.correctness_evidence import (
    summary_consistency_validation as summary_consistency,
)
from benchmarks.density_matrix.correctness_evidence import (
    unsupported_boundary_validation as unsupported_boundary,
)
from benchmarks.density_matrix.correctness_evidence.common import DEFAULT_OUTPUT_ROOT
from benchmarks.density_matrix.correctness_evidence.common import (
    write_artifact_bundle,
)


def _write_slice_bundle(module, bundle: dict) -> Path:
    return write_artifact_bundle(bundle, module.DEFAULT_OUTPUT_DIR, module.ARTIFACT_FILENAME)


def run_pipeline() -> list[tuple[str, str, Path]]:
    results: list[tuple[str, str, Path]] = []

    correctness_matrix_cases = correctness_matrix.build_cases()
    correctness_matrix_bundle = correctness_matrix.build_artifact_bundle(
        correctness_matrix_cases
    )
    results.append(
        (
            correctness_matrix.SUITE_NAME,
            correctness_matrix_bundle["status"],
            _write_slice_bundle(correctness_matrix, correctness_matrix_bundle),
        )
    )

    sequential_cases = sequential_correctness.build_cases()
    sequential_bundle = sequential_correctness.build_artifact_bundle(sequential_cases)
    results.append(
        (
            sequential_correctness.SUITE_NAME,
            sequential_bundle["status"],
            _write_slice_bundle(sequential_correctness, sequential_bundle),
        )
    )

    external_cases = external_correctness.build_cases()
    external_bundle = external_correctness.build_artifact_bundle(external_cases)
    results.append(
        (
            external_correctness.SUITE_NAME,
            external_bundle["status"],
            _write_slice_bundle(external_correctness, external_bundle),
        )
    )

    output_integrity_cases = output_integrity.build_cases()
    output_integrity_bundle = output_integrity.build_artifact_bundle(
        output_integrity_cases
    )
    results.append(
        (
            output_integrity.SUITE_NAME,
            output_integrity_bundle["status"],
            _write_slice_bundle(output_integrity, output_integrity_bundle),
        )
    )

    runtime_classification_cases = runtime_classification.build_cases()
    runtime_classification_bundle = runtime_classification.build_artifact_bundle(
        runtime_classification_cases
    )
    results.append(
        (
            runtime_classification.SUITE_NAME,
            runtime_classification_bundle["status"],
            _write_slice_bundle(
                runtime_classification, runtime_classification_bundle
            ),
        )
    )

    unsupported_boundary_cases = unsupported_boundary.build_cases()
    unsupported_boundary_bundle = unsupported_boundary.build_artifact_bundle(
        unsupported_boundary_cases
    )
    results.append(
        (
            unsupported_boundary.SUITE_NAME,
            unsupported_boundary_bundle["status"],
            _write_slice_bundle(unsupported_boundary, unsupported_boundary_bundle),
        )
    )

    correctness_package_bundle = correctness_package.build_artifact_bundle()
    results.append(
        (
            correctness_package.SUITE_NAME,
            correctness_package_bundle["status"],
            _write_slice_bundle(correctness_package, correctness_package_bundle),
        )
    )

    summary_bundle = summary_consistency.build_artifact_bundle()
    results.append(
        (
            summary_consistency.SUITE_NAME,
            summary_bundle["status"],
            _write_slice_bundle(summary_consistency, summary_bundle),
        )
    )

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-bundle console output.",
    )
    args = parser.parse_args(argv)

    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = run_pipeline()
    for suite_name, status, output_path in results:
        if not args.quiet:
            print(f"{suite_name}: status={status} path={output_path}")
    return 0 if all(status == "pass" for _, status, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
