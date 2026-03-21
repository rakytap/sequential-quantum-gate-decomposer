#!/usr/bin/env python3
"""Run and emit all Phase 3 performance-evidence validation bundles in one process.

Run with:
    python benchmarks/density_matrix/performance_evidence/validation_pipeline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence import (  # noqa: E402
    benchmark_bundle_validation as benchmark_package,
)
from benchmarks.density_matrix.performance_evidence import (  # noqa: E402
    benchmark_matrix_validation as benchmark_matrix,
)
from benchmarks.density_matrix.performance_evidence import (  # noqa: E402
    counted_supported_validation as counted_supported,
)
from benchmarks.density_matrix.performance_evidence import (  # noqa: E402
    diagnosis_validation as diagnosis,
)
from benchmarks.density_matrix.performance_evidence import (  # noqa: E402
    metric_surface_validation as metric_surface,
)
from benchmarks.density_matrix.performance_evidence import (  # noqa: E402
    positive_threshold_validation as positive_threshold,
)
from benchmarks.density_matrix.performance_evidence import (  # noqa: E402
    sensitivity_matrix_validation as sensitivity_matrix,
)
from benchmarks.density_matrix.performance_evidence import (  # noqa: E402
    summary_consistency_validation as summary_consistency,
)
from benchmarks.density_matrix.performance_evidence.common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    write_artifact_bundle,
)


def _write_slice_bundle(module, bundle: dict) -> Path:
    return write_artifact_bundle(bundle, module.DEFAULT_OUTPUT_DIR, module.ARTIFACT_FILENAME)


def run_pipeline() -> list[tuple[str, str, Path]]:
    results: list[tuple[str, str, Path]] = []

    benchmark_matrix_cases = benchmark_matrix.build_cases()
    benchmark_matrix_bundle = benchmark_matrix.build_artifact_bundle(
        benchmark_matrix_cases
    )
    results.append(
        (
            benchmark_matrix.SUITE_NAME,
            benchmark_matrix_bundle["status"],
            _write_slice_bundle(benchmark_matrix, benchmark_matrix_bundle),
        )
    )

    counted_supported_cases = counted_supported.build_cases()
    counted_supported_bundle = counted_supported.build_artifact_bundle(
        counted_supported_cases
    )
    results.append(
        (
            counted_supported.SUITE_NAME,
            counted_supported_bundle["status"],
            _write_slice_bundle(counted_supported, counted_supported_bundle),
        )
    )

    positive_threshold_cases = positive_threshold.build_cases()
    positive_threshold_bundle = positive_threshold.build_artifact_bundle(
        positive_threshold_cases
    )
    results.append(
        (
            positive_threshold.SUITE_NAME,
            positive_threshold_bundle["status"],
            _write_slice_bundle(positive_threshold, positive_threshold_bundle),
        )
    )

    sensitivity_cases = sensitivity_matrix.build_cases()
    sensitivity_bundle = sensitivity_matrix.build_artifact_bundle(sensitivity_cases)
    results.append(
        (
            sensitivity_matrix.SUITE_NAME,
            sensitivity_bundle["status"],
            _write_slice_bundle(sensitivity_matrix, sensitivity_bundle),
        )
    )

    metric_surface_cases = metric_surface.build_cases()
    metric_surface_bundle = metric_surface.build_artifact_bundle(
        metric_surface_cases
    )
    results.append(
        (
            metric_surface.SUITE_NAME,
            metric_surface_bundle["status"],
            _write_slice_bundle(metric_surface, metric_surface_bundle),
        )
    )

    diagnosis_cases = diagnosis.build_cases()
    diagnosis_bundle = diagnosis.build_artifact_bundle(diagnosis_cases)
    results.append(
        (
            diagnosis.SUITE_NAME,
            diagnosis_bundle["status"],
            _write_slice_bundle(diagnosis, diagnosis_bundle),
        )
    )

    benchmark_package_bundle = benchmark_package.build_artifact_bundle()
    results.append(
        (
            benchmark_package.SUITE_NAME,
            benchmark_package_bundle["status"],
            _write_slice_bundle(benchmark_package, benchmark_package_bundle),
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
