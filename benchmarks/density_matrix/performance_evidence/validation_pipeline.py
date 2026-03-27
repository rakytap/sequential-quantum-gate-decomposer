#!/usr/bin/env python3
"""Run and write every performance-evidence validation bundle (full pipeline) in one process.

Run with:
    python benchmarks/density_matrix/performance_evidence/validation_pipeline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


def _write_slice_bundle(module: Any, bundle: dict) -> Path:
    return write_artifact_bundle(bundle, module.DEFAULT_OUTPUT_DIR, module.ARTIFACT_FILENAME)


# (module, build_cases_attr, build_bundle_attr)
_CASE_SLICE_REGISTRY: tuple[tuple[Any, str, str], ...] = (
    (benchmark_matrix, "build_benchmark_matrix_cases", "build_benchmark_matrix_bundle"),
    (counted_supported, "build_counted_supported_cases", "build_counted_supported_bundle"),
    (positive_threshold, "build_positive_threshold_cases", "build_positive_threshold_bundle"),
    (sensitivity_matrix, "build_sensitivity_matrix_cases", "build_sensitivity_matrix_bundle"),
    (metric_surface, "build_metric_surface_cases", "build_metric_surface_bundle"),
    (diagnosis, "build_diagnosis_cases", "build_diagnosis_bundle"),
)

# (module, build_bundle_attr)
_SPECIAL_BUNDLE_REGISTRY: tuple[tuple[Any, str], ...] = (
    (benchmark_package, "build_performance_evidence_benchmark_package"),
    (summary_consistency, "build_summary_consistency_bundle"),
)


def run_pipeline() -> list[tuple[str, str, Path]]:
    results: list[tuple[str, str, Path]] = []

    for mod, cases_attr, bundle_attr in _CASE_SLICE_REGISTRY:
        cases = getattr(mod, cases_attr)()
        bundle = getattr(mod, bundle_attr)(cases)
        results.append(
            (mod.SUITE_NAME, bundle["status"], _write_slice_bundle(mod, bundle))
        )

    for mod, bundle_attr in _SPECIAL_BUNDLE_REGISTRY:
        bundle = getattr(mod, bundle_attr)()
        results.append(
            (mod.SUITE_NAME, bundle["status"], _write_slice_bundle(mod, bundle))
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
