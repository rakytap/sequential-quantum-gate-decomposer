#!/usr/bin/env python3
"""Run and emit all density-matrix correctness-evidence validation bundles in one process.

Run with:
    python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


def _write_slice_bundle(module: Any, bundle: dict) -> Path:
    return write_artifact_bundle(bundle, module.DEFAULT_OUTPUT_DIR, module.ARTIFACT_FILENAME)


# (module, build_cases_attr, build_bundle_attr)
_CASE_SLICE_REGISTRY: tuple[tuple[Any, str, str], ...] = (
    (correctness_matrix, "build_cases", "build_artifact_bundle"),
    (sequential_correctness, "build_cases", "build_artifact_bundle"),
    (external_correctness, "build_cases", "build_artifact_bundle"),
    (output_integrity, "build_cases", "build_artifact_bundle"),
    (runtime_classification, "build_cases", "build_artifact_bundle"),
    (unsupported_boundary, "build_cases", "build_artifact_bundle"),
)

# (module, build_bundle_attr) — nullary bundle builders
_NULLARY_BUNDLE_REGISTRY: tuple[tuple[Any, str], ...] = (
    (correctness_package, "build_artifact_bundle"),
    (summary_consistency, "build_artifact_bundle"),
)


def run_pipeline() -> list[tuple[str, str, Path]]:
    results: list[tuple[str, str, Path]] = []

    for mod, cases_attr, bundle_attr in _CASE_SLICE_REGISTRY:
        cases = getattr(mod, cases_attr)()
        bundle = getattr(mod, bundle_attr)(cases)
        results.append(
            (mod.SUITE_NAME, bundle["status"], _write_slice_bundle(mod, bundle))
        )

    for mod, bundle_attr in _NULLARY_BUNDLE_REGISTRY:
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
