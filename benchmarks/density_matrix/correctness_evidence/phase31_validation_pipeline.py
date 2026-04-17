#!/usr/bin/env python3
"""Run Stage-A Phase 3.1 correctness-evidence validation bundles in one process.

Run with:
    python benchmarks/density_matrix/correctness_evidence/phase31_validation_pipeline.py
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
    phase31_correctness_bundle_validation as phase31_correctness_package,
)
from benchmarks.density_matrix.correctness_evidence import (
    phase31_correctness_matrix_validation as phase31_correctness_matrix,
)
from benchmarks.density_matrix.correctness_evidence import (
    phase31_external_correctness_validation as phase31_external_correctness,
)
from benchmarks.density_matrix.correctness_evidence import (
    phase31_output_integrity_validation as phase31_output_integrity,
)
from benchmarks.density_matrix.correctness_evidence import (
    phase31_runtime_classification_validation as phase31_runtime_classification,
)
from benchmarks.density_matrix.correctness_evidence import (
    phase31_sequential_correctness_validation as phase31_sequential_correctness,
)
from benchmarks.density_matrix.correctness_evidence import (
    phase31_summary_consistency_validation as phase31_summary_consistency,
)
from benchmarks.density_matrix.correctness_evidence.common import DEFAULT_OUTPUT_ROOT
from benchmarks.density_matrix.correctness_evidence.common import (
    PHASE31_CORRECTNESS_EVIDENCE_STAGE_A_ROOT,
    write_artifact_bundle,
)


def _write_slice_bundle(module: Any, bundle: dict) -> Path:
    return write_artifact_bundle(bundle, module.DEFAULT_OUTPUT_DIR, module.ARTIFACT_FILENAME)


_CASE_SLICE_REGISTRY: tuple[tuple[Any, str, str], ...] = (
    (phase31_correctness_matrix, "build_cases", "build_artifact_bundle"),
    (phase31_sequential_correctness, "build_cases", "build_artifact_bundle"),
    (phase31_external_correctness, "build_cases", "build_artifact_bundle"),
    (phase31_output_integrity, "build_cases", "build_artifact_bundle"),
    (phase31_runtime_classification, "build_cases", "build_artifact_bundle"),
)

_NULLARY_BUNDLE_REGISTRY: tuple[tuple[Any, str], ...] = (
    (phase31_correctness_package, "build_artifact_bundle"),
    (phase31_summary_consistency, "build_artifact_bundle"),
)


def run_phase31_pipeline() -> list[tuple[str, str, Path]]:
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

    stage_root = DEFAULT_OUTPUT_ROOT / PHASE31_CORRECTNESS_EVIDENCE_STAGE_A_ROOT
    stage_root.mkdir(parents=True, exist_ok=True)
    results = run_phase31_pipeline()
    for suite_name, status, output_path in results:
        if not args.quiet:
            print(f"{suite_name}: status={status} path={output_path}")
    return 0 if all(status == "pass" for _, status, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
