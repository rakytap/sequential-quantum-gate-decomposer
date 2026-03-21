#!/usr/bin/env python3
"""Run and emit all publication-evidence validation bundles in one process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_evidence import (
    claim_package_validation as claim_package,
)
from benchmarks.density_matrix.publication_evidence import (
    surface_alignment_validation as surface_alignment,
)
from benchmarks.density_matrix.publication_evidence import (
    claim_traceability_validation as claim_traceability,
)
from benchmarks.density_matrix.publication_evidence import (
    evidence_closure_validation as evidence_closure,
)
from benchmarks.density_matrix.publication_evidence import (
    supported_path_validation as supported_path_scope,
)
from benchmarks.density_matrix.publication_evidence import (
    publication_manifest_validation as publication_manifest,
)
from benchmarks.density_matrix.publication_evidence import (
    future_work_boundary_validation as future_work_boundary,
)
from benchmarks.density_matrix.publication_evidence import (
    package_consistency_validation as package_consistency,
)
from benchmarks.density_matrix.publication_evidence.common import (
    DEFAULT_OUTPUT_ROOT,
    write_artifact_bundle,
)


def _write_slice_bundle(module, bundle: dict) -> Path:
    return write_artifact_bundle(bundle, module.DEFAULT_OUTPUT_DIR, module.ARTIFACT_FILENAME)


def run_pipeline() -> list[tuple[str, str, Path]]:
    results: list[tuple[str, str, Path]] = []

    claim_package_bundle = claim_package.build_artifact_bundle()
    results.append(
        (
            claim_package.SUITE_NAME,
            claim_package_bundle["status"],
            _write_slice_bundle(claim_package, claim_package_bundle),
        )
    )

    surface_alignment_bundle = surface_alignment.build_artifact_bundle()
    results.append(
        (
            surface_alignment.SUITE_NAME,
            surface_alignment_bundle["status"],
            _write_slice_bundle(surface_alignment, surface_alignment_bundle),
        )
    )

    claim_traceability_bundle = claim_traceability.build_artifact_bundle()
    results.append(
        (
            claim_traceability.SUITE_NAME,
            claim_traceability_bundle["status"],
            _write_slice_bundle(claim_traceability, claim_traceability_bundle),
        )
    )

    evidence_closure_bundle = evidence_closure.build_artifact_bundle()
    results.append(
        (
            evidence_closure.SUITE_NAME,
            evidence_closure_bundle["status"],
            _write_slice_bundle(evidence_closure, evidence_closure_bundle),
        )
    )

    supported_path_scope_bundle = supported_path_scope.build_artifact_bundle()
    results.append(
        (
            supported_path_scope.SUITE_NAME,
            supported_path_scope_bundle["status"],
            _write_slice_bundle(supported_path_scope, supported_path_scope_bundle),
        )
    )

    publication_manifest_bundle = publication_manifest.build_artifact_bundle()
    results.append(
        (
            publication_manifest.SUITE_NAME,
            publication_manifest_bundle["status"],
            _write_slice_bundle(publication_manifest, publication_manifest_bundle),
        )
    )

    future_work_bundle = future_work_boundary.build_artifact_bundle()
    results.append(
        (
            future_work_boundary.SUITE_NAME,
            future_work_bundle["status"],
            _write_slice_bundle(future_work_boundary, future_work_bundle),
        )
    )

    package_consistency_bundle = package_consistency.build_artifact_bundle()
    results.append(
        (
            package_consistency.SUITE_NAME,
            package_consistency_bundle["status"],
            _write_slice_bundle(package_consistency, package_consistency_bundle),
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
