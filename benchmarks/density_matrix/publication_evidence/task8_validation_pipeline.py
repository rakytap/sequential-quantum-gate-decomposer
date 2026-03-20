#!/usr/bin/env python3
"""Run and emit all Phase 3 Task 8 story validation bundles in one process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_evidence import (
    claim_package_validation as story1,
)
from benchmarks.density_matrix.publication_evidence import (
    surface_alignment_validation as story2,
)
from benchmarks.density_matrix.publication_evidence import (
    claim_traceability_validation as story3,
)
from benchmarks.density_matrix.publication_evidence import (
    evidence_closure_validation as story4,
)
from benchmarks.density_matrix.publication_evidence import (
    supported_path_validation as story5,
)
from benchmarks.density_matrix.publication_evidence import (
    publication_manifest_validation as story6,
)
from benchmarks.density_matrix.publication_evidence import (
    future_work_boundary_validation as story7,
)
from benchmarks.density_matrix.publication_evidence import (
    package_consistency_validation as story8,
)
from benchmarks.density_matrix.publication_evidence.common import (
    DEFAULT_OUTPUT_ROOT,
    write_artifact_bundle,
)


def _write_story_bundle(module, bundle: dict) -> Path:
    return write_artifact_bundle(bundle, module.DEFAULT_OUTPUT_DIR, module.ARTIFACT_FILENAME)


def run_pipeline() -> list[tuple[str, str, Path]]:
    results: list[tuple[str, str, Path]] = []

    bundle1 = story1.build_artifact_bundle()
    results.append((story1.SUITE_NAME, bundle1["status"], _write_story_bundle(story1, bundle1)))

    bundle2 = story2.build_artifact_bundle()
    results.append((story2.SUITE_NAME, bundle2["status"], _write_story_bundle(story2, bundle2)))

    bundle3 = story3.build_artifact_bundle()
    results.append((story3.SUITE_NAME, bundle3["status"], _write_story_bundle(story3, bundle3)))

    bundle4 = story4.build_artifact_bundle()
    results.append((story4.SUITE_NAME, bundle4["status"], _write_story_bundle(story4, bundle4)))

    bundle5 = story5.build_artifact_bundle()
    results.append((story5.SUITE_NAME, bundle5["status"], _write_story_bundle(story5, bundle5)))

    bundle6 = story6.build_artifact_bundle()
    results.append((story6.SUITE_NAME, bundle6["status"], _write_story_bundle(story6, bundle6)))

    bundle7 = story7.build_artifact_bundle()
    results.append((story7.SUITE_NAME, bundle7["status"], _write_story_bundle(story7, bundle7)))

    bundle8 = story8.build_artifact_bundle()
    results.append((story8.SUITE_NAME, bundle8["status"], _write_story_bundle(story8, bundle8)))

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
