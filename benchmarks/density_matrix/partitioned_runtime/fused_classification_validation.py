#!/usr/bin/env python3
"""Fusion classification validation: fused, supported-unfused, and deferred labels.

Checks that fused, supported-but-unfused, and deferred or unsupported candidate
records remain explicit with no silent fallback.

Run with:
    python benchmarks/density_matrix/partitioned_runtime/fused_classification_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_surface.common import build_software_metadata
from benchmarks.density_matrix.partitioned_runtime.fusion_case_selection import (
    iter_fusion_microcase_cases,
    iter_fusion_structured_cases,
)
from squander.partitioning.noisy_runtime import (
    PHASE3_FUSION_CLASS_DEFERRED,
    PHASE3_FUSION_CLASS_FUSED,
    PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED,
    PHASE3_RUNTIME_PATH_BASELINE,
    PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS,
    execute_partitioned_density_fused,
)

SUITE_NAME = "phase3_partitioned_runtime_fused_classification"
ARTIFACT_FILENAME = "classification_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "partitioned_runtime"
    / "fused_classification"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "software",
    "summary",
    "cases",
)


def _classification_case(metadata: dict, descriptor_set, parameters) -> dict:
    runtime_result = execute_partitioned_density_fused(descriptor_set, parameters)
    payload = runtime_result.to_dict(include_density_matrix=False)
    classifications = [region["classification"] for region in payload["fused_regions"]]
    return {
        "case_name": metadata["workload_id"],
        "case_kind": metadata["case_kind"],
        "runtime_path": payload["runtime_path"],
        "actual_fused_execution": payload["summary"]["actual_fused_execution"],
        "classifications": classifications,
        "classification_counts": {
            PHASE3_FUSION_CLASS_FUSED: classifications.count(PHASE3_FUSION_CLASS_FUSED),
            PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED: classifications.count(
                PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED
            ),
            PHASE3_FUSION_CLASS_DEFERRED: classifications.count(
                PHASE3_FUSION_CLASS_DEFERRED
            ),
        },
        "classification_pass": (
            (
                payload["summary"]["actual_fused_execution"] is True
                and payload["runtime_path"] == PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS
            )
            or (
                payload["summary"]["actual_fused_execution"] is False
                and payload["runtime_path"] == PHASE3_RUNTIME_PATH_BASELINE
            )
        ),
        "metadata": dict(metadata),
    }


def build_cases() -> list[dict]:
    cases: list[dict] = []
    observed = set()
    for iterator in (iter_fusion_structured_cases(), iter_fusion_microcase_cases()):
        for metadata, descriptor_set, parameters in iterator:
            case = _classification_case(metadata, descriptor_set, parameters)
            cases.append(case)
            observed.update(case["classifications"])
            if observed >= {
                PHASE3_FUSION_CLASS_FUSED,
                PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED,
                PHASE3_FUSION_CLASS_DEFERRED,
            }:
                return cases
    raise RuntimeError(
        "Could not observe all required fusion classifications in sampled workloads"
    )


def build_artifact_bundle(cases: list[dict]) -> dict:
    classification_passes = sum(case["classification_pass"] for case in cases)
    observed = {
        classification
        for case in cases
        for classification in case["classifications"]
    }
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if classification_passes == len(cases)
        and observed
        >= {
            PHASE3_FUSION_CLASS_FUSED,
            PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED,
            PHASE3_FUSION_CLASS_DEFERRED,
        }
        else "fail",
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "classification_passes": classification_passes,
            "observed_classifications": sorted(observed),
            "fused_cases": sum(case["actual_fused_execution"] for case in cases),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Fusion classification bundle missing required fields: {}".format(
                ", ".join(missing)
            )
        )
    return bundle


def write_artifact_bundle(bundle: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ARTIFACT_FILENAME
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the fusion classification bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case console output.",
    )
    args = parser.parse_args(argv)

    cases = build_cases()
    bundle = build_artifact_bundle(cases)
    output_path = write_artifact_bundle(bundle, output_dir=args.output_dir)

    if not args.quiet:
        for case in cases:
            print(
                "{case_name}: path={runtime_path}, classifications={classifications}".format(
                    **case
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
