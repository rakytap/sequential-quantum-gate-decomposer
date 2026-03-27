#!/usr/bin/env python3
"""Fused eligibility validation: eligible unitary regions on the baseline runtime path.

Finds one representative continuity, microcase, and structured workload whose
descriptor-level runtime evidence exposes at least one eligible unitary-island
span on the shared baseline partitioned runtime surface.

Run with:
    python benchmarks/density_matrix/partitioned_runtime/fused_eligibility_validation.py
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
    iter_fusion_continuity_cases,
    iter_fusion_microcase_cases,
    iter_fusion_structured_cases,
)
from squander.partitioning.noisy_runtime import (
    PHASE3_FUSION_CLASS_DEFERRED,
    PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED,
    PHASE3_FUSION_KIND_NOISE_BOUNDARY,
    PHASE3_FUSION_KIND_UNITARY_ISLAND,
    PHASE3_RUNTIME_PATH_BASELINE,
    execute_partitioned_density,
)

SUITE_NAME = "phase3_partitioned_runtime_fused_eligibility"
ARTIFACT_FILENAME = "eligibility_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "partitioned_runtime"
    / "fused_eligibility"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "software",
    "summary",
    "cases",
)


def _case_from_runtime(metadata: dict, runtime_result) -> dict:
    payload = runtime_result.to_dict(include_density_matrix=False)
    fused_regions = payload["fused_regions"]
    eligible_regions = [
        region
        for region in fused_regions
        if region["candidate_kind"] == PHASE3_FUSION_KIND_UNITARY_ISLAND
        and region["classification"] == PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED
        and region["reason"] == "fusion_disabled"
    ]
    singleton_regions = [
        region
        for region in fused_regions
        if region["candidate_kind"] == PHASE3_FUSION_KIND_UNITARY_ISLAND
        and region["classification"] == PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED
        and region["reason"] == "singleton_unitary_region"
    ]
    deferred_regions = [
        region
        for region in fused_regions
        if region["candidate_kind"] == PHASE3_FUSION_KIND_NOISE_BOUNDARY
        and region["classification"] == PHASE3_FUSION_CLASS_DEFERRED
    ]
    return {
        "case_name": metadata["case_name"] if "case_name" in metadata else metadata["workload_id"],
        "case_kind": metadata["case_kind"],
        "workload_id": payload["workload_id"],
        "runtime_path": payload["runtime_path"],
        "partition_count": payload["summary"]["partition_count"],
        "eligible_unitary_region_count": len(eligible_regions),
        "singleton_unitary_region_count": len(singleton_regions),
        "deferred_noise_boundary_count": len(deferred_regions),
        "fused_region_count": payload["summary"]["fused_region_count"],
        "supported_unfused_region_count": payload["summary"][
            "supported_unfused_region_count"
        ],
        "deferred_region_count": payload["summary"]["deferred_region_count"],
        "eligibility_pass": (
            payload["runtime_path"] == PHASE3_RUNTIME_PATH_BASELINE
            and len(eligible_regions) > 0
            and payload["summary"]["fused_region_count"] == 0
        ),
        "metadata": dict(metadata),
    }


def _select_first_case(case_iter) -> dict:
    for case in case_iter:
        if len(case) == 4:
            metadata, descriptor_set, parameters, _ = case
        else:
            metadata, descriptor_set, parameters = case
        runtime_result = execute_partitioned_density(descriptor_set, parameters)
        case_record = _case_from_runtime(metadata, runtime_result)
        if case_record["eligibility_pass"]:
            return case_record
    raise RuntimeError(
        "No representative fused-eligibility case exposed an eligible unitary island"
    )


def build_cases() -> list[dict]:
    return [
        _select_first_case(iter_fusion_continuity_cases()),
        _select_first_case(iter_fusion_microcase_cases()),
        _select_first_case(iter_fusion_structured_cases()),
    ]


def build_artifact_bundle(cases: list[dict]) -> dict:
    eligibility_passes = sum(case["eligibility_pass"] for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if eligibility_passes == len(cases) else "fail",
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "eligibility_passes": eligibility_passes,
            "continuity_cases": sum(case["case_kind"] == "continuity" for case in cases),
            "microcases": sum(case["case_kind"] == "microcase" for case in cases),
            "structured_cases": sum(
                case["case_kind"] == "structured_family" for case in cases
            ),
            "eligible_unitary_regions": sum(
                case["eligible_unitary_region_count"] for case in cases
            ),
            "singleton_unitary_regions": sum(
                case["singleton_unitary_region_count"] for case in cases
            ),
            "deferred_noise_boundaries": sum(
                case["deferred_noise_boundary_count"] for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Fused eligibility bundle missing required fields: {}".format(
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
        help="Directory to write the fused eligibility bundle into.",
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
                "{case_name}: eligible={eligible_unitary_region_count}, singletons={singleton_unitary_region_count}, deferred={deferred_noise_boundary_count}".format(
                    **case
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
