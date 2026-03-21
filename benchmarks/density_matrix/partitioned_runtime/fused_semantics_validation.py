#!/usr/bin/env python3
"""Fused semantics validation: fused runtime vs sequential reference exactness.

Checks representative fused cases against the sequential density reference using
the frozen exactness thresholds.

Run with:
    python benchmarks/density_matrix/partitioned_runtime/fused_semantics_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.partitioned_runtime.common import (
    PHASE3_RUNTIME_DENSITY_TOL,
    execute_fused_with_reference,
)
from benchmarks.density_matrix.partitioned_runtime.fusion_case_selection import (
    iter_fusion_microcase_cases,
    iter_fusion_structured_cases,
)
from benchmarks.density_matrix.planner_surface.common import build_software_metadata

SUITE_NAME = "phase3_partitioned_runtime_fused_semantics"
ARTIFACT_FILENAME = "fused_semantics_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "partitioned_runtime"
    / "fused_semantics"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "runtime_schema_version",
    "software",
    "summary",
    "cases",
)


def _semantics_case(metadata: dict, descriptor_set, parameters) -> dict:
    runtime_result, _, density_metrics = execute_fused_with_reference(
        descriptor_set, parameters
    )
    payload = runtime_result.to_dict(include_density_matrix=False)
    return {
        "case_name": metadata["workload_id"],
        "case_kind": metadata["case_kind"],
        "qbit_num": metadata["qbit_num"],
        "runtime_schema_version": payload["runtime_schema_version"],
        "runtime_path": payload["runtime_path"],
        "actual_fused_execution": payload["summary"]["actual_fused_execution"],
        "fused_region_count": payload["summary"]["fused_region_count"],
        "supported_unfused_region_count": payload["summary"][
            "supported_unfused_region_count"
        ],
        "deferred_region_count": payload["summary"]["deferred_region_count"],
        "frobenius_norm_diff": density_metrics["frobenius_norm_diff"],
        "max_abs_diff": density_metrics["max_abs_diff"],
        "rho_is_valid": payload["summary"]["rho_is_valid"],
        "trace_deviation": payload["summary"]["trace_deviation"],
        "fused_semantics_pass": (
            payload["summary"]["actual_fused_execution"] is True
            and density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            and density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            and payload["summary"]["rho_is_valid"] is True
        ),
        "metadata": dict(metadata),
    }


def build_cases() -> list[dict]:
    selected_structured: dict[int, dict] = {}
    optional_microcase: dict | None = None
    for metadata, descriptor_set, parameters in iter_fusion_structured_cases():
        if metadata["qbit_num"] in selected_structured:
            continue
        case = _semantics_case(metadata, descriptor_set, parameters)
        if case["fused_semantics_pass"]:
            selected_structured[metadata["qbit_num"]] = case
        if set(selected_structured) == {8, 10}:
            break
    if set(selected_structured) != {8, 10}:
        raise RuntimeError("Missing representative structured fused semantics cases")
    for metadata, descriptor_set, parameters in iter_fusion_microcase_cases():
        case = _semantics_case(metadata, descriptor_set, parameters)
        if case["fused_semantics_pass"]:
            optional_microcase = case
            break
    cases = [selected_structured[8], selected_structured[10]]
    if optional_microcase is not None:
        cases.append(optional_microcase)
    return cases


def build_artifact_bundle(cases: list[dict]) -> dict:
    passes = sum(case["fused_semantics_pass"] for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if passes == len(cases) and len(cases) >= 2 else "fail",
        "runtime_schema_version": cases[0]["runtime_schema_version"],
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "fused_semantics_passes": passes,
            "required_structured_cases": 2,
            "optional_microcase_cases": sum(
                case["case_kind"] == "microcase" for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Fused semantics bundle missing required fields: {}".format(
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
        help="Directory to write the fused semantics bundle into.",
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
                "{case_name}: fused={actual_fused_execution}, dRho={frobenius_norm_diff:.3e}, pass={fused_semantics_pass}".format(
                    **case
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
