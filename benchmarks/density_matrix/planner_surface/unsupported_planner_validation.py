#!/usr/bin/env python3
"""Validation: Phase 3 Task 1 Story 5 unsupported planner boundary.

Exercises representative unsupported planner-entry requests and records the
stable no-fallback outcome for each one. The goal is to prove that unsupported
requests fail before execution and are not silently relabeled as supported
`partitioned_density` behavior.

Run with:
    python benchmarks/density_matrix/planner_surface/unsupported_planner_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_surface.common import (
    build_phase3_story1_continuity_vqe,
    build_software_metadata,
)
from benchmarks.density_matrix.planner_surface.workloads import mandatory_microcase_definitions
from squander import Circuit
from squander.partitioning.noisy_planner import (
    PHASE3_ENTRY_ROUTE_MICROCASE,
    PHASE3_WORKLOAD_FAMILY_MICROCASE,
    preflight_planner_request,
)

SUITE_NAME = "phase3_task1_story5_unsupported_planner_boundary"
ARTIFACT_FILENAME = "unsupported_planner_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "planner_surface"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "schema_version",
    "software",
    "summary",
    "cases",
)


def _unsupported_case(case_name: str, runner) -> dict:
    try:
        runner()
    except Exception as err:
        payload = err.to_dict() if hasattr(err, "to_dict") else {"reason": str(err)}
        payload.update(
            {
                "case_name": case_name,
                "status": "unsupported",
                "fallback_used": False,
                "supported_partitioned_case_recorded": False,
            }
        )
        return payload
    return {
        "case_name": case_name,
        "status": "unexpected_pass",
        "fallback_used": False,
        "supported_partitioned_case_recorded": False,
        "reason": "unsupported case unexpectedly passed",
    }


def build_cases() -> list[dict]:
    microcase = mandatory_microcase_definitions()[0]
    continuity_vqe, _, _ = build_phase3_story1_continuity_vqe(4)
    continuity_bridge = continuity_vqe.describe_density_bridge()

    legacy_h = Circuit(1)
    legacy_h.add_H(0)

    return [
        _unsupported_case(
            "unsupported_source_type",
            lambda: preflight_planner_request(
                source_type="binary_import",
                workload_id="unsupported_source_type",
                operation_specs=[],
                qbit_num=2,
                entry_route=PHASE3_ENTRY_ROUTE_MICROCASE,
                workload_family=PHASE3_WORKLOAD_FAMILY_MICROCASE,
            ),
        ),
        _unsupported_case(
            "missing_source_payload",
            lambda: preflight_planner_request(
                source_type="microcase_builder",
                workload_id="missing_source_payload",
                qbit_num=2,
                entry_route=PHASE3_ENTRY_ROUTE_MICROCASE,
                workload_family=PHASE3_WORKLOAD_FAMILY_MICROCASE,
            ),
        ),
        _unsupported_case(
            "unsupported_mode_claim",
            lambda: preflight_planner_request(
                source_type="generated_hea",
                workload_id="unsupported_mode_claim",
                requested_mode="state_vector",
                bridge_metadata=continuity_bridge,
                entry_route="phase2_continuity_lowering",
                workload_family="phase2_continuity_workflow",
            ),
        ),
        _unsupported_case(
            "unsupported_noise_model",
            lambda: preflight_planner_request(
                source_type="microcase_builder",
                workload_id="unsupported_noise_model",
                operation_specs=[
                    *microcase["operation_specs"],
                    {
                        "kind": "noise",
                        "name": "readout_noise",
                        "target_qbit": 0,
                        "source_gate_index": 2,
                        "fixed_value": 0.1,
                        "param_count": 0,
                    },
                ],
                qbit_num=microcase["qbit_num"],
                entry_route=PHASE3_ENTRY_ROUTE_MICROCASE,
                workload_family=PHASE3_WORKLOAD_FAMILY_MICROCASE,
            ),
        ),
        _unsupported_case(
            "legacy_gate_family_h",
            lambda: preflight_planner_request(
                source_type="legacy_qgd_circuit_exact",
                workload_id="legacy_gate_family_h",
                legacy_circuit=legacy_h,
            ),
        ),
        _unsupported_case(
            "invalid_noise_insertion_index",
            lambda: preflight_planner_request(
                source_type="microcase_builder",
                workload_id="invalid_noise_insertion_index",
                operation_specs=[
                    {
                        "kind": "gate",
                        "name": "U3",
                        "target_qbit": 0,
                        "param_count": 3,
                    },
                    {
                        "kind": "noise",
                        "name": "phase_damping",
                        "target_qbit": 0,
                        "source_gate_index": 99,
                        "fixed_value": 0.07,
                        "param_count": 0,
                    },
                ],
                qbit_num=1,
                entry_route=PHASE3_ENTRY_ROUTE_MICROCASE,
                workload_family=PHASE3_WORKLOAD_FAMILY_MICROCASE,
            ),
        ),
    ]


def build_artifact_bundle(cases: list[dict]) -> dict:
    unsupported_cases = sum(case["status"] == "unsupported" for case in cases)
    unexpected_passes = sum(case["status"] == "unexpected_pass" for case in cases)
    fallback_count = sum(bool(case["fallback_used"]) for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if unsupported_cases == len(cases) and unexpected_passes == 0 and fallback_count == 0
        else "fail",
        "schema_version": "phase3_canonical_noisy_planner_v1",
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "unsupported_cases": unsupported_cases,
            "unexpected_passes": unexpected_passes,
            "fallback_count": fallback_count,
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Unsupported planner bundle missing required fields: {}".format(
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
        help="Directory to write the unsupported planner bundle into.",
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
            print("{case_name}: {status}".format(**case))
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
