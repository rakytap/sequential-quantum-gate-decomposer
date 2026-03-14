#!/usr/bin/env python3
"""Validation: Task 4 Story 2 required local-noise exact microcases.

Reuses the canonical Story 2 exactness harness and adds Task 4-specific
traceability for:
- required local-noise coverage,
- mixed required-noise sequence order,
- mandatory-baseline classification that keeps these cases milestone-defining,
- and one stable task-level artifact bundle for the mandatory 1 to 3 qubit
  exact micro-validation matrix.

Run with:
    python benchmarks/density_matrix/task4_story2_required_local_noise_micro_validation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.circuits import STORY2_MANDATORY_MICROCASES
from benchmarks.density_matrix.validate_squander_vs_qiskit import (
    PRIMARY_BACKEND,
    REFERENCE_BACKEND,
    build_software_metadata,
    build_threshold_metadata,
    run_validation as run_story2_validation,
)
from benchmarks.density_matrix.task4_support_tiers import (
    SUPPORT_TIER_VOCABULARY,
    build_required_case_classification,
    build_task4_support_tier_summary,
)

SUITE_NAME = "task4_story2_required_local_noise_micro_validation"
ARTIFACT_FILENAME = "story2_required_local_noise_micro_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "phase2_task4"
)
REQUIRED_LOCAL_NOISE_MODELS = (
    "local_depolarizing",
    "amplitude_damping",
    "phase_damping",
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "backend",
    "reference_backend",
    "requirements",
    "thresholds",
    "software",
    "summary",
    "cases",
)


def build_requirement_metadata():
    individual_noise_cases = [
        case["case_name"]
        for case in STORY2_MANDATORY_MICROCASES
        if case["case_kind"] == "individual_noise"
    ]
    mixed_sequence_cases = [
        case["case_name"]
        for case in STORY2_MANDATORY_MICROCASES
        if case["case_kind"] == "mixed_sequence"
    ]
    return {
        "required_local_noise_models": list(REQUIRED_LOCAL_NOISE_MODELS),
        "required_gate_families": ["U3", "CNOT"],
        "support_tier_vocabulary": list(SUPPORT_TIER_VOCABULARY),
        "mandatory_case_names": [case["case_name"] for case in STORY2_MANDATORY_MICROCASES],
        "individual_noise_case_names": individual_noise_cases,
        "mixed_sequence_case_names": mixed_sequence_cases,
        "microcase_qubits": sorted({case["qbit_num"] for case in STORY2_MANDATORY_MICROCASES}),
        "required_pass_rate": 1.0,
    }


def validate_case_payload(case):
    required_fields = (
        "case_name",
        "status",
        "required_noise_models",
        "case_kind",
        "energy_pass",
        "density_valid_pass",
        "trace_pass",
        "observable_pass",
        "required_gate_coverage_pass",
        "required_noise_model_coverage_pass",
        "noise_sequence_match_pass",
        "operation_audit_pass",
        "noise_operation_sequence",
        "task4_story2_case_pass",
        "support_tier",
        "case_purpose",
        "counts_toward_mandatory_baseline",
    )
    missing_fields = [field for field in required_fields if field not in case]
    if missing_fields:
        raise ValueError(
            "Task 4 Story 2 case payload is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def build_artifact_bundle(results):
    cases = []
    for result in results:
        case = dict(result)
        case.update(build_required_case_classification())
        case["task4_story2_case_pass"] = bool(
            case["status"] == "pass"
            and case["energy_pass"]
            and case["density_valid_pass"]
            and case["trace_pass"]
            and case["observable_pass"]
            and case["required_gate_coverage_pass"]
            and case["required_noise_model_coverage_pass"]
            and case["noise_sequence_match_pass"]
            and case["operation_audit_pass"]
        )
        validate_case_payload(case)
        cases.append(case)

    total_cases = len(cases)
    passed_cases = sum(case["task4_story2_case_pass"] for case in cases)
    exact_threshold_passed_cases = sum(
        case["energy_pass"]
        and case["density_valid_pass"]
        and case["trace_pass"]
        and case["observable_pass"]
        for case in cases
    )
    operation_audit_passed_cases = sum(case["operation_audit_pass"] for case in cases)
    mixed_sequence_case_count = sum(case["case_kind"] == "mixed_sequence" for case in cases)
    mixed_sequence_passed_cases = sum(
        case["case_kind"] == "mixed_sequence" and case["mixed_sequence_order_pass"]
        for case in cases
    )
    required_noise_models_covered = sorted(
        {
            noise_model
            for case in cases
            for noise_model in case["required_noise_models"]
        }
    )
    pass_rate = 0.0 if total_cases == 0 else passed_cases / total_cases
    support_tier_summary = build_task4_support_tier_summary(cases)

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if support_tier_summary["mandatory_baseline_completed"]
        and passed_cases == total_cases
        and total_cases
        else "fail",
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "requirements": build_requirement_metadata(),
        "thresholds": build_threshold_metadata(),
        "software": build_software_metadata(),
        "summary": {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": total_cases - passed_cases,
            "pass_rate": pass_rate,
            **support_tier_summary,
            "exact_threshold_passed_cases": exact_threshold_passed_cases,
            "operation_audit_passed_cases": operation_audit_passed_cases,
            "mixed_sequence_case_count": mixed_sequence_case_count,
            "mixed_sequence_passed_cases": mixed_sequence_passed_cases,
            "required_noise_models_covered": required_noise_models_covered,
        },
        "cases": cases,
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 4 Story 2 artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def write_artifact_bundle(output_path: Path, bundle):
    validate_artifact_bundle(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 4 Story 2 JSON artifact bundle.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case validation output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_story2_validation(verbose=not args.quiet)
    bundle = build_artifact_bundle(results)
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} ({}/{})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["passed_cases"],
            bundle["summary"]["total_cases"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
