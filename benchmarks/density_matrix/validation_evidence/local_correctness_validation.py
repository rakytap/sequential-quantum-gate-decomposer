#!/usr/bin/env python3
"""Validation: Task 5 Story 1 local correctness baseline.

Builds the phase-level local correctness gate from the already authoritative
micro-validation surfaces:
- the canonical Story 2 exactness matrix against Qiskit Aer,
- and the Task 4 Story 2 required-local-noise wrapper that adds required-noise
  coverage, mixed-sequence auditability, and mandatory-baseline semantics.

The resulting bundle is intentionally a thin Task 5 layer:
- it freezes the mandatory 1 to 3 qubit local correctness inventory,
- it validates stable case identity and explicit status fields,
- it preserves required gate/noise coverage and mixed-sequence auditability,
- and it makes incomplete or partial local evidence fail explicitly.

Run with:
    python benchmarks/density_matrix/validation_evidence/local_correctness_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.circuits import MANDATORY_MICROCASES
from benchmarks.density_matrix.noise_support.required_local_noise_micro_validation import (
    REQUIRED_LOCAL_NOISE_MODELS,
    build_artifact_bundle as build_required_local_noise_micro_validation_bundle,
)
from benchmarks.density_matrix.noise_support.support_tiers import (
    SUPPORT_TIER_VOCABULARY,
    build_support_tier_summary,
)
from benchmarks.density_matrix.validate_squander_vs_qiskit import (
    PRIMARY_BACKEND,
    REFERENCE_BACKEND,
    build_artifact_bundle as build_story2_bundle,
    build_software_metadata,
    build_threshold_metadata,
    run_validation as run_story2_validation,
)

SUITE_NAME = "local_correctness_validation"
ARTIFACT_FILENAME = "local_correctness_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "validation_evidence"
)
MANDATORY_CASE_NAMES = tuple(case["case_name"] for case in MANDATORY_MICROCASES)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "backend",
    "reference_backend",
    "requirements",
    "thresholds",
    "software",
    "summary",
    "required_artifacts",
    "cases",
)


def build_requirement_metadata():
    return {
        "mandatory_case_names": list(MANDATORY_CASE_NAMES),
        "required_gate_families": ["U3", "CNOT"],
        "required_local_noise_models": list(REQUIRED_LOCAL_NOISE_MODELS),
        "support_tier_vocabulary": list(SUPPORT_TIER_VOCABULARY),
        "required_bundle_sources": [
            "story2_mandatory_micro_validation",
            "required_local_noise_micro_validation",
        ],
        "required_pass_rate": 1.0,
    }


def validate_case_payload(case):
    required_fields = (
        "case_name",
        "status",
        "case_kind",
        "energy_pass",
        "density_valid_pass",
        "trace_pass",
        "observable_pass",
        "required_gate_coverage_pass",
        "required_noise_model_coverage_pass",
        "noise_sequence_match_pass",
        "operation_audit_pass",
        "support_tier",
        "case_purpose",
        "counts_toward_mandatory_baseline",
        "required_local_noise_microcase_pass",
    )
    missing_fields = [field for field in required_fields if field not in case]
    if missing_fields:
        raise ValueError(
            "Task 5 Story 1 case payload is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def build_case_identity_summary(cases):
    case_names = [case["case_name"] for case in cases]
    counts = Counter(case_names)
    duplicate_case_names = sorted(
        case_name for case_name, count in counts.items() if count > 1
    )
    missing_mandatory_case_names = sorted(set(MANDATORY_CASE_NAMES) - set(case_names))
    unexpected_case_names = sorted(set(case_names) - set(MANDATORY_CASE_NAMES))
    stable_case_ids_present = (
        not missing_mandatory_case_names
        and not duplicate_case_names
        and not unexpected_case_names
        and len(case_names) == len(MANDATORY_CASE_NAMES)
    )
    return {
        "observed_case_names": case_names,
        "missing_mandatory_case_names": missing_mandatory_case_names,
        "duplicate_case_names": duplicate_case_names,
        "unexpected_case_names": unexpected_case_names,
        "stable_case_ids_present": stable_case_ids_present,
    }


def build_artifact_bundle(story2_bundle, required_local_noise_micro_validation_bundle):
    cases = [dict(case) for case in required_local_noise_micro_validation_bundle["cases"]]
    for case in cases:
        validate_case_payload(case)

    case_identity = build_case_identity_summary(cases)
    support_tier_summary = build_support_tier_summary(cases)
    total_cases = len(cases)
    passed_cases = sum(case["status"] == "pass" for case in cases)
    pass_rate = (passed_cases / total_cases) if total_cases else 0.0
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
    all_cases_required = all(case["support_tier"] == "required" for case in cases)
    all_cases_mandatory = all(
        case["counts_toward_mandatory_baseline"] for case in cases
    )
    local_correctness_gate_completed = bool(
        story2_bundle["status"] == "pass"
        and required_local_noise_micro_validation_bundle["status"] == "pass"
        and case_identity["stable_case_ids_present"]
        and all_cases_required
        and all_cases_mandatory
        and support_tier_summary["mandatory_baseline_completed"]
        and exact_threshold_passed_cases == total_cases
        and operation_audit_passed_cases == total_cases
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if local_correctness_gate_completed else "fail",
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
            "required_cases": support_tier_summary["required_cases"],
            "required_passed_cases": support_tier_summary["required_passed_cases"],
            "required_pass_rate": support_tier_summary["required_pass_rate"],
            "mandatory_baseline_case_count": support_tier_summary[
                "mandatory_baseline_case_count"
            ],
            "mandatory_baseline_passed_cases": support_tier_summary[
                "mandatory_baseline_passed_cases"
            ],
            "mandatory_baseline_completed": support_tier_summary[
                "mandatory_baseline_completed"
            ],
            "optional_cases_count_toward_mandatory_baseline": support_tier_summary[
                "optional_cases_count_toward_mandatory_baseline"
            ],
            "exact_threshold_passed_cases": exact_threshold_passed_cases,
            "operation_audit_passed_cases": operation_audit_passed_cases,
            "mixed_sequence_case_count": mixed_sequence_case_count,
            "mixed_sequence_passed_cases": mixed_sequence_passed_cases,
            "required_noise_models_covered": required_noise_models_covered,
            "support_tiers_present": support_tier_summary["support_tiers_present"],
            **case_identity,
            "all_cases_required": all_cases_required,
            "all_cases_count_toward_mandatory_baseline": all_cases_mandatory,
            "local_correctness_gate_completed": local_correctness_gate_completed,
        },
        "required_artifacts": {
            "micro_validation_reference": {
                "suite_name": story2_bundle["suite_name"],
                "status": story2_bundle["status"],
                "summary": story2_bundle["summary"],
            },
            "required_local_noise_micro_validation": {
                "suite_name": required_local_noise_micro_validation_bundle["suite_name"],
                "status": required_local_noise_micro_validation_bundle["status"],
                "summary": required_local_noise_micro_validation_bundle["summary"],
            },
        },
        "cases": cases,
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 5 Story 1 artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def write_artifact_bundle(output_path: Path, bundle):
    validate_artifact_bundle(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_validation(*, verbose=False):
    results = run_story2_validation(verbose=verbose)
    story2_bundle = build_story2_bundle(results)
    required_local_noise_micro_validation_bundle = build_required_local_noise_micro_validation_bundle(results)
    bundle = build_artifact_bundle(story2_bundle, required_local_noise_micro_validation_bundle)
    if verbose:
        print(
            "{} [{}] mandatory cases={}/{} stable_case_ids={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["required_passed_cases"],
                bundle["summary"]["required_cases"],
                bundle["summary"]["stable_case_ids_present"],
            )
        )
    return story2_bundle, required_local_noise_micro_validation_bundle, bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 5 Story 1 JSON artifact bundle.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case validation output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, bundle = run_validation(verbose=not args.quiet)
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} ({}/{})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["required_passed_cases"],
            bundle["summary"]["required_cases"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
