#!/usr/bin/env python3
"""Validation: exact-regime workflow baseline.

Builds the phase-level workflow-scale gate from the canonical exact-regime
workflow matrix while keeping the dedicated optimization trace out of scope
until the trace-anchor layer.

The resulting bundle is intentionally a thin validation-evidence layer:
- it freezes the mandatory 4 / 6 / 8 / 10 workflow inventory,
- it validates stable workflow case identity and parameter-set coverage,
- it preserves exactness, backend attribution, and unsupported-free completion,
- and it makes incomplete or partial workflow evidence fail explicitly.

Run with:
    python benchmarks/density_matrix/validation_evidence/workflow_baseline_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import (
    PRIMARY_BACKEND,
    REFERENCE_BACKEND,
    EXACT_REGIME_PARAMETER_SET_COUNT,
    EXACT_REGIME_WORKFLOW_QUBITS,
    build_software_metadata,
    build_exact_regime_threshold_metadata,
    build_exact_regime_workflow_bundle,
    run_exact_regime_workflow_matrix,
)
from benchmarks.density_matrix.noise_support.required_local_noise_micro_validation import (
    REQUIRED_LOCAL_NOISE_MODELS,
)
from benchmarks.density_matrix.noise_support.support_tiers import (
    SUPPORT_TIER_VOCABULARY,
    build_support_tier_summary,
)

SUITE_NAME = "workflow_baseline_validation"
ARTIFACT_FILENAME = "workflow_baseline_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "validation_evidence"
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
    "required_artifacts",
    "cases",
)


def build_requirement_metadata(
    *, qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS, parameter_set_count=EXACT_REGIME_PARAMETER_SET_COUNT
):
    parameter_set_ids = [f"set_{idx:02d}" for idx in range(parameter_set_count)]
    mandatory_case_names = [
        f"exact_regime_{qbit_num}q_{parameter_set_id}"
        for qbit_num in qubit_sizes
        for parameter_set_id in parameter_set_ids
    ]
    return {
        "mandatory_workflow_qubits": list(qubit_sizes),
        "fixed_parameter_sets_per_size": parameter_set_count,
        "mandatory_parameter_set_ids": parameter_set_ids,
        "mandatory_case_names": mandatory_case_names,
        "required_gate_families": ["U3", "CNOT"],
        "required_local_noise_models": list(REQUIRED_LOCAL_NOISE_MODELS),
        "support_tier_vocabulary": list(SUPPORT_TIER_VOCABULARY),
        "required_bundle_sources": ["exact_regime_workflow"],
        "required_pass_rate": 1.0,
    }


def validate_case_payload(case):
    required_fields = (
        "case_name",
        "status",
        "backend",
        "reference_backend",
        "qbit_num",
        "workflow_completed",
        "parameter_set_id",
        "absolute_energy_error",
        "energy_pass",
        "density_valid_pass",
        "trace_pass",
        "observable_pass",
        "bridge_supported_pass",
        "support_tier",
        "case_purpose",
        "counts_toward_mandatory_baseline",
    )
    missing_fields = [field for field in required_fields if field not in case]
    if missing_fields:
        raise ValueError(
            "Workflow-baseline case is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def build_case_identity_summary(
    cases,
    *,
    qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS,
    parameter_set_count: int = EXACT_REGIME_PARAMETER_SET_COUNT,
):
    observed_case_names = [case["case_name"] for case in cases]
    case_name_counts = Counter(observed_case_names)
    duplicate_case_names = sorted(
        case_name for case_name, count in case_name_counts.items() if count > 1
    )

    expected_parameter_set_ids = [f"set_{idx:02d}" for idx in range(parameter_set_count)]
    expected_case_names = {
        f"exact_regime_{qbit_num}q_{parameter_set_id}"
        for qbit_num in qubit_sizes
        for parameter_set_id in expected_parameter_set_ids
    }
    observed_case_name_set = set(observed_case_names)
    missing_mandatory_case_names = sorted(expected_case_names - observed_case_name_set)
    unexpected_case_names = sorted(observed_case_name_set - expected_case_names)

    parameter_sets_per_qbit = defaultdict(list)
    for case in cases:
        parameter_sets_per_qbit[case["qbit_num"]].append(case["parameter_set_id"])

    missing_parameter_set_ids_by_qbit = {}
    duplicate_parameter_set_ids_by_qbit = {}
    cases_per_qbit = {}
    for qbit_num in qubit_sizes:
        observed_ids = parameter_sets_per_qbit.get(qbit_num, [])
        counts = Counter(observed_ids)
        duplicate_ids = sorted(
            parameter_set_id
            for parameter_set_id, count in counts.items()
            if count > 1
        )
        missing_ids = sorted(set(expected_parameter_set_ids) - set(observed_ids))
        duplicate_parameter_set_ids_by_qbit[str(qbit_num)] = duplicate_ids
        missing_parameter_set_ids_by_qbit[str(qbit_num)] = missing_ids
        cases_per_qbit[str(qbit_num)] = len(observed_ids)

    stable_case_ids_present = (
        not missing_mandatory_case_names
        and not duplicate_case_names
        and not unexpected_case_names
        and len(observed_case_names) == len(expected_case_names)
    )
    stable_parameter_set_ids_present = all(
        not missing_parameter_set_ids_by_qbit[str(qbit_num)]
        and not duplicate_parameter_set_ids_by_qbit[str(qbit_num)]
        for qbit_num in qubit_sizes
    )

    return {
        "observed_case_names": observed_case_names,
        "missing_mandatory_case_names": missing_mandatory_case_names,
        "duplicate_case_names": duplicate_case_names,
        "unexpected_case_names": unexpected_case_names,
        "stable_case_ids_present": stable_case_ids_present,
        "cases_per_qbit": cases_per_qbit,
        "missing_parameter_set_ids_by_qbit": missing_parameter_set_ids_by_qbit,
        "duplicate_parameter_set_ids_by_qbit": duplicate_parameter_set_ids_by_qbit,
        "stable_parameter_set_ids_present": stable_parameter_set_ids_present,
    }


def build_artifact_bundle(
    exact_regime_workflow_bundle,
    workflow_results,
    *,
    qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS,
    parameter_set_count: int = EXACT_REGIME_PARAMETER_SET_COUNT,
):
    cases = [dict(case) for case in workflow_results]
    for case in cases:
        validate_case_payload(case)

    support_tier_summary = build_support_tier_summary(cases)
    case_identity = build_case_identity_summary(
        cases, qubit_sizes=qubit_sizes, parameter_set_count=parameter_set_count
    )
    total_cases = len(cases)
    passed_cases = sum(case["status"] == "pass" for case in cases)
    unsupported_cases = sum(case["status"] == "unsupported" for case in cases)
    required_pass_rate = support_tier_summary["required_pass_rate"]
    bridge_supported_cases = sum(
        case.get("bridge_supported_pass", False) for case in cases
    )
    documented_10q_anchor_required = 10 in qubit_sizes
    documented_10q_anchor_present = any(case["qbit_num"] == 10 for case in cases)
    all_cases_required = all(case["support_tier"] == "required" for case in cases)
    all_cases_mandatory = all(
        case["counts_toward_mandatory_baseline"] for case in cases
    )
    workflow_baseline_completed = bool(
        exact_regime_workflow_bundle["status"] == "pass"
        and case_identity["stable_case_ids_present"]
        and case_identity["stable_parameter_set_ids_present"]
        and all_cases_required
        and all_cases_mandatory
        and support_tier_summary["mandatory_baseline_completed"]
        and unsupported_cases == 0
        and bridge_supported_cases == total_cases
        and (
            not documented_10q_anchor_required or documented_10q_anchor_present
        )
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if workflow_baseline_completed else "fail",
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "requirements": build_requirement_metadata(
            qubit_sizes=qubit_sizes, parameter_set_count=parameter_set_count
        ),
        "thresholds": build_exact_regime_threshold_metadata(
            qubit_sizes=qubit_sizes, parameter_set_count=parameter_set_count
        ),
        "software": build_software_metadata(),
        "summary": {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": total_cases - passed_cases,
            "unsupported_cases": unsupported_cases,
            "unsupported_status_cases": unsupported_cases,
            "bridge_supported_cases": bridge_supported_cases,
            "required_cases": support_tier_summary["required_cases"],
            "required_passed_cases": support_tier_summary["required_passed_cases"],
            "required_pass_rate": required_pass_rate,
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
            "required_workflow_qubits": list(qubit_sizes),
            "fixed_parameter_sets_per_size": parameter_set_count,
            "documented_10q_anchor_required": documented_10q_anchor_required,
            "documented_10q_anchor_present": documented_10q_anchor_present,
            **case_identity,
            "all_cases_required": all_cases_required,
            "all_cases_count_toward_mandatory_baseline": all_cases_mandatory,
            "workflow_baseline_completed": workflow_baseline_completed,
        },
        "required_artifacts": {
            "exact_regime_workflow_reference": {
                "suite_name": exact_regime_workflow_bundle["suite_name"],
                "status": exact_regime_workflow_bundle["status"],
                "summary": exact_regime_workflow_bundle["summary"],
            }
        },
        "cases": cases,
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Workflow-baseline bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def write_artifact_bundle(output_path: Path, bundle):
    validate_artifact_bundle(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_validation(
    *,
    qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS,
    parameter_set_count: int = EXACT_REGIME_PARAMETER_SET_COUNT,
    verbose=False,
):
    workflow_results = run_exact_regime_workflow_matrix(
        qubit_sizes=qubit_sizes, parameter_set_count=parameter_set_count
    )
    exact_regime_workflow_bundle = build_exact_regime_workflow_bundle(
        workflow_results,
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
        trace_result=None,
    )
    bundle = build_artifact_bundle(
        exact_regime_workflow_bundle,
        workflow_results,
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
    )
    if verbose:
        print(
            "{} [{}] required cases={}/{} stable_case_ids={} stable_parameter_sets={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["required_passed_cases"],
                bundle["summary"]["required_cases"],
                bundle["summary"]["stable_case_ids_present"],
                bundle["summary"]["stable_parameter_set_ids_present"],
            )
        )
    return exact_regime_workflow_bundle, bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the workflow-baseline JSON artifact bundle.",
    )
    parser.add_argument(
        "--parameter-set-count",
        type=int,
        default=EXACT_REGIME_PARAMETER_SET_COUNT,
        help="Number of fixed parameter vectors per qubit size.",
    )
    parser.add_argument(
        "--qubit-sizes",
        type=int,
        nargs="*",
        default=list(EXACT_REGIME_WORKFLOW_QUBITS),
        help="Workflow qubit sizes to include.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, bundle = run_validation(
        qubit_sizes=tuple(args.qubit_sizes),
        parameter_set_count=args.parameter_set_count,
        verbose=not args.quiet,
    )
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
