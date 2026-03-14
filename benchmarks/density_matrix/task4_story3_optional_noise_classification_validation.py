#!/usr/bin/env python3
"""Validation: Task 4 Story 3 optional-noise classification layer.

Builds on the already passing required Task 4 bundles and adds explicit optional
whole-register depolarizing cases so reviewers can see that:
- required local-noise evidence remains milestone-defining,
- optional evidence is clearly marked optional,
- and optional cases do not count toward mandatory Task 4 completion.

Run with:
    python benchmarks/density_matrix/task4_story3_optional_noise_classification_validation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.circuits import DualBuilder
from benchmarks.density_matrix.task4_story1_required_local_noise_validation import (
    REQUIRED_LOCAL_NOISE_MODELS,
    build_artifact_bundle as build_story1_bundle,
    run_validation as run_story1_validation,
)
from benchmarks.density_matrix.task4_story2_required_local_noise_micro_validation import (
    build_artifact_bundle as build_story2_bundle,
    run_story2_validation,
)
from benchmarks.density_matrix.task4_support_tiers import (
    CASE_PURPOSE_OPTIONAL_REGRESSION,
    CASE_PURPOSE_OPTIONAL_STRESS,
    OPTIONAL_REASON_WHOLE_REGISTER_DEPOLARIZING,
    SUPPORT_TIER_VOCABULARY,
    build_optional_case_classification,
    build_task4_support_tier_summary,
)
from benchmarks.density_matrix.validate_squander_vs_qiskit import (
    PRIMARY_BACKEND,
    REFERENCE_BACKEND,
    build_software_metadata,
    build_threshold_metadata,
    validate_story2_case,
)

SUITE_NAME = "task4_story3_optional_noise_classification"
ARTIFACT_FILENAME = "story3_optional_noise_classification_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "phase2_task4"
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
PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_string_matrix(pauli_string):
    matrix = np.array([[1.0 + 0.0j]])
    for symbol in pauli_string:
        matrix = np.kron(matrix, PAULI_MATRICES[symbol])
    return matrix


def _build_hamiltonian(term_specs):
    qbit_num = len(term_specs[0][1])
    matrix = np.zeros((2**qbit_num, 2**qbit_num), dtype=complex)
    for coeff, pauli_string in term_specs:
        matrix += coeff * _pauli_string_matrix(pauli_string)
    return matrix


def _build_hamiltonian_metadata(term_specs):
    return {
        "terms": [
            {"coeff": float(coeff), "pauli_string": pauli_string}
            for coeff, pauli_string in term_specs
        ]
    }


def build_optional_1q_u3_whole_register_depolarizing():
    builder = DualBuilder(1)
    builder.U3(0, 0.61, -0.24, 0.39)
    builder.depolarizing(0.06)
    return builder


def build_optional_2q_u3_cnot_whole_register_depolarizing():
    builder = DualBuilder(2)
    builder.U3(0, 0.54, -0.19, 0.42)
    builder.U3(1, 0.38, 0.27, -0.31)
    builder.CNOT(1, 0)
    builder.depolarizing(0.04)
    return builder


def _make_optional_case(
    *,
    case_name,
    qbit_num,
    builder_fn,
    required_gate_family,
    required_noise_models,
    case_kind,
    purpose,
    term_specs,
):
    return {
        "case_name": case_name,
        "qbit_num": qbit_num,
        "builder_fn": builder_fn,
        "required_gate_family": list(required_gate_family),
        "required_noise_models": list(required_noise_models),
        "case_kind": case_kind,
        "purpose": purpose,
        "hamiltonian_matrix": _build_hamiltonian(term_specs),
        "hamiltonian_metadata": _build_hamiltonian_metadata(term_specs),
    }


OPTIONAL_WHOLE_REGISTER_CASES = (
    _make_optional_case(
        case_name="task4_story3_1q_optional_whole_register_depolarizing_regression",
        qbit_num=1,
        builder_fn=build_optional_1q_u3_whole_register_depolarizing,
        required_gate_family=["U3"],
        required_noise_models=["depolarizing"],
        case_kind=CASE_PURPOSE_OPTIONAL_REGRESSION,
        purpose="Optional regression baseline using whole-register depolarizing on a 1-qubit U3 case.",
        term_specs=[(0.32, "Z"), (-0.18, "X")],
    ),
    _make_optional_case(
        case_name="task4_story3_2q_optional_whole_register_depolarizing_stress",
        qbit_num=2,
        builder_fn=build_optional_2q_u3_cnot_whole_register_depolarizing,
        required_gate_family=["U3", "CNOT"],
        required_noise_models=["depolarizing"],
        case_kind=CASE_PURPOSE_OPTIONAL_STRESS,
        purpose="Optional stress baseline using whole-register depolarizing on a 2-qubit U3/CNOT case.",
        term_specs=[(0.41, "ZI"), (-0.23, "IZ"), (0.17, "XX")],
    ),
)


def build_requirement_metadata():
    return {
        "support_tier_vocabulary": list(SUPPORT_TIER_VOCABULARY),
        "required_local_noise_models": list(REQUIRED_LOCAL_NOISE_MODELS),
        "optional_noise_models": ["depolarizing"],
        "optional_case_names": [case["case_name"] for case in OPTIONAL_WHOLE_REGISTER_CASES],
        "required_bundle_sources": [
            "task4_story1_required_local_noise",
            "task4_story2_required_local_noise_micro_validation",
        ],
        "required_pass_rate": 1.0,
    }


def validate_optional_case_payload(case):
    required_fields = (
        "case_name",
        "status",
        "support_tier",
        "case_purpose",
        "optional_reason",
        "counts_toward_mandatory_baseline",
        "required_noise_models",
        "noise_operation_sequence",
        "whole_register_baseline_classification_pass",
        "task4_story3_case_pass",
    )
    missing_fields = [field for field in required_fields if field not in case]
    if missing_fields:
        raise ValueError(
            "Task 4 Story 3 optional case payload is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def run_optional_case(case, *, verbose=False):
    result = dict(validate_story2_case(case, verbose=verbose))
    result.update(
        build_optional_case_classification(
            case_purpose=case["case_kind"],
            optional_reason=OPTIONAL_REASON_WHOLE_REGISTER_DEPOLARIZING,
        )
    )
    result["whole_register_baseline_classification_pass"] = bool(
        result["noise_operation_sequence"] == ["depolarizing"]
        and result["required_noise_model_coverage_pass"]
        and not result["counts_toward_mandatory_baseline"]
    )
    result["task4_story3_case_pass"] = bool(
        result["status"] == "pass"
        and result["whole_register_baseline_classification_pass"]
        and result["support_tier"] == "optional"
        and result["case_purpose"] in {
            CASE_PURPOSE_OPTIONAL_REGRESSION,
            CASE_PURPOSE_OPTIONAL_STRESS,
        }
    )
    validate_optional_case_payload(result)
    return result


def run_validation(*, verbose=False):
    story1_results = run_story1_validation(verbose=verbose)
    story1_bundle = build_story1_bundle(story1_results)
    story2_results = run_story2_validation(verbose=verbose)
    story2_bundle = build_story2_bundle(story2_results)
    optional_results = [run_optional_case(case, verbose=verbose) for case in OPTIONAL_WHOLE_REGISTER_CASES]
    return story1_bundle, story2_bundle, optional_results


def build_artifact_bundle(story1_bundle, story2_bundle, optional_results):
    optional_results = list(optional_results)
    required_cases = list(story1_bundle["cases"]) + list(story2_bundle["cases"])
    required_summary = build_task4_support_tier_summary(required_cases)
    optional_summary = build_task4_support_tier_summary(optional_results)

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if required_summary["mandatory_baseline_completed"]
        and optional_summary["optional_cases_count_toward_mandatory_baseline"] == 0
        and all(case["task4_story3_case_pass"] for case in optional_results)
        else "fail",
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "requirements": build_requirement_metadata(),
        "thresholds": build_threshold_metadata(),
        "software": build_software_metadata(),
        "summary": {
            "required_cases": required_summary["required_cases"],
            "required_passed_cases": required_summary["required_passed_cases"],
            "required_pass_rate": required_summary["required_pass_rate"],
            "optional_cases": optional_summary["optional_cases"],
            "optional_passed_cases": optional_summary["optional_passed_cases"],
            "optional_pass_rate": optional_summary["optional_pass_rate"],
            "optional_cases_count_toward_mandatory_baseline": optional_summary[
                "optional_cases_count_toward_mandatory_baseline"
            ],
            "mandatory_baseline_case_count": required_summary[
                "mandatory_baseline_case_count"
            ],
            "mandatory_baseline_passed_cases": required_summary[
                "mandatory_baseline_passed_cases"
            ],
            "mandatory_baseline_completed": required_summary[
                "mandatory_baseline_completed"
            ],
            "support_tiers_present": sorted(
                set(
                    required_summary["support_tiers_present"]
                    + optional_summary["support_tiers_present"]
                )
            ),
        },
        "required_artifacts": {
            "story1": {
                "suite_name": story1_bundle["suite_name"],
                "status": story1_bundle["status"],
                "summary": story1_bundle["summary"],
            },
            "story2": {
                "suite_name": story2_bundle["suite_name"],
                "status": story2_bundle["status"],
                "summary": story2_bundle["summary"],
            },
        },
        "cases": optional_results,
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 4 Story 3 artifact bundle is missing required fields: {}".format(
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
        help="Directory for the Task 4 Story 3 JSON artifact bundle.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case validation output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    story1_bundle, story2_bundle, optional_results = run_validation(
        verbose=not args.quiet
    )
    bundle = build_artifact_bundle(story1_bundle, story2_bundle, optional_results)
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} (required baseline completed: {}; optional cases: {}/{})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["mandatory_baseline_completed"],
            bundle["summary"]["optional_passed_cases"],
            bundle["summary"]["optional_cases"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
