#!/usr/bin/env python3
"""Validation: unsupported and deferred noise boundary.

Captures representative negative cases as structured artifacts. The
focus is the frozen Phase 2 noise support matrix:
- deferred families fail before execution,
- invalid ordered-noise schedule metadata fails deterministically,
- the first unsupported condition is recorded in a machine-reviewable way,
- and every emitted case stays outside the mandatory baseline.

Run with:
    python benchmarks/density_matrix/noise_support/unsupported_noise_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.VQE.shot_noise_measurement import generate_zz_xx_hamiltonian
from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import (
    DEFAULT_ANSATZ,
    DEFAULT_INNER_BLOCKS,
    DEFAULT_LAYERS,
    PRIMARY_BACKEND,
    build_case_metadata,
    build_open_chain_topology,
    build_software_metadata,
    build_optimizer_config,
    build_hamiltonian_metadata,
    build_vqe,
)
from benchmarks.density_matrix.noise_support.support_tiers import (
    BOUNDARY_CLASS_CONFIGURATION,
    BOUNDARY_CLASS_MODEL_FAMILY,
    BOUNDARY_CLASS_SCHEDULE_ELEMENT,
    DEFERRED_REASON_CALIBRATION_AWARE,
    DEFERRED_REASON_CORRELATED_MULTI_QUBIT,
    DEFERRED_REASON_NON_MARKOVIAN,
    DEFERRED_REASON_READOUT,
    SUPPORT_TIER_DEFERRED,
    SUPPORT_TIER_UNSUPPORTED,
    SUPPORT_TIER_VOCABULARY,
    DEFERRED_FAMILY_REASONS,
    classify_noise_boundary_reason,
    build_support_tier_summary,
)
from squander import Variational_Quantum_Eigensolver

SUITE_NAME = "unsupported_noise_boundary"
ARTIFACT_FILENAME = "unsupported_noise_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "noise_support"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "backend",
    "requirements",
    "summary",
    "software",
    "cases",
)

UNSUPPORTED_NOISE_BOUNDARY_CASES = (
    {
        "case_name": "deferred_readout_noise",
        "qbit_num": 1,
        "density_noise": [
            {
                "channel": "readout_noise",
                "target": 0,
                "after_gate_index": 0,
                "value": 0.1,
            }
        ],
        "purpose": "Verify readout-noise requests fail before execution on the VQE-facing density path.",
        "expected_error_fragment": "Unsupported density-noise channel 'readout_noise'",
        "expected_support_tier": SUPPORT_TIER_DEFERRED,
        "expected_unsupported_category": "noise_type",
        "expected_first_unsupported_condition": DEFERRED_REASON_READOUT,
        "expected_noise_boundary_class": BOUNDARY_CLASS_MODEL_FAMILY,
        "expected_failure_stage": "python_normalization",
        "runner_kind": "constructor",
    },
    {
        "case_name": "deferred_correlated_multi_qubit_noise",
        "qbit_num": 2,
        "density_noise": [
            {
                "channel": "correlated_multi_qubit_noise",
                "target": 0,
                "after_gate_index": 0,
                "value": 0.1,
            }
        ],
        "purpose": "Verify correlated multi-qubit noise requests remain deferred and fail before execution.",
        "expected_error_fragment": "Unsupported density-noise channel 'correlated_multi_qubit_noise'",
        "expected_support_tier": SUPPORT_TIER_DEFERRED,
        "expected_unsupported_category": "noise_type",
        "expected_first_unsupported_condition": (
            DEFERRED_REASON_CORRELATED_MULTI_QUBIT
        ),
        "expected_noise_boundary_class": BOUNDARY_CLASS_MODEL_FAMILY,
        "expected_failure_stage": "python_normalization",
        "runner_kind": "constructor",
    },
    {
        "case_name": "deferred_calibration_aware_noise",
        "qbit_num": 1,
        "density_noise": [
            {
                "channel": "calibration_aware_noise",
                "target": 0,
                "after_gate_index": 0,
                "value": 0.1,
            }
        ],
        "purpose": "Verify calibration-aware noise requests remain deferred and fail before execution.",
        "expected_error_fragment": "Unsupported density-noise channel 'calibration_aware_noise'",
        "expected_support_tier": SUPPORT_TIER_DEFERRED,
        "expected_unsupported_category": "noise_type",
        "expected_first_unsupported_condition": DEFERRED_REASON_CALIBRATION_AWARE,
        "expected_noise_boundary_class": BOUNDARY_CLASS_MODEL_FAMILY,
        "expected_failure_stage": "python_normalization",
        "runner_kind": "constructor",
    },
    {
        "case_name": "deferred_non_markovian_noise",
        "qbit_num": 1,
        "density_noise": [
            {
                "channel": "non_markovian_noise",
                "target": 0,
                "after_gate_index": 0,
                "value": 0.1,
            }
        ],
        "purpose": "Verify non-Markovian noise requests remain deferred and fail before execution.",
        "expected_error_fragment": "Unsupported density-noise channel 'non_markovian_noise'",
        "expected_support_tier": SUPPORT_TIER_DEFERRED,
        "expected_unsupported_category": "noise_type",
        "expected_first_unsupported_condition": DEFERRED_REASON_NON_MARKOVIAN,
        "expected_noise_boundary_class": BOUNDARY_CLASS_MODEL_FAMILY,
        "expected_failure_stage": "python_normalization",
        "runner_kind": "constructor",
    },
    {
        "case_name": "invalid_negative_after_gate_index",
        "qbit_num": 4,
        "density_noise": [
            {
                "channel": "local_depolarizing",
                "target": 0,
                "after_gate_index": -1,
                "error_rate": 0.1,
            }
        ],
        "purpose": "Verify negative after_gate_index fails during density-noise configuration.",
        "expected_error_fragment": "after_gate_index must be non-negative",
        "expected_support_tier": SUPPORT_TIER_UNSUPPORTED,
        "expected_unsupported_category": "noise_insertion",
        "expected_first_unsupported_condition": "after_gate_index_negative",
        "expected_noise_boundary_class": BOUNDARY_CLASS_SCHEDULE_ELEMENT,
        "expected_failure_stage": "cxx_noise_spec_validation",
        "runner_kind": "constructor",
    },
    {
        "case_name": "invalid_after_gate_index_exceeds_gate_count",
        "qbit_num": 4,
        "density_noise": [
            {
                "channel": "local_depolarizing",
                "target": 0,
                "after_gate_index": 999,
                "error_rate": 0.1,
            }
        ],
        "purpose": "Verify insertion points beyond the generated gate count fail before execution.",
        "expected_error_fragment": "after_gate_index exceeds generated gate count",
        "expected_support_tier": SUPPORT_TIER_UNSUPPORTED,
        "expected_unsupported_category": "noise_insertion",
        "expected_first_unsupported_condition": (
            "after_gate_index_exceeds_gate_count"
        ),
        "expected_noise_boundary_class": BOUNDARY_CLASS_SCHEDULE_ELEMENT,
        "expected_failure_stage": "density_anchor_preflight",
        "runner_kind": "bridge_preflight",
    },
    {
        "case_name": "invalid_noise_target_out_of_range",
        "qbit_num": 2,
        "density_noise": [
            {
                "channel": "phase_damping",
                "target": 10,
                "after_gate_index": 0,
                "lambda": 0.1,
            }
        ],
        "purpose": "Verify invalid noise targets fail during density-noise configuration.",
        "expected_error_fragment": "target_qbit out of range",
        "expected_support_tier": SUPPORT_TIER_UNSUPPORTED,
        "expected_unsupported_category": "noise_target",
        "expected_first_unsupported_condition": "target_qbit_out_of_range",
        "expected_noise_boundary_class": BOUNDARY_CLASS_CONFIGURATION,
        "expected_failure_stage": "cxx_noise_spec_validation",
        "runner_kind": "constructor",
    },
    {
        "case_name": "invalid_noise_value_out_of_range",
        "qbit_num": 2,
        "density_noise": [
            {
                "channel": "amplitude_damping",
                "target": 0,
                "after_gate_index": 0,
                "gamma": 1.5,
            }
        ],
        "purpose": "Verify out-of-range noise probabilities fail during density-noise configuration.",
        "expected_error_fragment": "noise values must be in [0, 1]",
        "expected_support_tier": SUPPORT_TIER_UNSUPPORTED,
        "expected_unsupported_category": "noise_value",
        "expected_first_unsupported_condition": "noise_value_out_of_range",
        "expected_noise_boundary_class": BOUNDARY_CLASS_CONFIGURATION,
        "expected_failure_stage": "cxx_noise_spec_validation",
        "runner_kind": "constructor",
    },
)


def build_requirement_metadata():
    return {
        "expected_status": "unsupported",
        "support_tier_vocabulary": list(SUPPORT_TIER_VOCABULARY),
        "required_support_tiers": [SUPPORT_TIER_DEFERRED, SUPPORT_TIER_UNSUPPORTED],
        "deferred_family_reasons": list(DEFERRED_FAMILY_REASONS),
        "required_categories": sorted(
            {
                case["expected_unsupported_category"]
                for case in UNSUPPORTED_NOISE_BOUNDARY_CASES
            }
        ),
        "required_boundary_classes": sorted(
            {
                case["expected_noise_boundary_class"]
                for case in UNSUPPORTED_NOISE_BOUNDARY_CASES
            }
        ),
        "required_failure_stages": sorted(
            {case["expected_failure_stage"] for case in UNSUPPORTED_NOISE_BOUNDARY_CASES}
        ),
        "case_names": [case["case_name"] for case in UNSUPPORTED_NOISE_BOUNDARY_CASES],
        "canonical_negative_fields": [
            "support_tier",
            "case_purpose",
            "counts_toward_mandatory_baseline",
            "unsupported_category",
            "first_unsupported_condition",
            "noise_boundary_class",
            "failure_stage",
            "unsupported_reason",
        ],
    }


def _build_hamiltonian(qbit_num: int, topology):
    return generate_zz_xx_hamiltonian(
        n_qubits=qbit_num,
        h=0.5,
        topology=topology,
        Jz=1.0,
        Jx=1.0,
        Jy=1.0,
    )[0]


def _constructor_failure_runner(case):
    topology = build_open_chain_topology(case["qbit_num"])

    def runner():
        Variational_Quantum_Eigensolver(
            _build_hamiltonian(case["qbit_num"], topology),
            case["qbit_num"],
            build_optimizer_config(),
            backend=PRIMARY_BACKEND,
            density_noise=[dict(item) for item in case["density_noise"]],
        )

    return topology, runner


def _bridge_preflight_runner(case):
    def runner():
        vqe, _, _ = build_vqe(
            case["qbit_num"],
            density_noise=[dict(item) for item in case["density_noise"]],
        )
        vqe.describe_density_bridge()

    return build_open_chain_topology(case["qbit_num"]), runner


def _build_case_runner(case):
    if case["runner_kind"] == "constructor":
        return _constructor_failure_runner(case)
    if case["runner_kind"] == "bridge_preflight":
        return _bridge_preflight_runner(case)
    raise ValueError(
        "Unsupported noise-boundary runner kind '{}'".format(case["runner_kind"])
    )


def _base_case_metadata(case, topology):
    metadata = build_case_metadata(
        backend=PRIMARY_BACKEND,
        qbit_num=case["qbit_num"],
        topology=topology,
        density_noise=[dict(item) for item in case["density_noise"]],
        ansatz=DEFAULT_ANSATZ,
        layers=DEFAULT_LAYERS,
        inner_blocks=DEFAULT_INNER_BLOCKS,
        hamiltonian=build_hamiltonian_metadata(),
    )
    metadata.update(
        {
            "case_name": case["case_name"],
            "case_kind": "unsupported_noise_boundary_validation",
            "purpose": case["purpose"],
            "requested_noise_channel": case["density_noise"][0]["channel"],
            "expected_support_tier": case["expected_support_tier"],
            "expected_unsupported_category": case["expected_unsupported_category"],
            "expected_first_unsupported_condition": case[
                "expected_first_unsupported_condition"
            ],
            "expected_noise_boundary_class": case["expected_noise_boundary_class"],
            "expected_failure_stage": case["expected_failure_stage"],
            "silent_substitution_detected": False,
            "silent_fallback_detected": False,
            "pre_execution_failure_pass": False,
            "error_match_pass": False,
            "unsupported_boundary_pass": False,
        }
    )
    return metadata


def validate_case_payload(case):
    required_fields = (
        "case_name",
        "status",
        "backend",
        "qbit_num",
        "topology",
        "ansatz",
        "layers",
        "inner_blocks",
        "density_noise",
        "requested_noise_channel",
        "support_tier",
        "case_purpose",
        "counts_toward_mandatory_baseline",
        "unsupported_category",
        "first_unsupported_condition",
        "noise_boundary_class",
        "failure_stage",
        "unsupported_reason",
        "error_match_pass",
        "pre_execution_failure_pass",
        "unsupported_boundary_pass",
        "silent_substitution_detected",
        "silent_fallback_detected",
    )
    missing_fields = [field for field in required_fields if field not in case]
    if missing_fields:
        raise ValueError(
            "Unsupported-noise boundary case payload is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    if case["support_tier"] == SUPPORT_TIER_DEFERRED and "deferred_reason" not in case:
        raise ValueError(
            "Deferred noise-boundary case '{}' is missing deferred_reason".format(
                case["case_name"]
            )
        )

    if (
        case["support_tier"] == SUPPORT_TIER_UNSUPPORTED
        and "unsupported_scope_reason" not in case
    ):
        raise ValueError(
            "Unsupported noise-boundary case '{}' is missing unsupported_scope_reason".format(
                case["case_name"]
            )
        )


def capture_unsupported_case(case, *, verbose=False):
    topology, runner = _build_case_runner(case)
    metadata = _base_case_metadata(case, topology)
    try:
        runner()
        result = dict(metadata)
        result.update(
            {
                "status": "fail",
                "unsupported_reason": (
                    "Case executed successfully but was expected to fail before "
                    "density execution."
                ),
                "case_purpose": "unexpected_success",
                "counts_toward_mandatory_baseline": False,
                "unsupported_category": case["expected_unsupported_category"],
                "first_unsupported_condition": case[
                    "expected_first_unsupported_condition"
                ],
                "noise_boundary_class": case["expected_noise_boundary_class"],
                "failure_stage": case["expected_failure_stage"],
                "silent_fallback_detected": True,
            }
        )
        if case["expected_support_tier"] == SUPPORT_TIER_DEFERRED:
            result.update(
                {
                    "support_tier": SUPPORT_TIER_DEFERRED,
                    "deferred_reason": case["expected_first_unsupported_condition"],
                }
            )
        else:
            result.update(
                {
                    "support_tier": SUPPORT_TIER_UNSUPPORTED,
                    "unsupported_scope_reason": case[
                        "expected_first_unsupported_condition"
                    ],
                }
            )
    except Exception as exc:
        reason = str(exc)
        classification = classify_noise_boundary_reason(reason)
        result = dict(metadata)
        result.update(classification)
        if result["requested_noise_channel"] is None:
            result["requested_noise_channel"] = metadata["requested_noise_channel"]
        result.update(
            {
                "status": "unsupported",
                "unsupported_reason": reason,
                "error_match_pass": case["expected_error_fragment"] in reason,
                "pre_execution_failure_pass": True,
                "unsupported_boundary_pass": bool(
                    classification["support_tier"] == case["expected_support_tier"]
                    and classification["unsupported_category"]
                    == case["expected_unsupported_category"]
                    and classification["first_unsupported_condition"]
                    == case["expected_first_unsupported_condition"]
                    and classification["noise_boundary_class"]
                    == case["expected_noise_boundary_class"]
                    and classification["failure_stage"] == case["expected_failure_stage"]
                    and not classification["counts_toward_mandatory_baseline"]
                ),
            }
        )

    validate_case_payload(result)
    if verbose:
        print(
            "  {case_name:<54} tier={tier:<11} category={category:<15} status={status}".format(
                case_name=result["case_name"],
                tier=result["support_tier"],
                category=result["unsupported_category"],
                status=result["status"].upper(),
            )
        )
    return result


def run_validation(*, verbose=False):
    return [
        capture_unsupported_case(case, verbose=verbose)
        for case in UNSUPPORTED_NOISE_BOUNDARY_CASES
    ]


def build_artifact_bundle(results):
    results = list(results)
    support_tier_summary = build_support_tier_summary(results)
    total_cases = len(results)
    unsupported_cases = sum(case["status"] == "unsupported" for case in results)
    error_match_count = sum(case["error_match_pass"] for case in results)
    pre_execution_failure_count = sum(case["pre_execution_failure_pass"] for case in results)
    boundary_passed_cases = sum(case["unsupported_boundary_pass"] for case in results)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if total_cases
        and unsupported_cases == total_cases
        and error_match_count == total_cases
        and pre_execution_failure_count == total_cases
        and boundary_passed_cases == total_cases
        else "fail",
        "backend": PRIMARY_BACKEND,
        "requirements": build_requirement_metadata(),
        "software": build_software_metadata(),
        "summary": {
            "total_cases": total_cases,
            "unsupported_status_cases": unsupported_cases,
            "unsupported_cases": unsupported_cases,
            "failed_cases": total_cases - unsupported_cases,
            "error_match_count": error_match_count,
            "pre_execution_failure_count": pre_execution_failure_count,
            "boundary_passed_cases": boundary_passed_cases,
            **support_tier_summary,
            "categories_present": sorted(
                {case["unsupported_category"] for case in results}
            ),
            "boundary_classes_present": sorted(
                {case["noise_boundary_class"] for case in results}
            ),
            "failure_stages_present": sorted(
                {case["failure_stage"] for case in results}
            ),
            "first_unsupported_conditions": sorted(
                {case["first_unsupported_condition"] for case in results}
            ),
            "silent_substitution_failures": sum(
                case["silent_substitution_detected"] for case in results
            ),
            "silent_fallback_failures": sum(
                case["silent_fallback_detected"] for case in results
            ),
        },
        "cases": results,
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Unsupported-noise boundary bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def write_artifact_bundle(output_path: Path, bundle):
    validate_artifact_bundle(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the unsupported-noise boundary JSON artifact bundle.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case progress output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_validation(verbose=not args.quiet)
    bundle = build_artifact_bundle(results)
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} (boundary passed: {}/{})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["boundary_passed_cases"],
            bundle["summary"]["total_cases"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
