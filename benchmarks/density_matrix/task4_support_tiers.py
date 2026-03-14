"""Shared Task 4 support-tier vocabulary and bundle summary helpers."""

from __future__ import annotations

import re

SUPPORT_TIER_REQUIRED = "required"
SUPPORT_TIER_OPTIONAL = "optional"
SUPPORT_TIER_DEFERRED = "deferred"
SUPPORT_TIER_UNSUPPORTED = "unsupported"

CASE_PURPOSE_MANDATORY_BASELINE = "mandatory_baseline"
CASE_PURPOSE_OPTIONAL_REGRESSION = "optional_regression"
CASE_PURPOSE_OPTIONAL_STRESS = "optional_stress"
CASE_PURPOSE_OPTIONAL_COMPARISON = "optional_comparison"
CASE_PURPOSE_DEFERRED_SCOPE_GUARD = "deferred_scope_guard"
CASE_PURPOSE_UNSUPPORTED_SCOPE_GUARD = "unsupported_scope_guard"

OPTIONAL_REASON_WHOLE_REGISTER_DEPOLARIZING = "whole_register_depolarizing_baseline"
OPTIONAL_REASON_WORKFLOW_JUSTIFIED_EXTENSION = "workflow_justified_extension"

BOUNDARY_CLASS_MODEL_FAMILY = "model_family"
BOUNDARY_CLASS_SCHEDULE_ELEMENT = "schedule_element"
BOUNDARY_CLASS_CONFIGURATION = "configuration"

DEFERRED_REASON_CORRELATED_MULTI_QUBIT = "correlated_multi_qubit_noise"
DEFERRED_REASON_READOUT = "readout_noise"
DEFERRED_REASON_CALIBRATION_AWARE = "calibration_aware_noise"
DEFERRED_REASON_NON_MARKOVIAN = "non_markovian_noise"

UNSUPPORTED_REASON_DENSITY_NOISE_CHANNEL = "unsupported_density_noise_channel"
UNSUPPORTED_REASON_AFTER_GATE_INDEX_NEGATIVE = "after_gate_index_negative"
UNSUPPORTED_REASON_AFTER_GATE_INDEX_EXCEEDS_GATE_COUNT = (
    "after_gate_index_exceeds_gate_count"
)
UNSUPPORTED_REASON_TARGET_QBIT_OUT_OF_RANGE = "target_qbit_out_of_range"
UNSUPPORTED_REASON_NOISE_VALUE_OUT_OF_RANGE = "noise_value_out_of_range"
UNSUPPORTED_REASON_NOISE_VALUE_NON_FINITE = "noise_value_non_finite"

TASK4_DEFERRED_FAMILY_REASONS = (
    DEFERRED_REASON_CORRELATED_MULTI_QUBIT,
    DEFERRED_REASON_READOUT,
    DEFERRED_REASON_CALIBRATION_AWARE,
    DEFERRED_REASON_NON_MARKOVIAN,
)

TASK4_UNSUPPORTED_CONFIGURATION_REASONS = (
    UNSUPPORTED_REASON_DENSITY_NOISE_CHANNEL,
    UNSUPPORTED_REASON_AFTER_GATE_INDEX_NEGATIVE,
    UNSUPPORTED_REASON_AFTER_GATE_INDEX_EXCEEDS_GATE_COUNT,
    UNSUPPORTED_REASON_TARGET_QBIT_OUT_OF_RANGE,
    UNSUPPORTED_REASON_NOISE_VALUE_OUT_OF_RANGE,
    UNSUPPORTED_REASON_NOISE_VALUE_NON_FINITE,
)

SUPPORT_TIER_VOCABULARY = (
    SUPPORT_TIER_REQUIRED,
    SUPPORT_TIER_OPTIONAL,
    SUPPORT_TIER_DEFERRED,
    SUPPORT_TIER_UNSUPPORTED,
)


def build_required_case_classification():
    return {
        "support_tier": SUPPORT_TIER_REQUIRED,
        "case_purpose": CASE_PURPOSE_MANDATORY_BASELINE,
        "counts_toward_mandatory_baseline": True,
    }


def build_optional_case_classification(*, case_purpose, optional_reason):
    return {
        "support_tier": SUPPORT_TIER_OPTIONAL,
        "case_purpose": case_purpose,
        "optional_reason": optional_reason,
        "counts_toward_mandatory_baseline": False,
    }


def build_deferred_case_classification(
    *,
    deferred_reason,
    case_purpose=CASE_PURPOSE_DEFERRED_SCOPE_GUARD,
):
    return {
        "support_tier": SUPPORT_TIER_DEFERRED,
        "case_purpose": case_purpose,
        "deferred_reason": deferred_reason,
        "counts_toward_mandatory_baseline": False,
    }


def build_unsupported_case_classification(
    *,
    unsupported_reason,
    case_purpose=CASE_PURPOSE_UNSUPPORTED_SCOPE_GUARD,
):
    return {
        "support_tier": SUPPORT_TIER_UNSUPPORTED,
        "case_purpose": case_purpose,
        "unsupported_scope_reason": unsupported_reason,
        "counts_toward_mandatory_baseline": False,
    }


def _extract_requested_channel(reason: str):
    match = re.search(r"Unsupported density-noise channel '([^']+)'", reason)
    if match:
        return match.group(1)
    return None


def _map_requested_channel_to_task4_reason(requested_channel: str | None):
    if not requested_channel:
        return None

    lowered = requested_channel.lower()
    if "readout" in lowered:
        return DEFERRED_REASON_READOUT
    if "correlated" in lowered:
        return DEFERRED_REASON_CORRELATED_MULTI_QUBIT
    if "calibration" in lowered:
        return DEFERRED_REASON_CALIBRATION_AWARE
    if "non_markovian" in lowered or "non-markovian" in lowered:
        return DEFERRED_REASON_NON_MARKOVIAN
    return UNSUPPORTED_REASON_DENSITY_NOISE_CHANNEL


def classify_task4_noise_boundary_reason(reason: str):
    requested_channel = _extract_requested_channel(reason)
    if "Unsupported density-noise channel" in reason:
        first_condition = _map_requested_channel_to_task4_reason(requested_channel)
        if first_condition in TASK4_DEFERRED_FAMILY_REASONS:
            classification = build_deferred_case_classification(
                deferred_reason=first_condition
            )
        else:
            classification = build_unsupported_case_classification(
                unsupported_reason=UNSUPPORTED_REASON_DENSITY_NOISE_CHANNEL
            )
        return {
            "unsupported_category": "noise_type",
            "first_unsupported_condition": first_condition,
            "task4_boundary_class": BOUNDARY_CLASS_MODEL_FAMILY,
            "failure_stage": "python_normalization",
            "requested_noise_channel": requested_channel,
            **classification,
        }

    if "after_gate_index must be non-negative" in reason:
        return {
            "unsupported_category": "noise_insertion",
            "first_unsupported_condition": UNSUPPORTED_REASON_AFTER_GATE_INDEX_NEGATIVE,
            "task4_boundary_class": BOUNDARY_CLASS_SCHEDULE_ELEMENT,
            "failure_stage": "cxx_noise_spec_validation",
            "requested_noise_channel": None,
            **build_unsupported_case_classification(
                unsupported_reason=UNSUPPORTED_REASON_AFTER_GATE_INDEX_NEGATIVE
            ),
        }

    if "after_gate_index exceeds generated gate count" in reason:
        return {
            "unsupported_category": "noise_insertion",
            "first_unsupported_condition": (
                UNSUPPORTED_REASON_AFTER_GATE_INDEX_EXCEEDS_GATE_COUNT
            ),
            "task4_boundary_class": BOUNDARY_CLASS_SCHEDULE_ELEMENT,
            "failure_stage": "density_anchor_preflight",
            "requested_noise_channel": None,
            **build_unsupported_case_classification(
                unsupported_reason=UNSUPPORTED_REASON_AFTER_GATE_INDEX_EXCEEDS_GATE_COUNT
            ),
        }

    if "target_qbit out of range" in reason:
        return {
            "unsupported_category": "noise_target",
            "first_unsupported_condition": UNSUPPORTED_REASON_TARGET_QBIT_OUT_OF_RANGE,
            "task4_boundary_class": BOUNDARY_CLASS_CONFIGURATION,
            "failure_stage": "cxx_noise_spec_validation",
            "requested_noise_channel": None,
            **build_unsupported_case_classification(
                unsupported_reason=UNSUPPORTED_REASON_TARGET_QBIT_OUT_OF_RANGE
            ),
        }

    if "noise values must be in [0, 1]" in reason:
        return {
            "unsupported_category": "noise_value",
            "first_unsupported_condition": UNSUPPORTED_REASON_NOISE_VALUE_OUT_OF_RANGE,
            "task4_boundary_class": BOUNDARY_CLASS_CONFIGURATION,
            "failure_stage": "cxx_noise_spec_validation",
            "requested_noise_channel": None,
            **build_unsupported_case_classification(
                unsupported_reason=UNSUPPORTED_REASON_NOISE_VALUE_OUT_OF_RANGE
            ),
        }

    if "non-finite value" in reason:
        return {
            "unsupported_category": "noise_value",
            "first_unsupported_condition": UNSUPPORTED_REASON_NOISE_VALUE_NON_FINITE,
            "task4_boundary_class": BOUNDARY_CLASS_CONFIGURATION,
            "failure_stage": "python_normalization",
            "requested_noise_channel": None,
            **build_unsupported_case_classification(
                unsupported_reason=UNSUPPORTED_REASON_NOISE_VALUE_NON_FINITE
            ),
        }

    return {
        "unsupported_category": "workflow_execution",
        "first_unsupported_condition": "workflow_execution",
        "task4_boundary_class": "workflow_execution",
        "failure_stage": "workflow_execution",
        "requested_noise_channel": requested_channel,
        **build_unsupported_case_classification(
            unsupported_reason="workflow_execution"
        ),
    }


def build_task4_support_tier_summary(cases):
    cases = list(cases)
    required_cases = [
        case for case in cases if case.get("support_tier") == SUPPORT_TIER_REQUIRED
    ]
    optional_cases = [
        case for case in cases if case.get("support_tier") == SUPPORT_TIER_OPTIONAL
    ]
    deferred_cases = [
        case for case in cases if case.get("support_tier") == SUPPORT_TIER_DEFERRED
    ]
    unsupported_cases = [
        case for case in cases if case.get("support_tier") == SUPPORT_TIER_UNSUPPORTED
    ]
    mandatory_cases = [
        case for case in cases if case.get("counts_toward_mandatory_baseline")
    ]

    required_passed_cases = sum(case.get("status") == "pass" for case in required_cases)
    optional_passed_cases = sum(case.get("status") == "pass" for case in optional_cases)
    mandatory_passed_cases = sum(case.get("status") == "pass" for case in mandatory_cases)

    def _rate(passed, total):
        return 0.0 if total == 0 else passed / total

    return {
        "required_cases": len(required_cases),
        "required_passed_cases": required_passed_cases,
        "required_pass_rate": _rate(required_passed_cases, len(required_cases)),
        "optional_cases": len(optional_cases),
        "optional_passed_cases": optional_passed_cases,
        "optional_pass_rate": _rate(optional_passed_cases, len(optional_cases)),
        "deferred_cases": len(deferred_cases),
        "unsupported_cases": len(unsupported_cases),
        "mandatory_baseline_case_count": len(mandatory_cases),
        "mandatory_baseline_passed_cases": mandatory_passed_cases,
        "mandatory_baseline_completed": bool(mandatory_cases)
        and mandatory_passed_cases == len(mandatory_cases),
        "optional_cases_count_toward_mandatory_baseline": sum(
            case.get("counts_toward_mandatory_baseline", False)
            for case in optional_cases
        ),
        "support_tiers_present": sorted(
            {
                case.get("support_tier")
                for case in cases
                if case.get("support_tier") is not None
            }
        ),
    }
