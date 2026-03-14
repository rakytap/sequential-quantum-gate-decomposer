"""Shared Task 4 support-tier vocabulary and bundle summary helpers."""

SUPPORT_TIER_REQUIRED = "required"
SUPPORT_TIER_OPTIONAL = "optional"
SUPPORT_TIER_DEFERRED = "deferred"
SUPPORT_TIER_UNSUPPORTED = "unsupported"

CASE_PURPOSE_MANDATORY_BASELINE = "mandatory_baseline"
CASE_PURPOSE_OPTIONAL_REGRESSION = "optional_regression"
CASE_PURPOSE_OPTIONAL_STRESS = "optional_stress"
CASE_PURPOSE_OPTIONAL_COMPARISON = "optional_comparison"

OPTIONAL_REASON_WHOLE_REGISTER_DEPOLARIZING = "whole_register_depolarizing_baseline"
OPTIONAL_REASON_WORKFLOW_JUSTIFIED_EXTENSION = "workflow_justified_extension"

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
