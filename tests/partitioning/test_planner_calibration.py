from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_calibration.common import (
    PLANNER_CALIBRATION_CANDIDATE_SCHEMA_VERSION,
    PLANNER_CALIBRATION_PLANNER_FAMILY_SPAN_BUDGET,
    build_planner_calibration_planner_candidates,
)
from benchmarks.density_matrix.planner_calibration.planner_candidate_audit_validation import (
    build_artifact_bundle as build_story1_bundle,
)
from benchmarks.density_matrix.planner_calibration.planner_candidate_audit_validation import (
    build_cases as build_story1_cases,
)
from benchmarks.density_matrix.planner_calibration.calibration_workload_matrix_validation import (
    build_artifact_bundle as build_story2_bundle,
)
from benchmarks.density_matrix.planner_calibration.calibration_workload_matrix_validation import (
    build_cases as build_story2_cases,
)
from benchmarks.density_matrix.planner_calibration.density_signal_validation import (
    build_artifact_bundle as build_story3_bundle,
)
from benchmarks.density_matrix.planner_calibration.density_signal_validation import (
    build_cases as build_story3_cases,
)
from benchmarks.density_matrix.planner_calibration.calibration_correctness_validation import (
    build_artifact_bundle as build_story4_bundle,
)
from benchmarks.density_matrix.planner_calibration.calibration_correctness_validation import (
    build_cases as build_story4_cases,
)
from benchmarks.density_matrix.planner_calibration.claim_selection import (
    PLANNER_CALIBRATION_CLAIM_STATUS_COMPARISON,
    PLANNER_CALIBRATION_CLAIM_STATUS_SUPPORTED,
    build_planner_calibration_claim_selection_payload,
)
from benchmarks.density_matrix.planner_calibration.calibrated_claim_selection_validation import (
    build_artifact_bundle as build_story5_bundle,
)
from benchmarks.density_matrix.planner_calibration.bundle import (
    build_planner_calibration_calibration_bundle_payload,
)
from benchmarks.density_matrix.planner_calibration.calibration_bundle_validation import (
    build_artifact_bundle as build_story6_bundle,
)
from benchmarks.density_matrix.planner_calibration.boundary import (
    build_planner_calibration_boundary_payload,
)
from benchmarks.density_matrix.planner_calibration.calibration_boundary_validation import (
    build_artifact_bundle as build_story7_bundle,
)


def test_planner_calibration_story1_candidate_enumeration_is_deterministic():
    candidates = build_planner_calibration_planner_candidates()
    assert [candidate.candidate_id for candidate in candidates] == [
        "span_budget_q2",
        "span_budget_q3",
        "span_budget_q4",
    ]
    assert [candidate.max_partition_qubits for candidate in candidates] == [2, 3, 4]
    assert {candidate.schema_version for candidate in candidates} == {
        PLANNER_CALIBRATION_CANDIDATE_SCHEMA_VERSION
    }
    assert {candidate.planner_family for candidate in candidates} == {
        PLANNER_CALIBRATION_PLANNER_FAMILY_SPAN_BUDGET
    }


def test_planner_calibration_planner_candidate_audit_covers_required_workload_kinds():
    cases = build_story1_cases()
    assert [case["candidate_id"] for case in cases] == [
        "span_budget_q2",
        "span_budget_q3",
        "span_budget_q4",
    ]
    for case in cases:
        assert case["candidate_surface_pass"] is True
        assert case["supported_case_kinds"] == [
            "continuity",
            "microcase",
            "structured_family",
        ]
        assert len(case["representative_workload_ids"]) == 3
        assert len(case["representative_cases"]) == 3
        assert all(
            representative["max_partition_qubits"] == case["max_partition_qubits"]
            for representative in case["representative_cases"]
        )


def test_planner_calibration_story1_bundle_core_fields_are_stable():
    bundle = build_story1_bundle(build_story1_cases())
    assert bundle["status"] == "pass"
    assert bundle["candidate_schema_version"] == PLANNER_CALIBRATION_CANDIDATE_SCHEMA_VERSION
    assert bundle["summary"]["total_candidates"] == 3
    assert bundle["summary"]["representative_case_kinds"] == [
        "continuity",
        "microcase",
        "structured_family",
    ]


def test_phase3_planner_calibration_workload_matrix_covers_required_case_inventory():
    cases = build_story2_cases()
    assert len(cases) == 75
    assert {case["candidate_id"] for case in cases} == {
        "span_budget_q2",
        "span_budget_q3",
        "span_budget_q4",
    }
    assert {case["case_kind"] for case in cases} == {
        "continuity",
        "microcase",
        "structured_family",
    }
    assert {case["qbit_num"] for case in cases if case["case_kind"] == "continuity"} == {
        4,
        6,
        8,
        10,
    }
    assert all(case["workload_matrix_pass"] is True for case in cases)


def test_planner_calibration_story2_bundle_core_fields_are_stable():
    bundle = build_story2_bundle(build_story2_cases())
    assert bundle["status"] == "pass"
    assert bundle["candidate_schema_version"] == PLANNER_CALIBRATION_CANDIDATE_SCHEMA_VERSION
    assert bundle["summary"]["continuity_cases"] == 12
    assert bundle["summary"]["microcases"] == 9
    assert bundle["summary"]["structured_cases"] == 54


def test_phase3_planner_calibration_density_signal_surface_is_present_and_ranked():
    cases = build_story3_cases()
    assert len(cases) == 12
    assert all(case["density_signal_pass"] is True for case in cases)
    assert {case["candidate_id"] for case in cases} == {
        "span_budget_q2",
        "span_budget_q3",
        "span_budget_q4",
    }
    assert all(case["density_aware_rank"] in {1, 2, 3} for case in cases)
    assert all(case["state_vector_proxy_rank"] in {1, 2, 3} for case in cases)


def test_planner_calibration_story3_density_score_changes_with_noise_pattern():
    cases = build_story3_cases()
    layered_by_candidate = {}
    for case in cases:
        if (
            case["case_kind"] == "structured_family"
            and case["family_name"] == "layered_nearest_neighbor"
            and case["qbit_num"] == 8
        ):
            layered_by_candidate.setdefault(case["candidate_id"], {})[
                case["noise_pattern"]
            ] = case["density_aware_score"]

    assert set(layered_by_candidate) == {
        "span_budget_q2",
        "span_budget_q3",
        "span_budget_q4",
    }
    for scores in layered_by_candidate.values():
        assert set(scores) == {"sparse", "dense"}
        assert scores["dense"] != scores["sparse"]


def test_planner_calibration_story3_bundle_core_fields_are_stable():
    bundle = build_story3_bundle(build_story3_cases())
    assert bundle["status"] == "pass"
    assert bundle["signal_schema_version"] == "phase3_planner_calibration_density_signal_v1"
    assert bundle["summary"]["total_cases"] == 12
    assert bundle["summary"]["noise_sensitive_slices"] >= 1


@pytest.fixture(scope="module")
def story4_cases():
    return build_story4_cases()


def test_phase3_planner_calibration_correctness_gate_marks_full_matrix_counted(story4_cases):
    assert len(story4_cases) == 75
    assert all(case["internal_correctness_pass"] is True for case in story4_cases)
    assert all(case["counted_calibration_case"] is True for case in story4_cases)
    assert sum(case["external_reference_required"] for case in story4_cases) == 12
    assert all(
        (not case["external_reference_required"]) or case["external_reference_pass"]
        for case in story4_cases
    )


def test_planner_calibration_story4_bundle_core_fields_are_stable(story4_cases):
    bundle = build_story4_bundle(story4_cases)
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == "phase3_planner_calibration_record_v1"
    assert bundle["summary"]["total_cases"] == 75
    assert bundle["summary"]["external_reference_cases"] == 12


@pytest.fixture(scope="module")
def story5_payload():
    return build_planner_calibration_claim_selection_payload()


def test_planner_calibration_story5_selects_one_supported_candidate(story5_payload):
    assert story5_payload["selected_candidate"]["candidate_id"] in {
        "span_budget_q2",
        "span_budget_q3",
        "span_budget_q4",
    }
    assert {summary["candidate_id"] for summary in story5_payload["candidate_summaries"]} == {
        "span_budget_q2",
        "span_budget_q3",
        "span_budget_q4",
    }
    assert {case["claim_status"] for case in story5_payload["cases"]} == {
        PLANNER_CALIBRATION_CLAIM_STATUS_SUPPORTED,
        PLANNER_CALIBRATION_CLAIM_STATUS_COMPARISON,
    }


def test_planner_calibration_story5_bundle_core_fields_are_stable():
    bundle = build_story5_bundle()
    assert bundle["status"] == "pass"
    assert bundle["claim_selection_schema_version"] == "phase3_planner_calibration_claim_selection_v1"
    assert bundle["summary"]["selected_candidate_id"] in {
        "span_budget_q2",
        "span_budget_q3",
        "span_budget_q4",
    }


def test_planner_calibration_story6_shared_bundle_is_consumer_stable():
    payload = build_planner_calibration_calibration_bundle_payload()
    assert payload["selected_candidate"]["candidate_id"] in {
        "span_budget_q2",
        "span_budget_q3",
        "span_budget_q4",
    }
    assert payload["summary"]["total_cases"] == 75
    assert payload["summary"]["supported_claim_cases"] == 25
    assert payload["summary"]["comparison_cases"] == 50


def test_planner_calibration_story6_bundle_core_fields_are_stable():
    bundle = build_story6_bundle()
    assert bundle["status"] == "pass"
    assert bundle["calibration_bundle_schema_version"] == "phase3_planner_calibration_bundle_v1"
    assert bundle["summary"]["total_cases"] == 75


def test_planner_calibration_story7_boundary_surface_is_explicit():
    payload = build_planner_calibration_boundary_payload()
    assert payload["supported_claim"]["candidate_id"] in {
        "span_budget_q2",
        "span_budget_q3",
        "span_budget_q4",
    }
    assert payload["summary"]["comparison_baseline_case_count"] == 50
    assert payload["summary"]["approximation_area_count"] == 4
    assert payload["summary"]["deferred_follow_on_branch_count"] == 4


def test_planner_calibration_story7_bundle_core_fields_are_stable():
    bundle = build_story7_bundle()
    assert bundle["status"] == "pass"
    assert bundle["boundary_schema_version"] == "phase3_planner_calibration_claim_boundary_v1"
    assert bundle["summary"]["selected_candidate_id"] in {
        "span_budget_q2",
        "span_budget_q3",
        "span_budget_q4",
    }
