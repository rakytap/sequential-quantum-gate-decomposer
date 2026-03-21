from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION,
    CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_SCHEMA_VERSION,
    CORRECTNESS_EVIDENCE_NEGATIVE_RECORD_SCHEMA_VERSION,
    CORRECTNESS_EVIDENCE_RUNTIME_CLASS_BASELINE,
    CORRECTNESS_EVIDENCE_SUMMARY_SCHEMA_VERSION,
)
from benchmarks.density_matrix.correctness_evidence.correctness_matrix_validation import (
    build_artifact_bundle as build_story1_bundle,
    build_cases as build_story1_cases,
)
from benchmarks.density_matrix.correctness_evidence.sequential_correctness_validation import (
    build_artifact_bundle as build_story2_bundle,
    build_cases as build_story2_cases,
)
from benchmarks.density_matrix.correctness_evidence.external_correctness_validation import (
    build_artifact_bundle as build_story3_bundle,
    build_cases as build_story3_cases,
)
from benchmarks.density_matrix.correctness_evidence.output_integrity_validation import (
    build_artifact_bundle as build_story4_bundle,
    build_cases as build_story4_cases,
)
from benchmarks.density_matrix.correctness_evidence.runtime_classification_validation import (
    build_artifact_bundle as build_story5_bundle,
    build_cases as build_story5_cases,
)
from benchmarks.density_matrix.correctness_evidence.unsupported_boundary_validation import (
    build_artifact_bundle as build_story6_bundle,
    build_cases as build_story6_cases,
)
from benchmarks.density_matrix.correctness_evidence.correctness_bundle_validation import (
    build_artifact_bundle as build_story7_bundle,
)
from benchmarks.density_matrix.correctness_evidence.summary_consistency_validation import (
    build_artifact_bundle as build_story8_bundle,
)


def test_correctness_evidence_story1_matrix_covers_required_inventory():
    cases = build_story1_cases()
    assert len(cases) == 25
    assert {case["candidate_id"] for case in cases} == {cases[0]["candidate_id"]}
    assert {case["case_kind"] for case in cases} == {
        "continuity",
        "microcase",
        "structured_family",
    }
    assert sum(case["external_reference_required"] for case in cases) == 4


def test_correctness_evidence_story1_bundle_core_fields_are_stable():
    bundle = build_story1_bundle(build_story1_cases())
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION
    assert bundle["summary"]["continuity_cases"] == 4
    assert bundle["summary"]["microcases"] == 3
    assert bundle["summary"]["structured_cases"] == 18


@pytest.fixture(scope="module")
def story2_cases():
    return build_story2_cases()


def test_correctness_evidence_story2_internal_gate_passes_full_matrix(story2_cases):
    assert len(story2_cases) == 25
    assert all(case["internal_reference_pass"] for case in story2_cases)
    assert all(case["supported_runtime_case"] for case in story2_cases)
    assert all(case["record_schema_version"] == CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION for case in story2_cases)


def test_correctness_evidence_story2_bundle_core_fields_are_stable(story2_cases):
    bundle = build_story2_bundle(story2_cases)
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION
    assert bundle["summary"]["total_cases"] == 25
    assert bundle["summary"]["internal_reference_passes"] == 25


def test_phase3_correctness_evidence_external_correctness_is_bounded_and_exact():
    cases = build_story3_cases()
    assert len(cases) == 4
    assert sum(case["case_kind"] == "microcase" for case in cases) == 3
    assert sum(case["case_kind"] == "continuity" for case in cases) == 1
    assert all(case["external_reference_pass"] for case in cases)


def test_correctness_evidence_story3_bundle_core_fields_are_stable():
    bundle = build_story3_bundle(build_story3_cases())
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION
    assert bundle["summary"]["total_cases"] == 4
    assert bundle["summary"]["external_reference_passes"] == 4


@pytest.fixture(scope="module")
def story4_cases():
    return build_story4_cases()


def test_phase3_correctness_evidence_output_integrity_and_continuity_are_present(story4_cases):
    assert all(case["output_integrity_pass"] for case in story4_cases)
    continuity_cases = [case for case in story4_cases if case["continuity_energy_required"]]
    assert len(continuity_cases) == 4
    assert all(case["continuity_energy_pass"] for case in continuity_cases)


def test_correctness_evidence_story4_bundle_core_fields_are_stable(story4_cases):
    bundle = build_story4_bundle(story4_cases)
    assert bundle["status"] == "pass"
    assert bundle["summary"]["total_cases"] == 25
    assert bundle["summary"]["continuity_cases"] == 4
    assert bundle["summary"]["continuity_energy_passes"] == 4


def test_phase3_correctness_evidence_runtime_classifications_cover_full_matrix():
    cases = build_story5_cases()
    total = sum(
        1
        for case in cases
        if case["runtime_path_classification"]
        in {
            "actually_fused",
            "supported_but_unfused",
            "deferred_or_unsupported_candidate",
            CORRECTNESS_EVIDENCE_RUNTIME_CLASS_BASELINE,
        }
    )
    assert total == len(cases)
    assert all(case["supported_runtime_case"] for case in cases)


def test_correctness_evidence_story5_bundle_core_fields_are_stable():
    bundle = build_story5_bundle(build_story5_cases())
    assert bundle["status"] == "pass"
    assert bundle["summary"]["total_cases"] == 25
    assert sum(
        bundle["summary"][key]
        for key in (
            "actually_fused",
            "supported_but_unfused",
            "deferred_or_unsupported_candidate",
            CORRECTNESS_EVIDENCE_RUNTIME_CLASS_BASELINE,
        )
    ) == 25


def test_correctness_evidence_story6_negative_boundary_is_stage_separated():
    cases = build_story6_cases()
    assert len(cases) >= 3
    assert {case["boundary_stage"] for case in cases} == {
        "planner_entry",
        "descriptor_generation",
        "runtime_stage",
    }
    assert all(case["status"] == "unsupported" for case in cases)
    assert all(
        case["negative_record_schema_version"] == CORRECTNESS_EVIDENCE_NEGATIVE_RECORD_SCHEMA_VERSION
        for case in cases
    )


def test_correctness_evidence_story6_bundle_core_fields_are_stable():
    bundle = build_story6_bundle(build_story6_cases())
    assert bundle["status"] == "pass"
    assert bundle["negative_record_schema_version"] == CORRECTNESS_EVIDENCE_NEGATIVE_RECORD_SCHEMA_VERSION
    assert bundle["summary"]["planner_entry_cases"] >= 1
    assert bundle["summary"]["descriptor_generation_cases"] >= 1
    assert bundle["summary"]["runtime_stage_cases"] >= 1


def test_phase3_correctness_evidence_correctness_package_is_complete():
    bundle = build_story7_bundle()
    assert bundle["status"] == "pass"
    assert bundle["schema_version"] == CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_SCHEMA_VERSION
    assert bundle["summary"]["total_cases"] == 25
    assert bundle["summary"]["counted_supported_cases"] == 25
    assert bundle["summary"]["unsupported_boundary_cases"] == len(bundle["negative_cases"])


def test_phase3_correctness_evidence_summary_consistency_closes_only_from_counted_supported_evidence():
    bundle = build_story8_bundle()
    assert bundle["status"] == "pass"
    assert bundle["schema_version"] == CORRECTNESS_EVIDENCE_SUMMARY_SCHEMA_VERSION
    assert bundle["summary"]["summary_consistency_pass"] is True
    assert bundle["summary"]["main_correctness_claim_completed"] is True
    assert bundle["summary"]["counted_supported_cases"] == 25
