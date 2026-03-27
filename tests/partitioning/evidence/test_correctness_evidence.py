from collections.abc import Callable
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION,
    CORRECTNESS_EVIDENCE_NEGATIVE_RECORD_SCHEMA_VERSION,
    CORRECTNESS_EVIDENCE_RUNTIME_CLASS_BASELINE,
    CORRECTNESS_EVIDENCE_SUMMARY_SCHEMA_VERSION,
    CORRECTNESS_PACKAGE_SCHEMA_VERSION,
)
from benchmarks.density_matrix.correctness_evidence.correctness_matrix_validation import (
    build_artifact_bundle as build_correctness_matrix_bundle,
    build_cases as build_correctness_matrix_cases,
)
from benchmarks.density_matrix.correctness_evidence.sequential_correctness_validation import (
    build_artifact_bundle as build_sequential_correctness_bundle,
    build_cases as build_sequential_correctness_cases,
)
from benchmarks.density_matrix.correctness_evidence.external_correctness_validation import (
    build_artifact_bundle as build_external_correctness_bundle,
    build_cases as build_external_correctness_cases,
)
from benchmarks.density_matrix.correctness_evidence.output_integrity_validation import (
    build_artifact_bundle as build_output_integrity_bundle,
    build_cases as build_output_integrity_cases,
)
from benchmarks.density_matrix.correctness_evidence.runtime_classification_validation import (
    build_artifact_bundle as build_runtime_classification_bundle,
    build_cases as build_runtime_classification_cases,
)
from benchmarks.density_matrix.correctness_evidence.unsupported_boundary_validation import (
    build_artifact_bundle as build_unsupported_boundary_bundle,
    build_cases as build_unsupported_boundary_cases,
)
from benchmarks.density_matrix.correctness_evidence.correctness_bundle_validation import (
    build_artifact_bundle as build_correctness_package_bundle,
)
from benchmarks.density_matrix.correctness_evidence.summary_consistency_validation import (
    build_artifact_bundle as build_summary_consistency_bundle,
)
from tests.partitioning.evidence.bundle_assertions import (
    assert_correctness_full_package_bundle,
    assert_correctness_unsupported_boundary_bundle_core,
)

_CORRECTNESS_POSITIVE_CASE_SLICES: tuple[
    tuple[Callable[[], list[dict]], Callable[[list[dict]], dict]],
    ...,
] = (
    (build_correctness_matrix_cases, build_correctness_matrix_bundle),
    (build_sequential_correctness_cases, build_sequential_correctness_bundle),
    (build_external_correctness_cases, build_external_correctness_bundle),
    (build_output_integrity_cases, build_output_integrity_bundle),
    (build_runtime_classification_cases, build_runtime_classification_bundle),
)

_CORRECTNESS_POSITIVE_CASE_SLICE_IDS = (
    "correctness_matrix",
    "sequential_correctness",
    "external_correctness",
    "output_integrity",
    "runtime_classification",
)


@pytest.mark.parametrize(
    "build_cases_fn,build_bundle_fn",
    _CORRECTNESS_POSITIVE_CASE_SLICES,
    ids=list(_CORRECTNESS_POSITIVE_CASE_SLICE_IDS),
)
def test_correctness_evidence_positive_case_slice_bundle_schema_and_pass(
    build_cases_fn: Callable[[], list[dict]],
    build_bundle_fn: Callable[[list[dict]], dict],
):
    cases = build_cases_fn()
    bundle = build_bundle_fn(cases)
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION


def test_correctness_evidence_correctness_matrix_covers_required_inventory():
    cases = build_correctness_matrix_cases()
    assert len(cases) == 25
    assert {case["candidate_id"] for case in cases} == {cases[0]["candidate_id"]}
    assert {case["case_kind"] for case in cases} == {
        "continuity",
        "microcase",
        "structured_family",
    }
    assert sum(case["external_reference_required"] for case in cases) == 4


def test_correctness_evidence_correctness_matrix_bundle_summary_counts():
    bundle = build_correctness_matrix_bundle(build_correctness_matrix_cases())
    assert bundle["summary"]["continuity_cases"] == 4
    assert bundle["summary"]["microcases"] == 3
    assert bundle["summary"]["structured_cases"] == 18


@pytest.fixture(scope="module")
def sequential_correctness_cases():
    return build_sequential_correctness_cases()


def test_correctness_evidence_sequential_correctness_internal_gate_passes_full_matrix(
    sequential_correctness_cases,
):
    assert len(sequential_correctness_cases) == 25
    assert all(case["internal_reference_pass"] for case in sequential_correctness_cases)
    assert all(case["supported_runtime_case"] for case in sequential_correctness_cases)
    assert all(
        case["record_schema_version"] == CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION
        for case in sequential_correctness_cases
    )


def test_correctness_evidence_sequential_correctness_bundle_summary():
    bundle = build_sequential_correctness_bundle(build_sequential_correctness_cases())
    assert bundle["summary"]["total_cases"] == 25
    assert bundle["summary"]["internal_reference_passes"] == 25


def test_correctness_evidence_external_correctness_is_bounded_and_exact():
    cases = build_external_correctness_cases()
    assert len(cases) == 4
    assert sum(case["case_kind"] == "microcase" for case in cases) == 3
    assert sum(case["case_kind"] == "continuity" for case in cases) == 1
    assert all(case["external_reference_pass"] for case in cases)


def test_correctness_evidence_external_correctness_bundle_summary():
    bundle = build_external_correctness_bundle(build_external_correctness_cases())
    assert bundle["summary"]["total_cases"] == 4
    assert bundle["summary"]["external_reference_passes"] == 4


@pytest.fixture(scope="module")
def output_integrity_cases():
    return build_output_integrity_cases()


def test_correctness_evidence_output_integrity_and_continuity_are_present(
    output_integrity_cases,
):
    assert all(case["output_integrity_pass"] for case in output_integrity_cases)
    continuity_cases = [case for case in output_integrity_cases if case["continuity_energy_required"]]
    assert len(continuity_cases) == 4
    assert all(case["continuity_energy_pass"] for case in continuity_cases)


def test_correctness_evidence_output_integrity_bundle_summary(output_integrity_cases):
    bundle = build_output_integrity_bundle(output_integrity_cases)
    assert bundle["summary"]["total_cases"] == 25
    assert bundle["summary"]["continuity_cases"] == 4
    assert bundle["summary"]["continuity_energy_passes"] == 4


def test_correctness_evidence_runtime_classifications_cover_full_matrix():
    cases = build_runtime_classification_cases()
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


def test_correctness_evidence_runtime_classification_bundle_summary():
    bundle = build_runtime_classification_bundle(build_runtime_classification_cases())
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


def test_correctness_evidence_unsupported_boundary_negative_evidence_is_stage_separated():
    cases = build_unsupported_boundary_cases()
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


def test_correctness_evidence_unsupported_boundary_bundle_core_fields_are_stable():
    bundle = build_unsupported_boundary_bundle(build_unsupported_boundary_cases())
    assert_correctness_unsupported_boundary_bundle_core(
        bundle,
        negative_record_schema_version=CORRECTNESS_EVIDENCE_NEGATIVE_RECORD_SCHEMA_VERSION,
    )


def test_correctness_evidence_correctness_package_is_complete():
    bundle = build_correctness_package_bundle()
    assert_correctness_full_package_bundle(
        bundle,
        schema_version=CORRECTNESS_PACKAGE_SCHEMA_VERSION,
        unsupported_case_count=len(bundle["negative_cases"]),
    )


def test_correctness_evidence_summary_consistency_closes_only_from_counted_supported_evidence():
    bundle = build_summary_consistency_bundle()
    assert bundle["status"] == "pass"
    assert bundle["schema_version"] == CORRECTNESS_EVIDENCE_SUMMARY_SCHEMA_VERSION
    assert bundle["summary"]["summary_consistency_pass"] is True
    assert bundle["summary"]["main_correctness_claim_completed"] is True
    assert bundle["summary"]["counted_supported_cases"] == 25
