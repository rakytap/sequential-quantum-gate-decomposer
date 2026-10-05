from collections.abc import Callable
import copy
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
from benchmarks.density_matrix.correctness_evidence import (
    mf1a_q4_baseline_validation as mf1a,
    validation_pipeline,
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


def _clean_mf1a_provenance() -> dict:
    return {
        "implementation_revision": "a" * 40,
        "clean_start": True,
        "dirty_paths": [],
        "command": mf1a.REGENERATION_COMMAND,
        "environment": {
            "conda_default_env": "qgd",
            "conda_prefix": "/tmp/qgd",
            "python_executable": "/tmp/qgd/bin/python",
            "python_version": "3.13.0",
        },
        "dependencies": {"numpy": "test", "scipy": "test", "squander": "test"},
        "extension_identities": [
            {"path": "squander/density_matrix/_density_matrix_cpp.so", "sha256": "b" * 64}
        ],
        "input_artifact_identities": [],
        "provenance_pass": True,
    }


def test_mf1a_q4_baseline_manifest_accepts_only_the_reviewed_cell():
    assert mf1a.build_manifest() == {
        "schema_version": mf1a.MANIFEST_SCHEMA_VERSION,
        "cells": [
            {
                "anchor_qbits": 4,
                "workload": "phase2_xxz_hea_q4_continuity",
                "route": "partitioned_density_descriptor_baseline",
                "max_partition_qubits": 2,
            }
        ],
    }
    with pytest.raises(ValueError, match="exactly one reviewed"):
        mf1a.validate_manifest_cells([])


def test_mf1a_q4_baseline_record_has_complete_provenance_and_claim_boundary():
    cases = mf1a.build_cases(provenance=_clean_mf1a_provenance())
    assert len(cases) == 1
    case = cases[0]
    assert case["record_schema_version"] == mf1a.RECORD_SCHEMA_VERSION
    assert case["milestone_counted"] is False
    assert case["completeness_claim"] is False
    assert case["provenance"]["provenance_pass"] is True
    assert case["parameters"] == pytest.approx(
        [0.05 * index for index in range(1, 19)]
    )
    assert "q4 baseline tracer only" in case["claim_boundary"]


def test_mf1a_q4_baseline_dirty_pre_run_is_non_counted():
    provenance = _clean_mf1a_provenance()
    provenance.update(
        clean_start=False, dirty_paths=["tests/example.py"], provenance_pass=False
    )
    bundle = mf1a.build_artifact_bundle(
        mf1a.build_cases(provenance=provenance), prior_bundle=None
    )
    assert bundle["status"] == "fail"
    assert bundle["summary"]["first_failure"] == "provenance"
    assert bundle["cases"][0]["milestone_counted"] is False


def test_mf1a_q4_baseline_aer_and_energy_context_are_non_counted():
    cases = mf1a.build_cases(provenance=_clean_mf1a_provenance())
    baseline = mf1a.build_artifact_bundle(cases, prior_bundle=None)
    contextual = mf1a.build_artifact_bundle(
        cases,
        prior_bundle=None,
        non_counted_context={"aer": "fail", "energy": "fail"},
    )
    assert contextual["status"] == baseline["status"] == "pass"


def test_mf1a_q4_baseline_record_rejects_fused_realization():
    cases = mf1a.build_cases(provenance=_clean_mf1a_provenance())
    cases[0]["realization"]["actual_fused_execution"] = True
    bundle = mf1a.build_artifact_bundle(cases, prior_bundle=None)
    assert bundle["status"] == "fail"
    assert bundle["summary"]["first_failure"] == "route_realization"


def test_mf1a_q4_baseline_bundle_schema_and_summary():
    bundle = mf1a.build_artifact_bundle(
        mf1a.build_cases(provenance=_clean_mf1a_provenance()), prior_bundle=None
    )
    assert bundle["schema_version"] == mf1a.BUNDLE_SCHEMA_VERSION
    assert bundle["suite_name"] == mf1a.SUITE_NAME
    assert bundle["status"] == "pass"
    assert bundle["summary"]["qa001_passes"] == 1
    assert bundle["summary"]["milestone_counted_cases"] == 0


def test_mf1a_q4_baseline_regeneration_accepts_frozen_residuals():
    cases = mf1a.build_cases(provenance=_clean_mf1a_provenance())
    prior = mf1a.build_artifact_bundle(cases, prior_bundle=None)
    current = mf1a.build_artifact_bundle(cases, prior_bundle=prior)
    assert current["status"] == "pass"
    assert current["regeneration"]["pass"] is True


def test_mf1a_q4_baseline_regeneration_rejects_categorical_or_residual_drift():
    cases = mf1a.build_cases(provenance=_clean_mf1a_provenance())
    prior = copy.deepcopy(mf1a.build_artifact_bundle(cases, prior_bundle=None))
    prior["cases"][0]["route"] = "wrong"
    current = mf1a.build_artifact_bundle(cases, prior_bundle=prior)
    assert current["status"] == "fail"
    assert current["regeneration"]["first_mismatch"] == "cases[0].route"

    prior = copy.deepcopy(mf1a.build_artifact_bundle(cases, prior_bundle=None))
    prior["cases"][0]["qa001"]["frobenius_norm_diff"] += 2e-10
    current = mf1a.build_artifact_bundle(cases, prior_bundle=prior)
    assert current["status"] == "fail"
    assert current["regeneration"]["first_mismatch"].endswith(
        "frobenius_norm_diff"
    )


def test_mf1a_q4_baseline_exit_aggregate_uses_exact_g07_set():
    results = [
        (name, "pass", Path("/tmp") / f"{name}.json")
        for name in validation_pipeline.registered_suite_names()
    ]
    assert validation_pipeline.g07_exit_passes(results)
    included = set(validation_pipeline.g07_included_suite_names())
    registered = set(validation_pipeline.registered_suite_names())
    assert registered - included == {
        "correctness_evidence_external_correctness",
        "correctness_evidence_output_integrity",
    }
    assert mf1a.SUITE_NAME in included


def test_mf1a_q4_baseline_exit_aggregate_requires_sibling_and_included_passes():
    passing = [
        (name, "pass", Path("/tmp") / f"{name}.json")
        for name in validation_pipeline.registered_suite_names()
    ]
    assert not validation_pipeline.g07_exit_passes(
        [item for item in passing if item[0] != mf1a.SUITE_NAME]
    )
    assert not validation_pipeline.g07_exit_passes(
        [
            (name, "fail" if name == mf1a.SUITE_NAME else status, path)
            for name, status, path in passing
        ]
    )
    assert validation_pipeline.g07_exit_passes(
        [
            (
                name,
                "fail"
                if name
                in {
                    "correctness_evidence_external_correctness",
                    "correctness_evidence_output_integrity",
                }
                else status,
                path,
            )
            for name, status, path in passing
        ]
    )
