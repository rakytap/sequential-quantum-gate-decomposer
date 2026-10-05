from collections.abc import Callable
import copy
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

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


@pytest.fixture(scope="module")
def mf1a_q4_baseline_regeneration_cases():
    return mf1a.build_cases(provenance=_clean_mf1a_provenance())


def _mf1a_case_field_diff_paths(
    left: object, right: object, prefix: str = "cases[0]"
) -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child_prefix = f"{prefix}.{key}"
            if key not in left or key not in right:
                paths.append(child_prefix)
            else:
                paths.extend(
                    _mf1a_case_field_diff_paths(left[key], right[key], child_prefix)
                )
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        paths: list[str] = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(
                _mf1a_case_field_diff_paths(
                    left_item, right_item, f"{prefix}[{index}]"
                )
            )
        return paths
    if left != right:
        return [prefix]
    return []


def test_mf1a_q4_baseline_regeneration_allowlist_is_length_one():
    assert mf1a.Q4_REGENERATION_ALLOWLIST == (
        "cases[0].provenance.implementation_revision",
    )
    assert len(mf1a.Q4_REGENERATION_ALLOWLIST) == 1
    assert "cases[*].provenance.implementation_revision" not in mf1a.Q4_REGENERATION_ALLOWLIST
    assert mf1a.BUNDLE_SCHEMA_VERSION == (
        "correctness_evidence_mf1a_q4_baseline_bundle_v1"
    )
    assert mf1a.RECORD_SCHEMA_VERSION == (
        "correctness_evidence_mf1a_q4_baseline_case_v1"
    )
    assert mf1a.MANIFEST_SCHEMA_VERSION == (
        "correctness_evidence_mf1a_q4_baseline_manifest_v1"
    )
    assert set(mf1a._QA001_REGENERATION_TOLERANCES) == {
        "frobenius_norm_diff",
        "max_abs_diff",
        "trace_abs_deviation",
        "lambda_min",
    }
    assert mf1a._QA001_REGENERATION_TOLERANCES == {
        "frobenius_norm_diff": 1e-10,
        "max_abs_diff": 1e-10,
        "trace_abs_deviation": 1e-10,
        "lambda_min": 1e-12,
    }


def test_mf1a_q4_baseline_regeneration_passes_on_revision_only(
    mf1a_q4_baseline_regeneration_cases,
):
    prior_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    prior = mf1a.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40
    current = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior)
    assert current["status"] == "pass"
    assert current["regeneration"]["pass"] is True
    assert current["regeneration"]["first_mismatch"] is None
    assert current["regeneration"]["prior_present"] is True
    assert current["summary"]["first_failure"] is None
    assert set(current["regeneration"]) == {"prior_present", "pass", "first_mismatch"}
    assert current["schema_version"] == prior["schema_version"]
    case_diff = _mf1a_case_field_diff_paths(
        prior["cases"][0], current["cases"][0]
    )
    assert case_diff == ["cases[0].provenance.implementation_revision"]


def test_mf1a_q4_baseline_regeneration_fails_on_revision_plus_second_field(
    mf1a_q4_baseline_regeneration_cases,
):
    prior_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    prior = mf1a.build_artifact_bundle(prior_cases, prior_bundle=None)

    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40
    current_cases[0]["seed_policy"] = "changed"
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "cases[0].seed_policy"
    assert bundle["summary"]["first_failure"] == "regeneration"

    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40
    current_cases[0]["provenance"]["clean_start"] = False
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "cases[0].provenance.clean_start"
    assert bundle["summary"]["first_failure"] == "regeneration"

    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40
    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][0]["route"] = "wrong"
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "cases[0].route"
    assert bundle["summary"]["first_failure"] == "regeneration"

    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40
    current_cases[0]["realization"]["actual_fused_execution"] = True
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior)
    assert bundle["status"] == "fail"
    assert bundle["regeneration"]["first_mismatch"] == "cases[0].realization"
    assert bundle["summary"]["first_failure"] == "route_realization"


def test_mf1a_q4_baseline_regeneration_fails_on_extension_sha256(
    mf1a_q4_baseline_regeneration_cases,
):
    prior_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    prior = mf1a.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40

    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][0]["provenance"]["extension_identities"][0]["sha256"] = "e" * 64
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[0].provenance.extension_identities"
    )

    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][0]["provenance"]["extension_identities"][0]["path"] = (
        "other/path.so"
    )
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[0].provenance.extension_identities"
    )


def test_mf1a_q4_baseline_regeneration_fails_on_input_identity(
    mf1a_q4_baseline_regeneration_cases,
):
    prior_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    prior = mf1a.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40
    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][0]["provenance"]["input_artifact_identities"] = [
        {"path": "inputs/foo.json", "sha256": "f" * 64}
    ]
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[0].provenance.input_artifact_identities"
    )


def test_mf1a_q4_baseline_regeneration_fails_on_dependency_version(
    mf1a_q4_baseline_regeneration_cases,
):
    prior_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    prior = mf1a.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40
    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][0]["provenance"]["dependencies"]["numpy"] = "other"
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[0].provenance.dependencies"
    )


def test_mf1a_q4_baseline_regeneration_fails_on_environment_identity(
    mf1a_q4_baseline_regeneration_cases,
):
    prior_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    prior = mf1a.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40
    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][0]["provenance"]["environment"]["python_version"] = "3.13.1"
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[0].provenance.environment"
    )


def test_mf1a_q4_baseline_regeneration_fails_on_manifest_version(
    mf1a_q4_baseline_regeneration_cases,
):
    prior_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    prior = mf1a.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40

    prior_mut = copy.deepcopy(prior)
    prior_mut["schema_version"] = "other"
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "bundle_structure"

    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][0]["manifest_schema_version"] = "other"
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "cases[0].manifest_schema_version"


def test_mf1a_q4_baseline_regeneration_fails_on_residual_above_comparator_despite_allowlist(
    mf1a_q4_baseline_regeneration_cases,
):
    prior_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    prior = mf1a.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40
    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][0]["qa001"]["frobenius_norm_diff"] += 2e-10
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[0].qa001.frobenius_norm_diff"
    )
    assert bundle["summary"]["first_failure"] == "regeneration"


def test_mf1a_q4_baseline_regeneration_passes_when_residual_within_comparator(
    mf1a_q4_baseline_regeneration_cases,
):
    prior_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    prior = mf1a.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40
    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][0]["qa001"]["frobenius_norm_diff"] += 5e-11
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["status"] == "pass"
    assert bundle["regeneration"]["pass"] is True
    assert bundle["regeneration"]["first_mismatch"] is None


def test_mf1a_q4_baseline_regeneration_rejects_non_revision_value(
    mf1a_q4_baseline_regeneration_cases,
):
    prior_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    prior = mf1a.build_artifact_bundle(prior_cases, prior_bundle=None)

    for bad_revision in ("g" * 40, "c" * 39, "", "C" * 40):
        current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
        current_cases[0]["provenance"]["implementation_revision"] = bad_revision
        bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior)
        assert bundle["regeneration"]["pass"] is False
        assert (
            bundle["regeneration"]["first_mismatch"]
            == "cases[0].provenance.implementation_revision"
        )

    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    del current_cases[0]["provenance"]["implementation_revision"]
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[0].provenance.implementation_revision"
    )

    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    current_cases[0]["provenance"]["implementation_revision"] = "c" * 40
    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][0]["provenance"]["implementation_revision"] = "A" * 40
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[0].provenance.implementation_revision"
    )

    prior_mut = copy.deepcopy(prior)
    del prior_mut["cases"][0]["provenance"]["implementation_revision"]
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[0].provenance.implementation_revision"
    )


def test_mf1a_q4_baseline_regeneration_allowlist_does_not_cover_a_second_case(
    mf1a_q4_baseline_regeneration_cases,
):
    prior_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    prior = mf1a.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(mf1a_q4_baseline_regeneration_cases)
    second_case = copy.deepcopy(current_cases[0])
    second_case["provenance"]["implementation_revision"] = "c" * 40
    current_cases.append(second_case)
    bundle = mf1a.build_artifact_bundle(current_cases, prior_bundle=prior)
    assert bundle["regeneration"]["first_mismatch"] == "manifest_exact_set"

    prior_two = copy.deepcopy(prior)
    prior_two["cases"].append(copy.deepcopy(prior_two["cases"][0]))
    result = mf1a._regeneration_result(current_cases[0], prior_two)
    assert result["first_mismatch"] == "bundle_structure"


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


_MF1A_HISTORICAL_ARTIFACT_DIR_NAMES: tuple[str, ...] = (
    "correctness_package",
    "output_integrity",
    "runtime_classification",
    "sequential_correctness",
    "external_correctness",
    "unsupported_boundary",
    "correctness_matrix",
    "summary_consistency",
)

_MF1A_HISTORICAL_BUNDLE_FILENAMES: dict[str, str] = {
    "correctness_package": "correctness_package_bundle.json",
    "output_integrity": "output_integrity_bundle.json",
    "runtime_classification": "runtime_classification_bundle.json",
    "sequential_correctness": "sequential_correctness_bundle.json",
    "external_correctness": "external_correctness_bundle.json",
    "unsupported_boundary": "unsupported_boundary_bundle.json",
    "correctness_matrix": "correctness_matrix_bundle.json",
    "summary_consistency": "summary_consistency_bundle.json",
}

_CORRECTNESS_EVIDENCE_ARTIFACT_ROOT = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "correctness_evidence"
)


def _mf1a_historical_repo_bundle_paths() -> tuple[Path, ...]:
    return tuple(
        _CORRECTNESS_EVIDENCE_ARTIFACT_ROOT
        / directory
        / _MF1A_HISTORICAL_BUNDLE_FILENAMES[directory]
        for directory in _MF1A_HISTORICAL_ARTIFACT_DIR_NAMES
    )


def _mf1a_historical_redirected_bundle_paths(fake_root: Path) -> tuple[Path, ...]:
    return tuple(
        fake_root / directory / _MF1A_HISTORICAL_BUNDLE_FILENAMES[directory]
        for directory in _MF1A_HISTORICAL_ARTIFACT_DIR_NAMES
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mf1a_historical_registry_modules() -> list:
    return [
        entry.module
        for entry in validation_pipeline._CASE_SLICE_REGISTRY
        + validation_pipeline._NULLARY_BUNDLE_REGISTRY
    ]


def _mf1a_historical_redirect_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_root = tmp_path / "redirected" / "correctness_evidence"
    fake_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(validation_pipeline, "DEFAULT_OUTPUT_ROOT", fake_root)
    monkeypatch.setattr(
        "benchmarks.density_matrix.correctness_evidence.common.DEFAULT_OUTPUT_ROOT",
        fake_root,
    )
    repo_artifact_root = _CORRECTNESS_EVIDENCE_ARTIFACT_ROOT
    for module in _mf1a_historical_registry_modules():
        relative_dir = module.DEFAULT_OUTPUT_DIR.relative_to(repo_artifact_root)
        monkeypatch.setattr(module, "DEFAULT_OUTPUT_DIR", fake_root / relative_dir)
    for module in _mf1a_historical_registry_modules():
        resolved = module.DEFAULT_OUTPUT_DIR.resolve()
        assert not resolved.is_relative_to(REPO_ROOT.resolve())
    return fake_root


class _Mf1aHistoricalBuilderCalled(RuntimeError):
    """Raised when a stub builder runs during a pre-build refusal check."""


def _mf1a_historical_stub_builders(
    monkeypatch: pytest.MonkeyPatch, *, fail_on_call: bool = False
) -> dict[str, int]:
    calls = {"build_cases": 0, "build_artifact_bundle": 0}

    def stub_cases() -> list:
        calls["build_cases"] += 1
        if fail_on_call:
            raise _Mf1aHistoricalBuilderCalled("build_cases must not run")
        return []

    def stub_bundle(_cases=None) -> dict:
        calls["build_artifact_bundle"] += 1
        if fail_on_call:
            raise _Mf1aHistoricalBuilderCalled("build_artifact_bundle must not run")
        return {"status": "pass", "summary": {}, "cases": []}

    for entry in validation_pipeline._CASE_SLICE_REGISTRY:
        monkeypatch.setattr(entry.module, entry.cases_attr, stub_cases)
        monkeypatch.setattr(entry.module, entry.bundle_attr, stub_bundle)
    for entry in validation_pipeline._NULLARY_BUNDLE_REGISTRY:
        monkeypatch.setattr(entry.module, entry.bundle_attr, stub_bundle)
    return calls


def test_mf1a_historical_suites_bytes_unchanged_after_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_root = _mf1a_historical_redirect_outputs(tmp_path, monkeypatch)
    redirected_paths = _mf1a_historical_redirected_bundle_paths(fake_root)
    for repo_path, redirected_path in zip(
        _mf1a_historical_repo_bundle_paths(), redirected_paths, strict=True
    ):
        redirected_path.parent.mkdir(parents=True, exist_ok=True)
        redirected_path.write_bytes(repo_path.read_bytes())

    before = {path: _sha256_file(path) for path in redirected_paths}
    _mf1a_historical_stub_builders(monkeypatch)
    validation_pipeline.run_pipeline()

    for path, digest in before.items():
        assert _sha256_file(path) == digest


def test_mf1a_historical_registered_siblings_still_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_root = _mf1a_historical_redirect_outputs(tmp_path, monkeypatch)
    _mf1a_historical_stub_builders(monkeypatch)
    validation_pipeline.run_pipeline()

    sibling_path = (
        fake_root / "mf1a" / "q4_baseline" / mf1a.ARTIFACT_FILENAME
    )
    assert sibling_path.is_file()


def test_mf1a_historical_all_statuses_returned_g07_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mf1a_historical_redirect_outputs(tmp_path, monkeypatch)
    _mf1a_historical_stub_builders(monkeypatch)
    results = validation_pipeline.run_pipeline()
    names = {name for name, _, _ in results}
    assert names == set(validation_pipeline.registered_suite_names())
    statuses = {name: status for name, status, _ in results}
    assert all(status == "pass" for status in statuses.values())
    assert validation_pipeline.g07_exit_passes(results)


def test_mf1a_historical_output_dir_refuses_in_repo_and_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    repo_root = validation_pipeline._find_repo_root()
    assert repo_root is not None
    in_repo = _CORRECTNESS_EVIDENCE_ARTIFACT_ROOT / "correctness_matrix"

    _mf1a_historical_redirect_outputs(tmp_path, monkeypatch)
    _mf1a_historical_stub_builders(monkeypatch, fail_on_call=True)
    with pytest.raises(SystemExit) as exc:
        validation_pipeline.main(["--historical-output-dir", str(in_repo)])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        captured.err
        == f"refused: --historical-output-dir resolves inside the repository ({in_repo.resolve()}; repo {repo_root.resolve()})\n"
    )

    outside = tmp_path / "outside_symlink_parent"
    outside.mkdir()
    symlink = outside / "into_repo"
    symlink.symlink_to(in_repo.resolve())
    _mf1a_historical_stub_builders(monkeypatch, fail_on_call=True)
    with pytest.raises(SystemExit) as exc:
        validation_pipeline.main(["--historical-output-dir", str(symlink)])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        captured.err
        == f"refused: --historical-output-dir resolves inside the repository ({symlink.resolve()}; repo {repo_root.resolve()})\n"
    )


def test_mf1a_historical_suite_without_sibling_field_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_root = _mf1a_historical_redirect_outputs(tmp_path, monkeypatch)
    _mf1a_historical_stub_builders(monkeypatch)
    validation_pipeline.run_pipeline()

    for directory in _MF1A_HISTORICAL_ARTIFACT_DIR_NAMES:
        bundle_path = fake_root / directory / _MF1A_HISTORICAL_BUNDLE_FILENAMES[directory]
        assert not bundle_path.exists()

    fake_module = SimpleNamespace(
        SUITE_NAME="correctness_evidence_mf1a_historical_fake_truthy_sibling",
        ARTIFACT_FILENAME="fake_truthy_sibling_bundle.json",
        DEFAULT_OUTPUT_DIR=fake_root / "fake_truthy_sibling",
    )
    fake_module.DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patched_registry = validation_pipeline._CASE_SLICE_REGISTRY + (
        validation_pipeline._CaseSuiteEntry(
            fake_module,
            "build_cases",
            "build_artifact_bundle",
            1,
        ),
    )
    monkeypatch.setattr(validation_pipeline, "_CASE_SLICE_REGISTRY", patched_registry)
    fake_module.build_cases = lambda: []
    fake_module.build_artifact_bundle = lambda _cases: {"status": "pass"}
    validation_pipeline.run_pipeline()
    assert not (fake_root / "fake_truthy_sibling" / fake_module.ARTIFACT_FILENAME).exists()


def test_mf1a_historical_nonsibling_paths_match_req007():
    repo_artifact_root = _CORRECTNESS_EVIDENCE_ARTIFACT_ROOT
    sibling_dirs: set[str] = set()
    nonsibling_dirs: set[str] = set()
    for entry in validation_pipeline._CASE_SLICE_REGISTRY:
        relative = entry.module.DEFAULT_OUTPUT_DIR.relative_to(repo_artifact_root)
        target = relative.as_posix()
        if entry.mf1a_sibling is True:
            sibling_dirs.add(target)
        else:
            nonsibling_dirs.add(target)
    for entry in validation_pipeline._NULLARY_BUNDLE_REGISTRY:
        relative = entry.module.DEFAULT_OUTPUT_DIR.relative_to(repo_artifact_root)
        if entry.mf1a_sibling is True:
            sibling_dirs.add(relative.as_posix())
        else:
            nonsibling_dirs.add(relative.as_posix())

    assert sibling_dirs == {"mf1a/q4_baseline"}
    assert nonsibling_dirs == set(_MF1A_HISTORICAL_ARTIFACT_DIR_NAMES)


def test_mf1a_historical_output_dir_writes_outside_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fake_root = _mf1a_historical_redirect_outputs(tmp_path, monkeypatch)
    _mf1a_historical_stub_builders(monkeypatch)
    historical_dir = tmp_path / "historical_mirror"
    historical_dir.mkdir()

    stale = historical_dir / "correctness_matrix" / "correctness_matrix_bundle.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"status": "stale"}', encoding="utf-8")

    exit_code = validation_pipeline.main(["--historical-output-dir", str(historical_dir)])
    assert exit_code == 0

    outside_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "verified, written outside repo" in line
    ]
    assert len(outside_lines) == len(_MF1A_HISTORICAL_ARTIFACT_DIR_NAMES)

    for directory in _MF1A_HISTORICAL_ARTIFACT_DIR_NAMES:
        bundle_path = (
            historical_dir / directory / _MF1A_HISTORICAL_BUNDLE_FILENAMES[directory]
        )
        assert bundle_path.is_file()
        assert bundle_path.read_text(encoding="utf-8") != '{"status": "stale"}'

    sibling_path = fake_root / "mf1a" / "q4_baseline" / mf1a.ARTIFACT_FILENAME
    assert sibling_path.is_file()

    written_files = list(historical_dir.rglob("*.json"))
    assert len(written_files) == len(_MF1A_HISTORICAL_ARTIFACT_DIR_NAMES)
    all_files = [path for path in historical_dir.rglob("*") if path.is_file()]
    assert len(all_files) == len(_MF1A_HISTORICAL_ARTIFACT_DIR_NAMES)


@pytest.fixture(autouse=True)
def _mf1a_historical_review_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional env-gated mutations for Reviewer red evidence (does not edit validation_pipeline.py)."""
    if os.environ.get("MF1A_MUTATION_A_WRITE_ALL") == "1":

        def _run_pipeline_write_all(*, historical_output_dir: Path | None = None):
            results: list[tuple[str, str, Path | None]] = []
            for entry in validation_pipeline._CASE_SLICE_REGISTRY:
                mod = entry.module
                cases = getattr(mod, entry.cases_attr)()
                bundle = getattr(mod, entry.bundle_attr)(cases)
                output_path = validation_pipeline._write_slice_bundle(mod, bundle)
                results.append((mod.SUITE_NAME, bundle["status"], output_path))
            for entry in validation_pipeline._NULLARY_BUNDLE_REGISTRY:
                mod = entry.module
                bundle = getattr(mod, entry.bundle_attr)()
                output_path = validation_pipeline._write_slice_bundle(mod, bundle)
                results.append((mod.SUITE_NAME, bundle["status"], output_path))
            return results

        monkeypatch.setattr(validation_pipeline, "run_pipeline", _run_pipeline_write_all)

    if os.environ.get("MF1A_MUTATION_D_NO_REFUSE") == "1":
        monkeypatch.setattr(
            validation_pipeline,
            "_validate_historical_output_dir",
            lambda raw: Path(raw).expanduser().resolve(strict=False),
        )
