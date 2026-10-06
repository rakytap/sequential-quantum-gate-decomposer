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


def _mf1a_fused_module():
    from benchmarks.density_matrix.correctness_evidence import (
        mf1a_fused_validation as fused_mod,
    )

    return fused_mod


def _clean_mf1a_fused_provenance() -> dict:
    fused = _mf1a_fused_module()
    return {
        "implementation_revision": "a" * 40,
        "clean_start": True,
        "dirty_paths": [],
        "command": fused.REGENERATION_COMMAND,
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


def _mf1a_fused_synthetic_case(
    fused,
    *,
    anchor_qbits: int,
    workload: str,
    fused_region_count: int,
    classifications: list[str],
) -> dict:
    route = fused.ROUTE
    return {
        "record_schema_version": fused.RECORD_SCHEMA_VERSION,
        "manifest_schema_version": fused.MANIFEST_SCHEMA_VERSION,
        "route": route,
        "anchor_qbits": anchor_qbits,
        "workload": workload,
        "planner_setting": {"max_partition_qubits": 2},
        "parameters": [0.0],
        "seed_policy": "deterministic_workload_no_random_seed",
        "realization": {
            "requested_path": route,
            "realized_path": route,
            "partition_count": 2,
            "exact_output_present": True,
            "actual_fused_execution": fused_region_count > 0,
            "fused_region_count": fused_region_count,
            "fused_region_classifications": classifications,
            "fused_regions": [],
        },
        "qa001": {
            "qa001_pass": True,
            "frobenius_norm_diff": 1e-14,
            "max_abs_diff": 1e-14,
            "trace_abs_deviation": 1e-14,
            "lambda_min": -1e-14,
        },
        "milestone_counted": False,
        "completeness_claim": False,
        "claim_boundary": fused.CLAIM_BOUNDARY,
        "provenance": _clean_mf1a_fused_provenance(),
    }


def test_mf1a_fused_manifest_is_the_four_frozen_ids():
    fused = _mf1a_fused_module()
    manifest = fused.build_manifest()
    assert manifest["schema_version"] == fused.MANIFEST_SCHEMA_VERSION
    cells = manifest["cells"]
    assert len(cells) == 4
    assert [cell["anchor_qbits"] for cell in cells] == [4, 6, 8, 10]
    assert [cell["workload"] for cell in cells] == [
        "phase2_xxz_hea_q4_continuity",
        "phase2_xxz_hea_q6_continuity",
        "layered_nearest_neighbor_q8_sparse_seed20260318",
        "layered_nearest_neighbor_q10_sparse_seed20260318",
    ]
    assert all(cell["route"] == fused.ROUTE for cell in cells)
    assert all(cell["max_partition_qubits"] == 2 for cell in cells)


def test_mf1a_fused_q8_q10_builder_calls_match_frozen_ids():
    from benchmarks.density_matrix.planner_surface import workloads

    fused = _mf1a_fused_module()
    assert fused.FROZEN_STRUCTURED_BUILDER_CALLS == (
        {
            "family_name": "layered_nearest_neighbor",
            "qbit_num": 8,
            "noise_pattern": "sparse",
            "seed": 20260318,
            "max_partition_qubits": 2,
        },
        {
            "family_name": "layered_nearest_neighbor",
            "qbit_num": 10,
            "noise_pattern": "sparse",
            "seed": 20260318,
            "max_partition_qubits": 2,
        },
    )
    for kwargs in fused.FROZEN_STRUCTURED_BUILDER_CALLS:
        descriptor_set = workloads.build_structured_descriptor_set(**kwargs)
        assert (
            descriptor_set.workload_id
            == f"layered_nearest_neighbor_q{kwargs['qbit_num']}_sparse_seed20260318"
        )


def test_mf1a_fused_realization_requires_a_fused_region():
    fused = _mf1a_fused_module()
    manifest_cells = fused.build_manifest()["cells"]
    good = _mf1a_fused_synthetic_case(
        fused,
        anchor_qbits=4,
        workload=manifest_cells[0]["workload"],
        fused_region_count=4,
        classifications=["actually_fused"],
    )
    assert fused._route_realization_pass(good)
    zero_fusion = _mf1a_fused_synthetic_case(
        fused,
        anchor_qbits=4,
        workload=manifest_cells[0]["workload"],
        fused_region_count=0,
        classifications=[],
    )
    assert not fused._route_realization_pass(zero_fusion)


def test_mf1a_fused_qa001_tolerances_match_q4():
    fused = _mf1a_fused_module()
    assert fused.evaluate_mf1a_qa001 is mf1a.evaluate_mf1a_qa001
    assert fused.MF1A_QA001_MATRIX_TOL is mf1a.MF1A_QA001_MATRIX_TOL
    assert fused.MF1A_QA001_LAMBDA_MIN_FLOOR is mf1a.MF1A_QA001_LAMBDA_MIN_FLOOR
    assert fused._QA001_REGENERATION_TOLERANCES is mf1a._QA001_REGENERATION_TOLERANCES
    assert fused._QA001_VALUE_KEYS is mf1a._QA001_VALUE_KEYS


def test_mf1a_fused_allowlist_is_length_four_and_q4_stays_length_one():
    fused = _mf1a_fused_module()
    assert len(fused.FUSED_REGENERATION_ALLOWLIST) == 4
    assert fused.FUSED_REGENERATION_ALLOWLIST == (
        "cases[0].provenance.implementation_revision",
        "cases[1].provenance.implementation_revision",
        "cases[2].provenance.implementation_revision",
        "cases[3].provenance.implementation_revision",
    )
    assert len(mf1a.Q4_REGENERATION_ALLOWLIST) == 1
    assert fused._allowlisted_revision_difference is not mf1a._allowlisted_revision_difference


def test_mf1a_fused_revision_only_passes():
    fused = _mf1a_fused_module()
    manifest_cells = fused.build_manifest()["cells"]
    prior_cases = [
        _mf1a_fused_synthetic_case(
            fused,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            fused_region_count=1,
            classifications=["actually_fused"],
        )
        for cell in manifest_cells
    ]
    for case in prior_cases:
        case["provenance"]["implementation_revision"] = "a" * 40
    prior = fused.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(prior_cases)
    for case in current_cases:
        case["provenance"]["implementation_revision"] = "b" * 40
    current = fused.build_artifact_bundle(current_cases, prior_bundle=prior)
    assert current["regeneration"]["pass"] is True

    uneven_current = copy.deepcopy(prior_cases)
    for index, case in enumerate(uneven_current):
        case["provenance"]["implementation_revision"] = f"{index:040x}"
    bundle = fused.build_artifact_bundle(uneven_current, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[0].provenance.implementation_revision"
    )

    single_case_current_moved = copy.deepcopy(prior_cases)
    single_case_current_moved[2]["provenance"]["implementation_revision"] = "d" * 40
    bundle = fused.build_artifact_bundle(single_case_current_moved, prior_bundle=prior)
    assert bundle["status"] == "fail"
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[2].provenance.implementation_revision"
    )

    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][1]["provenance"]["implementation_revision"] = "d" * 40
    current_cases = copy.deepcopy(prior_cases)
    bundle = fused.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["status"] == "fail"
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[1].provenance.implementation_revision"
    )

    for case_index in (1, 2, 3):
        provenance_fail_cases = copy.deepcopy(prior_cases)
        provenance_fail_cases[case_index]["provenance"]["provenance_pass"] = False
        bundle = fused.build_artifact_bundle(provenance_fail_cases, prior_bundle=None)
        assert bundle["status"] == "fail"
        assert bundle["summary"]["first_failure"] == "provenance"


def test_mf1a_fused_second_field_fails():
    fused = _mf1a_fused_module()
    manifest_cells = fused.build_manifest()["cells"]
    prior_cases = [
        _mf1a_fused_synthetic_case(
            fused,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            fused_region_count=1,
            classifications=["actually_fused"],
        )
        for cell in manifest_cells
    ]
    for case in prior_cases:
        case["provenance"]["implementation_revision"] = "a" * 40
    prior = fused.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(prior_cases)
    for case in current_cases:
        case["provenance"]["implementation_revision"] = "b" * 40
    current_cases[1]["workload"] = "substituted_workload"
    bundle = fused.build_artifact_bundle(current_cases, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "cases[1].workload"


def test_mf1a_fused_finding_bands_follow_layer1():
    fused = _mf1a_fused_module()
    assert fused.classify_near_threshold_measure("frobenius_norm_diff", 5e-14) == (
        "expected_range"
    )
    assert fused.classify_near_threshold_measure("max_abs_diff", 5e-12) == (
        "outside_expected"
    )
    assert fused.classify_near_threshold_measure("trace_abs_deviation", 5e-11) == (
        "finding"
    )
    assert fused.classify_near_threshold_measure("frobenius_norm_diff", 2e-10) == (
        "qa001_fail"
    )
    assert fused.classify_near_threshold_measure("lambda_min", -5e-14) == "no_finding"
    assert fused.classify_near_threshold_measure("lambda_min", -5e-13) == "finding"
    assert fused.classify_near_threshold_measure("lambda_min", -2e-12) == "qa001_fail"
    manifest_cells = fused.build_manifest()["cells"]
    lambda_finding_cases = [
        _mf1a_fused_synthetic_case(
            fused,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            fused_region_count=1,
            classifications=["actually_fused"],
        )
        for cell in manifest_cells
    ]
    lambda_finding_cases[0]["qa001"]["lambda_min"] = -5e-13
    bundle = fused.build_artifact_bundle(lambda_finding_cases, prior_bundle=None)
    assert bundle["status"] == "pass"
    assert bundle["summary"]["first_failure"] is None
    findings = bundle["summary"]["findings"]
    assert len(findings) == 1
    assert set(findings[0].keys()) == {
        "route",
        "anchor_qbits",
        "workload",
        "measure",
        "value",
        "cause_hypothesis",
    }
    assert findings[0] == {
        "route": fused.ROUTE,
        "anchor_qbits": 4,
        "workload": "phase2_xxz_hea_q4_continuity",
        "measure": "lambda_min",
        "value": -5e-13,
        "cause_hypothesis": "lambda_min below -1e-13 Layer 1 finding band",
    }


def test_mf1a_fused_rejects_non_revision_value():
    fused = _mf1a_fused_module()
    manifest_cells = fused.build_manifest()["cells"]
    prior_cases = [
        _mf1a_fused_synthetic_case(
            fused,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            fused_region_count=1,
            classifications=["actually_fused"],
        )
        for cell in manifest_cells
    ]
    for case in prior_cases:
        case["provenance"]["implementation_revision"] = "a" * 40
    prior = fused.build_artifact_bundle(prior_cases, prior_bundle=None)
    invalid_values = ("g" * 40, "c" * 39, "", "C" * 40)
    for value in invalid_values:
        current_cases = copy.deepcopy(prior_cases)
        current_cases[0]["provenance"]["implementation_revision"] = value
        bundle = fused.build_artifact_bundle(current_cases, prior_bundle=prior)
        assert bundle["regeneration"]["pass"] is False
        assert (
            bundle["regeneration"]["first_mismatch"]
            == "cases[0].provenance.implementation_revision"
        )
        prior_mut = copy.deepcopy(prior)
        prior_mut["cases"][1]["provenance"]["implementation_revision"] = value
        current_cases = copy.deepcopy(prior_cases)
        bundle = fused.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
        assert bundle["regeneration"]["pass"] is False
        assert (
            bundle["regeneration"]["first_mismatch"]
            == "cases[1].provenance.implementation_revision"
        )
    current_cases = copy.deepcopy(prior_cases)
    del current_cases[2]["provenance"]["implementation_revision"]
    bundle = fused.build_artifact_bundle(current_cases, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[2].provenance.implementation_revision"
    )
    prior_mut = copy.deepcopy(prior)
    del prior_mut["cases"][2]["provenance"]["implementation_revision"]
    current_cases = copy.deepcopy(prior_cases)
    bundle = fused.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[2].provenance.implementation_revision"
    )
    for value in invalid_values:
        current_uniform = copy.deepcopy(prior_cases)
        for case in current_uniform:
            case["provenance"]["implementation_revision"] = value
        bundle = fused.build_artifact_bundle(current_uniform, prior_bundle=prior)
        assert bundle["regeneration"]["pass"] is False
        assert (
            bundle["regeneration"]["first_mismatch"]
            == "cases[0].provenance.implementation_revision"
        )
        prior_mut = copy.deepcopy(prior)
        for case in prior_mut["cases"]:
            case["provenance"]["implementation_revision"] = value
        current_uniform = copy.deepcopy(prior_cases)
        bundle = fused.build_artifact_bundle(current_uniform, prior_bundle=prior_mut)
        assert bundle["regeneration"]["pass"] is False
        assert (
            bundle["regeneration"]["first_mismatch"]
            == "cases[0].provenance.implementation_revision"
        )
    current_uniform = copy.deepcopy(prior_cases)
    for case in current_uniform:
        del case["provenance"]["implementation_revision"]
    bundle = fused.build_artifact_bundle(current_uniform, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[0].provenance.implementation_revision"
    )
    prior_mut = copy.deepcopy(prior)
    for case in prior_mut["cases"]:
        del case["provenance"]["implementation_revision"]
    current_uniform = copy.deepcopy(prior_cases)
    bundle = fused.build_artifact_bundle(current_uniform, prior_bundle=prior_mut)
    assert bundle["regeneration"]["pass"] is False
    assert (
        bundle["regeneration"]["first_mismatch"]
        == "cases[0].provenance.implementation_revision"
    )


def test_mf1a_fused_rejects_case_count():
    fused = _mf1a_fused_module()
    manifest_cells = fused.build_manifest()["cells"]
    prior_cases = [
        _mf1a_fused_synthetic_case(
            fused,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            fused_region_count=1,
            classifications=["actually_fused"],
        )
        for cell in manifest_cells
    ]
    prior = fused.build_artifact_bundle(prior_cases, prior_bundle=None)
    too_many = copy.deepcopy(prior_cases)
    too_many.append(copy.deepcopy(too_many[0]))
    bundle = fused.build_artifact_bundle(too_many, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "manifest_exact_set"

    too_few = copy.deepcopy(prior_cases[:3])
    bundle = fused.build_artifact_bundle(too_few, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] in {
        "manifest_exact_set",
        "bundle_structure",
    }


def test_mf1a_fused_rejects_substituted_or_reordered_id():
    fused = _mf1a_fused_module()
    manifest_cells = fused.build_manifest()["cells"]
    prior_cases = [
        _mf1a_fused_synthetic_case(
            fused,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            fused_region_count=1,
            classifications=["actually_fused"],
        )
        for cell in manifest_cells
    ]
    prior = fused.build_artifact_bundle(prior_cases, prior_bundle=None)
    substituted = copy.deepcopy(prior_cases)
    substituted[2]["workload"] = "layered_nearest_neighbor_q8_sparse_seed99999999"
    bundle = fused.build_artifact_bundle(substituted, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "manifest_exact_set"

    reordered = copy.deepcopy(prior_cases)
    reordered[2], reordered[3] = reordered[3], reordered[2]
    bundle = fused.build_artifact_bundle(reordered, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "manifest_exact_set"


def test_mf1a_fused_pipeline_builds_every_sibling_before_writing_any(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events: list[tuple[str, str]] = []
    fake_root = tmp_path / "mf1a_fake_siblings"

    def make_fake_sibling(name: str) -> SimpleNamespace:
        module = SimpleNamespace(
            SUITE_NAME=f"correctness_evidence_mf1a_fake_{name}",
            ARTIFACT_FILENAME=f"fake_{name}_bundle.json",
            DEFAULT_OUTPUT_DIR=fake_root / name,
        )

        def build_cases() -> list:
            events.append(("build_cases", name))
            return []

        def build_artifact_bundle(_cases: list | None = None) -> dict:
            events.append(("build_artifact_bundle", name))
            return {"status": "pass", "cases": []}

        module.build_cases = build_cases
        module.build_artifact_bundle = build_artifact_bundle
        return module

    fake_a = make_fake_sibling("a")
    fake_b = make_fake_sibling("b")
    patched_registry = (
        validation_pipeline._CaseSuiteEntry(
            fake_a, "build_cases", "build_artifact_bundle", mf1a_sibling=True
        ),
        validation_pipeline._CaseSuiteEntry(
            fake_b, "build_cases", "build_artifact_bundle", mf1a_sibling=True
        ),
    )
    monkeypatch.setattr(validation_pipeline, "_CASE_SLICE_REGISTRY", patched_registry)
    monkeypatch.setattr(validation_pipeline, "_NULLARY_BUNDLE_REGISTRY", ())

    original_write = validation_pipeline._write_slice_bundle

    def log_write(module, bundle: dict) -> Path:
        events.append(("write", module.SUITE_NAME))
        return original_write(module, bundle)

    monkeypatch.setattr(validation_pipeline, "_write_slice_bundle", log_write)
    validation_pipeline.run_pipeline()

    first_write_index = next(
        (index for index, event in enumerate(events) if event[0] == "write"), None
    )
    build_indices = [
        index
        for index, event in enumerate(events)
        if event[0].startswith("build")
    ]
    assert first_write_index is not None
    assert build_indices
    assert all(build_index < first_write_index for build_index in build_indices)


def _mf1a_hybrid_module():
    from benchmarks.density_matrix.correctness_evidence import (
        mf1a_hybrid_validation as hybrid_mod,
    )

    return hybrid_mod


def _clean_mf1a_hybrid_provenance() -> dict:
    hybrid = _mf1a_hybrid_module()
    return {
        "implementation_revision": "a" * 40,
        "clean_start": True,
        "dirty_paths": [],
        "command": hybrid.REGENERATION_COMMAND,
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


def _q4_hybrid_positive_realization(hybrid) -> dict:
    partitions: list[dict] = []
    fused_regions: list[dict] = []
    for index in (0, 1):
        partitions.append(
            {
                "partition_index": index,
                "partition_runtime_class": "phase31_channel_native",
                "partition_route_reason": "eligible_channel_native_motif",
            }
        )
        fused_regions.append(
            {
                "partition_index": index,
                "candidate_kind": "channel_native_motif",
                "classification": "actually_fused",
                "reason": "eligible_channel_native_motif",
                "operation_names": ["U3"],
                "global_target_qbits": [0],
            }
        )
    for index in (2, 3, 4):
        partitions.append(
            {
                "partition_index": index,
                "partition_runtime_class": "phase3_unitary_island_fused",
                "partition_route_reason": "pure_unitary_partition",
            }
        )
        fused_regions.append(
            {
                "partition_index": index,
                "candidate_kind": "unitary_island",
                "classification": "actually_fused",
                "reason": "pure_unitary_partition",
                "operation_names": ["CNOT"],
                "global_target_qbits": [0, 1],
            }
        )
    return {
        "requested_path": hybrid.ROUTE,
        "realized_path": hybrid.ROUTE,
        "partition_count": 5,
        "exact_output_present": True,
        "channel_native_partition_count": 2,
        "runtime_class_counts": {
            "phase31_channel_native": 2,
            "phase3_unitary_island_fused": 3,
        },
        "route_reason_counts": {
            "eligible_channel_native_motif": 2,
            "pure_unitary_partition": 3,
        },
        "partitions": partitions,
        "fused_regions": fused_regions,
    }


def _recount_hybrid_realization(realization: dict) -> dict:
    from collections import Counter

    classes = [row["partition_runtime_class"] for row in realization["partitions"]]
    reasons = [row["partition_route_reason"] for row in realization["partitions"]]
    realization["runtime_class_counts"] = dict(sorted(Counter(classes).items()))
    realization["route_reason_counts"] = dict(sorted(Counter(reasons).items()))
    realization["channel_native_partition_count"] = classes.count("phase31_channel_native")
    return realization


def _mf1a_hybrid_synthetic_case(
    hybrid,
    *,
    anchor_qbits: int,
    workload: str,
    realization: dict,
    seed_policy: str = "deterministic_workload_no_random_seed",
) -> dict:
    return {
        "record_schema_version": hybrid.RECORD_SCHEMA_VERSION,
        "manifest_schema_version": hybrid.MANIFEST_SCHEMA_VERSION,
        "route": hybrid.ROUTE,
        "anchor_qbits": anchor_qbits,
        "workload": workload,
        "planner_setting": {"max_partition_qubits": 2},
        "parameters": [0.0],
        "seed_policy": seed_policy,
        "realization": realization,
        "qa001": {
            "qa001_pass": True,
            "frobenius_norm_diff": 1e-14,
            "max_abs_diff": 1e-14,
            "trace_abs_deviation": 1e-14,
            "lambda_min": -1e-14,
        },
        "milestone_counted": False,
        "completeness_claim": False,
        "claim_boundary": hybrid.CLAIM_BOUNDARY,
        "provenance": _clean_mf1a_hybrid_provenance(),
    }


def _assert_hybrid_regeneration_negative(bundle: dict, expected_mismatch: str) -> None:
    assert bundle["status"] == "fail"
    assert bundle["regeneration"]["pass"] is False
    assert bundle["summary"]["first_failure"] == "regeneration"
    assert bundle["regeneration"]["first_mismatch"] == expected_mismatch


def test_mf1a_hybrid_manifest_is_the_four_frozen_ids():
    hybrid = _mf1a_hybrid_module()
    manifest = hybrid.build_manifest()
    assert manifest["schema_version"] == hybrid.MANIFEST_SCHEMA_VERSION
    cells = manifest["cells"]
    assert len(cells) == 4
    assert [cell["anchor_qbits"] for cell in cells] == [4, 6, 8, 10]
    assert [cell["workload"] for cell in cells] == [
        "phase2_xxz_hea_q4_continuity",
        "phase2_xxz_hea_q6_continuity",
        "phase31_pair_repeat_q8_dense_seed20260318",
        "phase31_pair_repeat_q10_dense_seed20260318",
    ]
    assert all(cell["route"] == "phase31_channel_native_hybrid" for cell in cells)
    assert hybrid.ROUTE == "phase31_channel_native_hybrid"
    assert all(cell["max_partition_qubits"] == 2 for cell in cells)


def test_mf1a_hybrid_q8_q10_builder_calls_match_frozen_ids():
    from benchmarks.density_matrix.planner_surface import workloads

    hybrid = _mf1a_hybrid_module()
    assert hybrid.FROZEN_STRUCTURED_BUILDER_CALLS == (
        {
            "family_name": "phase31_pair_repeat",
            "qbit_num": 8,
            "noise_pattern": "dense",
            "seed": 20260318,
            "max_partition_qubits": 2,
        },
        {
            "family_name": "phase31_pair_repeat",
            "qbit_num": 10,
            "noise_pattern": "dense",
            "seed": 20260318,
            "max_partition_qubits": 2,
        },
    )
    for kwargs in hybrid.FROZEN_STRUCTURED_BUILDER_CALLS:
        descriptor_set = workloads.build_phase31_structured_descriptor_set(**kwargs)
        assert (
            descriptor_set.workload_id
            == f"phase31_pair_repeat_q{kwargs['qbit_num']}_dense_seed20260318"
        )


def test_mf1a_hybrid_realization_requires_a_channel_native_partition():
    hybrid = _mf1a_hybrid_module()
    manifest_cells = hybrid.build_manifest()["cells"]
    good_realization = _q4_hybrid_positive_realization(hybrid)
    good = _mf1a_hybrid_synthetic_case(
        hybrid,
        anchor_qbits=4,
        workload=manifest_cells[0]["workload"],
        realization=copy.deepcopy(good_realization),
    )
    assert hybrid._route_realization_pass(good)

    no_motif = copy.deepcopy(good_realization)
    no_motif["fused_regions"] = [
        region
        for region in no_motif["fused_regions"]
        if not (
            region["partition_index"] == 0
            and region["candidate_kind"] == "channel_native_motif"
        )
    ]
    assert not hybrid._route_realization_pass(
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=no_motif,
        )
    )

    motif_on_island = copy.deepcopy(good_realization)
    for partition in motif_on_island["partitions"]:
        if partition["partition_index"] == 2:
            partition["partition_runtime_class"] = "phase3_unitary_island_fused"
    motif_on_island["fused_regions"].append(
        {
            "partition_index": 2,
            "candidate_kind": "channel_native_motif",
            "classification": "actually_fused",
            "reason": "eligible_channel_native_motif",
            "operation_names": ["U3"],
            "global_target_qbits": [0],
        }
    )
    assert not hybrid._route_realization_pass(
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=motif_on_island,
        )
    )

    unknown_class = copy.deepcopy(good_realization)
    unknown_class["partitions"][2]["partition_runtime_class"] = "unknown_class"
    _recount_hybrid_realization(unknown_class)
    assert not hybrid._route_realization_pass(
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=unknown_class,
        )
    )

    bad_reason = copy.deepcopy(good_realization)
    bad_reason["partitions"][2]["partition_route_reason"] = "channel_native_noise_presence"
    _recount_hybrid_realization(bad_reason)
    assert not hybrid._route_realization_pass(
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=bad_reason,
        )
    )

    dropped_partition = copy.deepcopy(good_realization)
    dropped_partition["partitions"] = dropped_partition["partitions"][:-1]
    dropped_partition["fused_regions"] = [
        region
        for region in dropped_partition["fused_regions"]
        if region["partition_index"] != 4
    ]
    dropped_partition["partition_count"] = 5
    _recount_hybrid_realization(dropped_partition)
    assert not hybrid._route_realization_pass(
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=dropped_partition,
        )
    )

    duplicated_index = copy.deepcopy(good_realization)
    duplicated_index["partitions"][4]["partition_index"] = 3
    _recount_hybrid_realization(duplicated_index)
    assert not hybrid._route_realization_pass(
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=duplicated_index,
        )
    )

    zero_channel_native = copy.deepcopy(good_realization)
    for partition in zero_channel_native["partitions"]:
        if partition["partition_runtime_class"] == "phase31_channel_native":
            partition["partition_runtime_class"] = "phase3_unitary_island_fused"
            partition["partition_route_reason"] = "pure_unitary_partition"
    for region in zero_channel_native["fused_regions"]:
        if region["candidate_kind"] == "channel_native_motif":
            region["candidate_kind"] = "unitary_island"
            region["classification"] = "actually_fused"
            region["reason"] = "pure_unitary_partition"
    _recount_hybrid_realization(zero_channel_native)
    assert zero_channel_native["channel_native_partition_count"] == 0
    assert not hybrid._route_realization_pass(
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=zero_channel_native,
        )
    )

    unfused_with_fused_island = copy.deepcopy(good_realization)
    unfused_with_fused_island["partitions"][2]["partition_runtime_class"] = (
        "phase3_supported_unfused"
    )
    unfused_with_fused_island["partitions"][2]["partition_route_reason"] = (
        "channel_native_support_surface"
    )
    unfused_with_fused_island["fused_regions"].append(
        {
            "partition_index": 2,
            "candidate_kind": "unitary_island",
            "classification": "supported_but_unfused",
            "reason": "channel_native_support_surface",
            "operation_names": ["U3"],
            "global_target_qbits": [1],
        }
    )
    _recount_hybrid_realization(unfused_with_fused_island)
    assert not hybrid._route_realization_pass(
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=unfused_with_fused_island,
        )
    )

    orphan_region = copy.deepcopy(good_realization)
    orphan_region["fused_regions"].append(
        {
            "partition_index": 99,
            "candidate_kind": "unitary_island",
            "classification": "actually_fused",
            "reason": "pure_unitary_partition",
            "operation_names": ["CNOT"],
            "global_target_qbits": [0, 1],
        }
    )
    assert not hybrid._route_realization_pass(
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=orphan_region,
        )
    )

    wrong_cn_count = copy.deepcopy(good_realization)
    wrong_cn_count["channel_native_partition_count"] = 99
    assert not hybrid._route_realization_pass(
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=wrong_cn_count,
        )
    )


def test_mf1a_hybrid_qa001_tolerances_match_q4():
    hybrid = _mf1a_hybrid_module()
    assert hybrid.evaluate_mf1a_qa001 is mf1a.evaluate_mf1a_qa001
    assert hybrid.MF1A_QA001_MATRIX_TOL is mf1a.MF1A_QA001_MATRIX_TOL
    assert hybrid.MF1A_QA001_LAMBDA_MIN_FLOOR is mf1a.MF1A_QA001_LAMBDA_MIN_FLOOR
    assert hybrid._QA001_REGENERATION_TOLERANCES is mf1a._QA001_REGENERATION_TOLERANCES
    assert hybrid._QA001_VALUE_KEYS is mf1a._QA001_VALUE_KEYS


def test_mf1a_hybrid_rejects_case_count():
    hybrid = _mf1a_hybrid_module()
    manifest_cells = hybrid.build_manifest()["cells"]
    realization = _q4_hybrid_positive_realization(hybrid)
    prior_cases = [
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            realization=copy.deepcopy(realization),
        )
        for cell in manifest_cells
    ]
    prior = hybrid.build_artifact_bundle(prior_cases, prior_bundle=None)
    too_many = copy.deepcopy(prior_cases)
    too_many.append(copy.deepcopy(too_many[0]))
    bundle = hybrid.build_artifact_bundle(too_many, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "manifest_exact_set"

    too_few = copy.deepcopy(prior_cases[:3])
    bundle = hybrid.build_artifact_bundle(too_few, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] in {
        "manifest_exact_set",
        "bundle_structure",
    }


def test_mf1a_hybrid_rejects_substituted_or_reordered_id():
    hybrid = _mf1a_hybrid_module()
    manifest_cells = hybrid.build_manifest()["cells"]
    realization = _q4_hybrid_positive_realization(hybrid)
    prior_cases = [
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            realization=copy.deepcopy(realization),
        )
        for cell in manifest_cells
    ]
    prior = hybrid.build_artifact_bundle(prior_cases, prior_bundle=None)
    substituted = copy.deepcopy(prior_cases)
    substituted[2]["workload"] = "phase31_pair_repeat_q8_dense_seed99999999"
    bundle = hybrid.build_artifact_bundle(substituted, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "manifest_exact_set"

    reordered = copy.deepcopy(prior_cases)
    reordered[2], reordered[3] = reordered[3], reordered[2]
    bundle = hybrid.build_artifact_bundle(reordered, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "manifest_exact_set"


def _mf1a_hybrid_uniform_prior_cases(hybrid):
    manifest_cells = hybrid.build_manifest()["cells"]
    realization = _q4_hybrid_positive_realization(hybrid)
    prior_cases = [
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            realization=copy.deepcopy(realization),
        )
        for cell in manifest_cells
    ]
    for case in prior_cases:
        case["provenance"]["implementation_revision"] = "a" * 40
    return prior_cases


def test_mf1a_hybrid_one_current_revision_diverges():
    hybrid = _mf1a_hybrid_module()
    prior_cases = _mf1a_hybrid_uniform_prior_cases(hybrid)
    prior = hybrid.build_artifact_bundle(prior_cases, prior_bundle=None)

    rc5_current = copy.deepcopy(prior_cases)
    for case in rc5_current:
        case["provenance"]["implementation_revision"] = "a" * 40
    rc5_current[2]["provenance"]["implementation_revision"] = "d" * 40
    bundle = hybrid.build_artifact_bundle(rc5_current, prior_bundle=prior)
    _assert_hybrid_regeneration_negative(
        bundle, "cases[2].provenance.implementation_revision"
    )

    all_moved_current = copy.deepcopy(prior_cases)
    revisions = ["c" * 40, "d" * 40, "c" * 40, "c" * 40]
    for case, revision in zip(all_moved_current, revisions, strict=True):
        case["provenance"]["implementation_revision"] = revision
    bundle = hybrid.build_artifact_bundle(all_moved_current, prior_bundle=prior)
    _assert_hybrid_regeneration_negative(
        bundle, "cases[0].provenance.implementation_revision"
    )


def test_mf1a_hybrid_one_prior_revision_diverges():
    hybrid = _mf1a_hybrid_module()
    prior_cases = _mf1a_hybrid_uniform_prior_cases(hybrid)
    for case in prior_cases:
        case["provenance"]["implementation_revision"] = "a" * 40
    prior = hybrid.build_artifact_bundle(prior_cases, prior_bundle=None)

    rc6_current = copy.deepcopy(prior_cases)
    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][1]["provenance"]["implementation_revision"] = "d" * 40
    bundle = hybrid.build_artifact_bundle(rc6_current, prior_bundle=prior_mut)
    _assert_hybrid_regeneration_negative(
        bundle, "cases[1].provenance.implementation_revision"
    )

    all_moved_prior = copy.deepcopy(prior)
    for case in rc6_current:
        case["provenance"]["implementation_revision"] = "c" * 40
    prior_revisions = ["a" * 40, "a" * 40, "b" * 40, "a" * 40]
    for case, revision in zip(all_moved_prior["cases"], prior_revisions, strict=True):
        case["provenance"]["implementation_revision"] = revision
    bundle = hybrid.build_artifact_bundle(rc6_current, prior_bundle=all_moved_prior)
    _assert_hybrid_regeneration_negative(
        bundle, "cases[0].provenance.implementation_revision"
    )


def test_mf1a_hybrid_one_case_non_allowlisted_path():
    hybrid = _mf1a_hybrid_module()
    prior_cases = _mf1a_hybrid_uniform_prior_cases(hybrid)
    prior = hybrid.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(prior_cases)
    for case in current_cases:
        case["provenance"]["implementation_revision"] = "c" * 40
    current_cases[1]["seed_policy"] = "mutated_seed_policy"
    bundle = hybrid.build_artifact_bundle(current_cases, prior_bundle=prior)
    _assert_hybrid_regeneration_negative(bundle, "cases[1].seed_policy")


def test_mf1a_hybrid_swapped_two_cases_revisions_only():
    hybrid = _mf1a_hybrid_module()
    prior_cases = _mf1a_hybrid_uniform_prior_cases(hybrid)
    for case in prior_cases:
        case["provenance"]["implementation_revision"] = "c" * 40
    prior = hybrid.build_artifact_bundle(prior_cases, prior_bundle=None)
    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][0]["provenance"]["implementation_revision"] = "a" * 40
    prior_mut["cases"][1]["provenance"]["implementation_revision"] = "b" * 40
    prior_mut["cases"][2]["provenance"]["implementation_revision"] = "a" * 40
    prior_mut["cases"][3]["provenance"]["implementation_revision"] = "a" * 40
    current_cases = copy.deepcopy(prior_cases)
    bundle = hybrid.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    _assert_hybrid_regeneration_negative(
        bundle, "cases[0].provenance.implementation_revision"
    )
    swapped = copy.deepcopy(prior_mut)
    swapped["cases"][0]["provenance"]["implementation_revision"] = "b" * 40
    swapped["cases"][1]["provenance"]["implementation_revision"] = "a" * 40
    bundle = hybrid.build_artifact_bundle(current_cases, prior_bundle=swapped)
    _assert_hybrid_regeneration_negative(
        bundle, "cases[0].provenance.implementation_revision"
    )


def test_mf1a_hybrid_empty_allowlist_fails(monkeypatch: pytest.MonkeyPatch):
    hybrid = _mf1a_hybrid_module()
    monkeypatch.setattr(hybrid, "HYBRID_REGENERATION_ALLOWLIST", ())
    prior_cases = _mf1a_hybrid_uniform_prior_cases(hybrid)
    prior = hybrid.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(prior_cases)
    for case in current_cases:
        case["provenance"]["implementation_revision"] = "c" * 40
    bundle = hybrid.build_artifact_bundle(current_cases, prior_bundle=prior)
    _assert_hybrid_regeneration_negative(
        bundle, "cases[0].provenance.implementation_revision"
    )


def test_mf1a_hybrid_length_three_allowlist_fails(monkeypatch: pytest.MonkeyPatch):
    hybrid = _mf1a_hybrid_module()
    monkeypatch.setattr(
        hybrid,
        "HYBRID_REGENERATION_ALLOWLIST",
        hybrid.HYBRID_REGENERATION_ALLOWLIST[:3],
    )
    prior_cases = _mf1a_hybrid_uniform_prior_cases(hybrid)
    prior = hybrid.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(prior_cases)
    for case in current_cases:
        case["provenance"]["implementation_revision"] = "c" * 40
    bundle = hybrid.build_artifact_bundle(current_cases, prior_bundle=prior)
    _assert_hybrid_regeneration_negative(
        bundle, "cases[3].provenance.implementation_revision"
    )


def test_mf1a_hybrid_revision_only_passes():
    hybrid = _mf1a_hybrid_module()
    prior_cases = _mf1a_hybrid_uniform_prior_cases(hybrid)
    prior = hybrid.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(prior_cases)
    for case in current_cases:
        case["provenance"]["implementation_revision"] = "c" * 40
    bundle = hybrid.build_artifact_bundle(current_cases, prior_bundle=prior)
    assert bundle["status"] == "pass"
    assert bundle["regeneration"] == {
        "prior_present": True,
        "pass": True,
        "first_mismatch": None,
    }
    for case_index in (1, 2, 3):
        provenance_fail_cases = copy.deepcopy(prior_cases)
        provenance_fail_cases[case_index]["provenance"]["provenance_pass"] = False
        bundle = hybrid.build_artifact_bundle(provenance_fail_cases, prior_bundle=None)
        assert bundle["status"] == "fail"
        assert bundle["summary"]["first_failure"] == "provenance"


def test_mf1a_hybrid_rejects_non_revision_value():
    hybrid = _mf1a_hybrid_module()
    prior_cases = _mf1a_hybrid_uniform_prior_cases(hybrid)
    prior = hybrid.build_artifact_bundle(prior_cases, prior_bundle=None)
    invalid_values = ("g" * 40, "c" * 39, "", "C" * 40)
    for value in invalid_values:
        current_cases = copy.deepcopy(prior_cases)
        current_cases[0]["provenance"]["implementation_revision"] = value
        bundle = hybrid.build_artifact_bundle(current_cases, prior_bundle=prior)
        _assert_hybrid_regeneration_negative(
            bundle, "cases[0].provenance.implementation_revision"
        )
        prior_mut = copy.deepcopy(prior)
        prior_mut["cases"][1]["provenance"]["implementation_revision"] = value
        current_cases = copy.deepcopy(prior_cases)
        bundle = hybrid.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
        _assert_hybrid_regeneration_negative(
            bundle, "cases[1].provenance.implementation_revision"
        )
    current_cases = copy.deepcopy(prior_cases)
    del current_cases[2]["provenance"]["implementation_revision"]
    bundle = hybrid.build_artifact_bundle(current_cases, prior_bundle=prior)
    _assert_hybrid_regeneration_negative(
        bundle, "cases[2].provenance.implementation_revision"
    )
    prior_mut = copy.deepcopy(prior)
    del prior_mut["cases"][2]["provenance"]["implementation_revision"]
    current_cases = copy.deepcopy(prior_cases)
    bundle = hybrid.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    _assert_hybrid_regeneration_negative(
        bundle, "cases[2].provenance.implementation_revision"
    )
    for value in invalid_values:
        current_uniform = copy.deepcopy(prior_cases)
        for case in current_uniform:
            case["provenance"]["implementation_revision"] = value
        bundle = hybrid.build_artifact_bundle(current_uniform, prior_bundle=prior)
        _assert_hybrid_regeneration_negative(
            bundle, "cases[0].provenance.implementation_revision"
        )
        prior_mut = copy.deepcopy(prior)
        for case in prior_mut["cases"]:
            case["provenance"]["implementation_revision"] = value
        current_uniform = copy.deepcopy(prior_cases)
        bundle = hybrid.build_artifact_bundle(current_uniform, prior_bundle=prior_mut)
        _assert_hybrid_regeneration_negative(
            bundle, "cases[0].provenance.implementation_revision"
        )
    current_uniform = copy.deepcopy(prior_cases)
    for case in current_uniform:
        del case["provenance"]["implementation_revision"]
    bundle = hybrid.build_artifact_bundle(current_uniform, prior_bundle=prior)
    _assert_hybrid_regeneration_negative(
        bundle, "cases[0].provenance.implementation_revision"
    )
    prior_mut = copy.deepcopy(prior)
    for case in prior_mut["cases"]:
        del case["provenance"]["implementation_revision"]
    current_uniform = copy.deepcopy(prior_cases)
    bundle = hybrid.build_artifact_bundle(current_uniform, prior_bundle=prior_mut)
    _assert_hybrid_regeneration_negative(
        bundle, "cases[0].provenance.implementation_revision"
    )


def test_mf1a_hybrid_finding_bands_follow_layer1():
    import json

    hybrid = _mf1a_hybrid_module()
    assert hybrid.classify_near_threshold_measure("frobenius_norm_diff", 5e-14) == (
        "expected_range"
    )
    assert hybrid.classify_near_threshold_measure("max_abs_diff", 5e-12) == (
        "outside_expected"
    )
    assert hybrid.classify_near_threshold_measure("trace_abs_deviation", 5e-11) == (
        "finding"
    )
    assert hybrid.classify_near_threshold_measure("frobenius_norm_diff", 2e-10) == (
        "qa001_fail"
    )
    assert hybrid.classify_near_threshold_measure("lambda_min", -5e-14) == "no_finding"
    assert hybrid.classify_near_threshold_measure("lambda_min", -5e-13) == "finding"
    assert hybrid.classify_near_threshold_measure("lambda_min", -2e-12) == "qa001_fail"
    manifest_cells = hybrid.build_manifest()["cells"]
    realization = _q4_hybrid_positive_realization(hybrid)
    lambda_finding_cases = [
        _mf1a_hybrid_synthetic_case(
            hybrid,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            realization=copy.deepcopy(realization),
        )
        for cell in manifest_cells
    ]
    lambda_finding_cases[0]["qa001"]["lambda_min"] = -5e-13
    bundle = hybrid.build_artifact_bundle(lambda_finding_cases, prior_bundle=None)
    assert bundle["status"] == "pass"
    assert bundle["summary"]["first_failure"] is None
    findings = bundle["summary"]["findings"]
    assert len(findings) == 1
    assert findings[0] == {
        "route": hybrid.ROUTE,
        "anchor_qbits": 4,
        "workload": "phase2_xxz_hea_q4_continuity",
        "measure": "lambda_min",
        "value": -5e-13,
        "cause_hypothesis": "lambda_min below -1e-13 Layer 1 finding band",
    }
    assert "oracle_lambda_min" not in json.dumps(bundle)


def test_mf1a_hybrid_second_field_fails():
    hybrid = _mf1a_hybrid_module()
    prior_cases = _mf1a_hybrid_uniform_prior_cases(hybrid)
    prior = hybrid.build_artifact_bundle(prior_cases, prior_bundle=None)
    current_cases = copy.deepcopy(prior_cases)
    for case in current_cases:
        case["provenance"]["implementation_revision"] = "c" * 40
    current_cases[1]["workload"] = "substituted_workload"
    bundle = hybrid.build_artifact_bundle(current_cases, prior_bundle=prior)
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "cases[1].workload"


def test_mf1a_hybrid_allowlist_is_length_four_and_siblings_stay():
    hybrid = _mf1a_hybrid_module()
    fused = _mf1a_fused_module()
    assert hybrid.HYBRID_REGENERATION_ALLOWLIST == (
        "cases[0].provenance.implementation_revision",
        "cases[1].provenance.implementation_revision",
        "cases[2].provenance.implementation_revision",
        "cases[3].provenance.implementation_revision",
    )
    assert len(hybrid.HYBRID_REGENERATION_ALLOWLIST) == 4
    assert len(mf1a.Q4_REGENERATION_ALLOWLIST) == 1
    assert len(fused.FUSED_REGENERATION_ALLOWLIST) == 4
    assert hybrid._allowlisted_revision_difference is not mf1a._allowlisted_revision_difference
    assert hybrid._allowlisted_revision_difference is not fused._allowlisted_revision_difference


def test_mf1a_hybrid_pipeline_builds_every_sibling_before_writing_any(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    hybrid = _mf1a_hybrid_module()
    fused = _mf1a_fused_module()
    from benchmarks.density_matrix.correctness_evidence import (
        mf1a_q4_baseline_validation as q4_mod,
    )

    live = validation_pipeline._CASE_SLICE_REGISTRY[0:3]
    assert len(live) == 3
    assert live[0].module is q4_mod
    assert live[0].mf1a_sibling is True
    assert live[1].module is fused
    assert live[1].mf1a_sibling is True
    assert live[2].module is hybrid
    assert live[2].mf1a_sibling is True

    events: list[tuple[str, str]] = []
    fake_root = tmp_path / "mf1a_fake_siblings"

    def make_fake_sibling(name: str) -> SimpleNamespace:
        module = SimpleNamespace(
            SUITE_NAME=f"correctness_evidence_mf1a_fake_{name}",
            ARTIFACT_FILENAME=f"fake_{name}_bundle.json",
            DEFAULT_OUTPUT_DIR=fake_root / name,
        )

        def build_cases() -> list:
            events.append(("build_cases", name))
            return []

        def build_artifact_bundle(_cases: list | None = None) -> dict:
            events.append(("build_artifact_bundle", name))
            return {"status": "pass", "cases": []}

        module.build_cases = build_cases
        module.build_artifact_bundle = build_artifact_bundle
        return module

    fake_a = make_fake_sibling("a")
    fake_b = make_fake_sibling("b")
    fake_c = make_fake_sibling("c")
    patched_registry = (
        validation_pipeline._CaseSuiteEntry(
            fake_a, "build_cases", "build_artifact_bundle", mf1a_sibling=True
        ),
        validation_pipeline._CaseSuiteEntry(
            fake_b, "build_cases", "build_artifact_bundle", mf1a_sibling=True
        ),
        validation_pipeline._CaseSuiteEntry(
            fake_c, "build_cases", "build_artifact_bundle", mf1a_sibling=True
        ),
    )
    monkeypatch.setattr(validation_pipeline, "_CASE_SLICE_REGISTRY", patched_registry)
    monkeypatch.setattr(validation_pipeline, "_NULLARY_BUNDLE_REGISTRY", ())

    original_write = validation_pipeline._write_slice_bundle

    def log_write(module, bundle: dict) -> Path:
        events.append(("write", module.SUITE_NAME))
        return original_write(module, bundle)

    monkeypatch.setattr(validation_pipeline, "_write_slice_bundle", log_write)
    validation_pipeline.run_pipeline()

    first_write_index = next(
        (index for index, event in enumerate(events) if event[0] == "write"), None
    )
    build_indices = [
        index
        for index, event in enumerate(events)
        if event[0].startswith("build")
    ]
    assert first_write_index is not None
    assert build_indices
    assert all(build_index < first_write_index for build_index in build_indices)



def _mf1a_strict_module():
    from benchmarks.density_matrix.correctness_evidence import (
        mf1a_strict_validation as strict_mod,
    )

    return strict_mod


def _clean_mf1a_strict_provenance() -> dict:
    strict = _mf1a_strict_module()
    return {
        "implementation_revision": "a" * 40,
        "clean_start": True,
        "dirty_paths": [],
        "command": strict.REGENERATION_COMMAND,
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


def _mf1a_strict_pair_witness(partition_index: int) -> dict:
    base = 2 * partition_index
    return {
        "partition_index": partition_index,
        "candidate_kind": "channel_native_motif",
        "classification": "actually_fused",
        "reason": "channel_native_motif_kraus_count_4",
        "operation_names": [
            "U3",
            "U3",
            "CNOT",
            "amplitude_damping",
            "phase_damping",
            "U3",
        ],
        "global_target_qbits": [base, base + 1],
    }


def _mf1a_strict_frozen_realization(*, partition_count: int) -> dict:
    return {
        "requested_path": "phase31_channel_native",
        "realized_path": "phase31_channel_native",
        "partition_count": partition_count,
        "exact_output_present": True,
        "channel_native_partition_count": partition_count,
        "partitions": [{"partition_index": index} for index in range(partition_count)],
        "fused_regions": [
            _mf1a_strict_pair_witness(index) for index in range(partition_count)
        ],
    }


def _mf1a_strict_synthetic_case(
    strict,
    *,
    anchor_qbits: int,
    workload: str,
    realization: dict,
    route: str | None = None,
    max_partition_qubits: int = 2,
    seed_policy: str = "deterministic_workload_no_random_seed",
) -> dict:
    return {
        "record_schema_version": strict.RECORD_SCHEMA_VERSION,
        "manifest_schema_version": strict.MANIFEST_SCHEMA_VERSION,
        "route": route or strict.ROUTE,
        "anchor_qbits": anchor_qbits,
        "workload": workload,
        "planner_setting": {"max_partition_qubits": max_partition_qubits},
        "parameters": [0.0],
        "seed_policy": seed_policy,
        "realization": realization,
        "qa001": {
            "qa001_pass": True,
            "frobenius_norm_diff": 1e-14,
            "max_abs_diff": 1e-14,
            "trace_abs_deviation": 1e-14,
            "lambda_min": -1e-14,
        },
        "milestone_counted": False,
        "completeness_claim": False,
        "claim_boundary": strict.CLAIM_BOUNDARY,
        "provenance": _clean_mf1a_strict_provenance(),
    }


def _assert_strict_regeneration_negative(bundle: dict, expected_mismatch: str) -> None:
    assert bundle["status"] == "fail"
    assert bundle["regeneration"]["pass"] is False
    assert bundle["summary"]["first_failure"] == "regeneration"
    assert bundle["regeneration"]["first_mismatch"] == expected_mismatch


def _assert_strict_manifest_exact_set(bundle: dict) -> None:
    assert bundle["status"] == "fail"
    assert bundle["regeneration"]["pass"] is False
    assert bundle["summary"]["first_failure"] == "manifest_exact_set"
    assert bundle["regeneration"]["first_mismatch"] == "manifest_exact_set"


def test_mf1a_strict_manifest_is_the_four_frozen_ids():
    strict = _mf1a_strict_module()
    manifest = strict.build_manifest()
    assert manifest["schema_version"] == strict.MANIFEST_SCHEMA_VERSION
    cells = manifest["cells"]
    assert len(cells) == 4
    assert [cell["anchor_qbits"] for cell in cells] == [4, 6, 8, 10]
    assert [cell["workload"] for cell in cells] == [
        "phase31_local_support_q4_spectator_embedding_smoke",
        "mf1a_strict_spectator_embed_q6",
        "mf1a_strict_spectator_embed_q8",
        "mf1a_strict_spectator_embed_q10",
    ]
    assert all(cell["route"] == "phase31_channel_native" for cell in cells)
    assert strict.ROUTE == "phase31_channel_native"
    assert all(cell["max_partition_qubits"] == 2 for cell in cells)


def test_mf1a_strict_builder_calls_match_frozen_ids(monkeypatch: pytest.MonkeyPatch):
    from benchmarks.density_matrix.planner_surface import workloads

    strict = _mf1a_strict_module()
    calls: list[tuple] = []

    original_build = workloads.build_phase31_microcase_descriptor_set

    def _spy(case_name: str, *, max_partition_qubits: int):
        calls.append((case_name, max_partition_qubits))
        return original_build(case_name, max_partition_qubits=max_partition_qubits)

    monkeypatch.setattr(workloads, "build_phase31_microcase_descriptor_set", _spy)
    q4_cell = strict.build_manifest()["cells"][0]
    descriptor = strict._build_cell_descriptor(q4_cell)
    assert calls == [
        ("phase31_local_support_q4_spectator_embedding_smoke", 2),
    ]
    assert descriptor.source_type == "microcase_builder"

    smoke_specs = next(
        case["operation_specs"]
        for case in workloads.phase31_microcase_definitions()
        if case["case_name"] == "phase31_local_support_q4_spectator_embedding_smoke"
    )
    assert strict.mf1a_strict_spectator_operation_specs(4) == smoke_specs

    def _expected_specs(qbit_num: int) -> list[dict]:
        specs: list[dict] = []
        for pair_index in range(qbit_num // 2):
            control = 2 * pair_index
            target = control + 1
            gate_index = 4 * pair_index + 2
            specs.extend(
                (
                    {"kind": "gate", "name": "U3", "target_qbit": control, "param_count": 3},
                    {"kind": "gate", "name": "U3", "target_qbit": target, "param_count": 3},
                    {
                        "kind": "gate",
                        "name": "CNOT",
                        "target_qbit": target,
                        "control_qbit": control,
                        "param_count": 0,
                    },
                    {
                        "kind": "noise",
                        "name": "amplitude_damping",
                        "target_qbit": target,
                        "source_gate_index": gate_index,
                        "fixed_value": 0.05,
                        "param_count": 0,
                    },
                    {
                        "kind": "noise",
                        "name": "phase_damping",
                        "target_qbit": control,
                        "source_gate_index": gate_index,
                        "fixed_value": 0.07,
                        "param_count": 0,
                    },
                    {"kind": "gate", "name": "U3", "target_qbit": control, "param_count": 3},
                )
            )
        return specs

    for qbit_num in (6, 8, 10):
        assert strict.mf1a_strict_spectator_operation_specs(qbit_num) == _expected_specs(
            qbit_num
        )
        descriptor_set = strict.build_mf1a_strict_spectator_descriptor_set(qbit_num)
        assert descriptor_set.workload_id == f"mf1a_strict_spectator_embed_q{qbit_num}"
        assert descriptor_set.source_type == "structured_family_builder"
        assert descriptor_set.max_partition_qubits == 2
        assert descriptor_set.parameter_count == 9 * qbit_num // 2
        assert len(descriptor_set.partitions) == qbit_num // 2
        for partition_index, partition in enumerate(descriptor_set.partitions):
            assert partition.local_to_global_qbits == (
                2 * partition_index,
                2 * partition_index + 1,
            )
            assert [
                descriptor_set.canonical_operation_for(member).name
                for member in partition.members
            ] == [
                "U3",
                "U3",
                "CNOT",
                "amplitude_damping",
                "phase_damping",
                "U3",
            ]

    with pytest.raises(ValueError):
        strict.mf1a_strict_spectator_operation_specs(3)
    with pytest.raises(ValueError):
        strict.mf1a_strict_spectator_operation_specs(1)

    from benchmarks.density_matrix.partitioned_runtime.common import (
        build_initial_parameters as expected_builder,
    )

    assert strict.build_initial_parameters is expected_builder
    for cell in strict.build_manifest()["cells"]:
        assert strict._seed_policy_for_cell(cell) == "deterministic_workload_no_random_seed"


def test_mf1a_strict_realization_positive_is_the_q4_smoke_shape():
    from squander.partitioning.noisy_runtime import execute_partitioned_density_channel_native

    strict = _mf1a_strict_module()
    manifest_cells = strict.build_manifest()["cells"]
    q4_expected = _mf1a_strict_frozen_realization(partition_count=2)
    q6_expected = _mf1a_strict_frozen_realization(partition_count=3)

    for cell, expected in (
        (manifest_cells[0], q4_expected),
        (manifest_cells[1], q6_expected),
    ):
        descriptor_set = strict._build_cell_descriptor(cell)
        parameters = strict.build_initial_parameters(descriptor_set.parameter_count)
        result = execute_partitioned_density_channel_native(descriptor_set, parameters)
        realization = strict._build_realization(result, descriptor_set)
        assert realization == expected
        case = _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            realization=realization,
        )
        assert strict._route_realization_pass(case)


def test_mf1a_strict_dropped_row_is_md():
    strict = _mf1a_strict_module()
    manifest_cells = strict.build_manifest()["cells"]
    realization = _mf1a_strict_frozen_realization(partition_count=2)
    realization["partitions"] = [realization["partitions"][0]]
    case = _mf1a_strict_synthetic_case(
        strict,
        anchor_qbits=4,
        workload=manifest_cells[0]["workload"],
        realization=realization,
    )
    assert not strict._route_realization_pass(case)


def test_mf1a_strict_duplicate_index_is_me():
    strict = _mf1a_strict_module()
    manifest_cells = strict.build_manifest()["cells"]
    realization = _mf1a_strict_frozen_realization(partition_count=2)
    realization["partitions"][1]["partition_index"] = 0
    case = _mf1a_strict_synthetic_case(
        strict,
        anchor_qbits=4,
        workload=manifest_cells[0]["workload"],
        realization=realization,
    )
    assert not strict._route_realization_pass(case)


def test_mf1a_strict_extra_row_kills_length_and_range_removal():
    strict = _mf1a_strict_module()
    manifest_cells = strict.build_manifest()["cells"]
    base = _mf1a_strict_frozen_realization(partition_count=2)

    extra_row = copy.deepcopy(base)
    extra_row["partitions"].append({"partition_index": 1})
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=extra_row,
        )
    )

    out_of_range = copy.deepcopy(base)
    out_of_range["partitions"][1]["partition_index"] = 2
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=out_of_range,
        )
    )


def test_mf1a_strict_motif_removed_fails():
    strict = _mf1a_strict_module()
    manifest_cells = strict.build_manifest()["cells"]
    base = _mf1a_strict_frozen_realization(partition_count=2)

    removed = copy.deepcopy(base)
    removed["fused_regions"] = [removed["fused_regions"][0]]
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=removed,
        )
    )

    duplicate = copy.deepcopy(base)
    duplicate["fused_regions"].append(copy.deepcopy(duplicate["fused_regions"][0]))
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=duplicate,
        )
    )

    orphan = copy.deepcopy(base)
    orphan["fused_regions"].append(
        {
            "partition_index": 99,
            "candidate_kind": "channel_native_motif",
            "classification": "actually_fused",
            "reason": "channel_native_motif_kraus_count_4",
            "operation_names": ["U3"],
            "global_target_qbits": [0, 1],
        }
    )
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=orphan,
        )
    )


def test_mf1a_strict_vocabulary_fails():
    strict = _mf1a_strict_module()
    manifest_cells = strict.build_manifest()["cells"]
    base = _mf1a_strict_frozen_realization(partition_count=2)

    label_key = copy.deepcopy(base)
    label_key["partitions"][0]["partition_runtime_class"] = "phase31_channel_native"
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=label_key,
        )
    )

    unfused = copy.deepcopy(base)
    unfused["fused_regions"][1]["classification"] = "supported_but_unfused"
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=unfused,
        )
    )

    island = copy.deepcopy(base)
    island["fused_regions"][1]["candidate_kind"] = "unitary_island"
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=island,
        )
    )

    bad_reason = copy.deepcopy(base)
    bad_reason["fused_regions"][1]["reason"] = "eligible_channel_native_motif"
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=4,
            workload=manifest_cells[0]["workload"],
            realization=bad_reason,
        )
    )


def test_mf1a_strict_pure_unitary_partition_raises(monkeypatch: pytest.MonkeyPatch):
    from squander.partitioning.noisy_planner import (
        build_canonical_planner_surface_from_operation_specs,
        build_partition_descriptor_set,
    )
    from squander.partitioning.noisy_runtime import execute_partitioned_density_channel_native
    from squander.partitioning.noisy_validation_errors import NoisyRuntimeValidationError

    strict = _mf1a_strict_module()
    noisy_pair = strict.mf1a_strict_spectator_operation_specs(2)
    pure_pair = [
        {"kind": "gate", "name": "U3", "target_qbit": 2, "param_count": 3},
        {"kind": "gate", "name": "U3", "target_qbit": 3, "param_count": 3},
        {
            "kind": "gate",
            "name": "CNOT",
            "target_qbit": 3,
            "control_qbit": 2,
            "param_count": 0,
        },
        {"kind": "gate", "name": "U3", "target_qbit": 2, "param_count": 3},
    ]
    surface = build_canonical_planner_surface_from_operation_specs(
        qbit_num=4,
        source_type="structured_family_builder",
        workload_id="mf1a_strict_ineligible_pure_pair",
        operation_specs=noisy_pair + pure_pair,
    )
    descriptor_set = build_partition_descriptor_set(surface, max_partition_qubits=2)
    parameters = strict.build_initial_parameters(descriptor_set.parameter_count)
    with pytest.raises(NoisyRuntimeValidationError) as excinfo:
        execute_partitioned_density_channel_native(descriptor_set, parameters)
    error = excinfo.value
    assert error.category == "unsupported_runtime_operation"
    assert error.first_unsupported_condition == "channel_native_noise_presence"
    assert error.failure_stage == "runtime_preflight"

    clean = _clean_mf1a_strict_provenance()

    def _return_ineligible(_cell):
        return descriptor_set

    monkeypatch.setattr(strict, "_build_cell_descriptor", _return_ineligible)
    with pytest.raises(NoisyRuntimeValidationError):
        strict.build_cases(provenance=clean)


def test_mf1a_strict_case_consistency_guards_fail():
    strict = _mf1a_strict_module()
    manifest_cells = strict.build_manifest()["cells"]
    base = _mf1a_strict_frozen_realization(partition_count=2)
    workload = manifest_cells[0]["workload"]

    f1 = _mf1a_strict_synthetic_case(
        strict,
        anchor_qbits=4,
        workload=workload,
        realization=copy.deepcopy(base),
        route="phase31_channel_native_hybrid",
    )
    assert not strict._route_realization_pass(f1)

    f2 = _mf1a_strict_synthetic_case(
        strict,
        anchor_qbits=4,
        workload=workload,
        realization=copy.deepcopy(base),
        max_partition_qubits=3,
    )
    assert not strict._route_realization_pass(f2)

    f3 = copy.deepcopy(base)
    f3["requested_path"] = "phase31_channel_native_hybrid"
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict, anchor_qbits=4, workload=workload, realization=f3
        )
    )

    f4 = copy.deepcopy(base)
    f4["realized_path"] = "partitioned_density_descriptor_baseline"
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict, anchor_qbits=4, workload=workload, realization=f4
        )
    )

    f5 = copy.deepcopy(base)
    f5["exact_output_present"] = False
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict, anchor_qbits=4, workload=workload, realization=f5
        )
    )

    f6 = {
        "requested_path": "phase31_channel_native",
        "realized_path": "phase31_channel_native",
        "partition_count": 0,
        "exact_output_present": True,
        "channel_native_partition_count": 0,
        "partitions": [],
        "fused_regions": [],
    }
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict, anchor_qbits=4, workload=workload, realization=f6
        )
    )

    f11 = copy.deepcopy(base)
    f11["channel_native_partition_count"] = 3
    assert not strict._route_realization_pass(
        _mf1a_strict_synthetic_case(
            strict, anchor_qbits=4, workload=workload, realization=f11
        )
    )


def test_mf1a_strict_qa001_tolerances_match_q4():
    strict = _mf1a_strict_module()
    assert strict.evaluate_mf1a_qa001 is mf1a.evaluate_mf1a_qa001
    assert strict.MF1A_QA001_MATRIX_TOL is mf1a.MF1A_QA001_MATRIX_TOL
    assert strict.MF1A_QA001_LAMBDA_MIN_FLOOR is mf1a.MF1A_QA001_LAMBDA_MIN_FLOOR
    assert strict._QA001_REGENERATION_TOLERANCES is mf1a._QA001_REGENERATION_TOLERANCES
    assert strict._QA001_VALUE_KEYS is mf1a._QA001_VALUE_KEYS


def test_mf1a_strict_allowlist_is_length_four_and_siblings_stay():
    strict = _mf1a_strict_module()
    fused = _mf1a_fused_module()
    hybrid = _mf1a_hybrid_module()
    assert strict.STRICT_REGENERATION_ALLOWLIST == (
        "cases[0].provenance.implementation_revision",
        "cases[1].provenance.implementation_revision",
        "cases[2].provenance.implementation_revision",
        "cases[3].provenance.implementation_revision",
    )
    assert len(strict.STRICT_REGENERATION_ALLOWLIST) == 4
    assert len(mf1a.Q4_REGENERATION_ALLOWLIST) == 1
    assert len(fused.FUSED_REGENERATION_ALLOWLIST) == 4
    assert len(hybrid.HYBRID_REGENERATION_ALLOWLIST) == 4
    assert strict._allowlisted_revision_difference is not mf1a._allowlisted_revision_difference
    assert strict._allowlisted_revision_difference is not fused._allowlisted_revision_difference
    assert strict._allowlisted_revision_difference is not hybrid._allowlisted_revision_difference


def test_mf1a_strict_rejects_case_count():
    strict = _mf1a_strict_module()
    manifest_cells = strict.build_manifest()["cells"]
    realization = _mf1a_strict_frozen_realization(partition_count=2)
    prior_cases = [
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            realization=copy.deepcopy(realization),
        )
        for cell in manifest_cells
    ]
    prior = strict.build_artifact_bundle(prior_cases, prior_bundle=None)
    too_many = copy.deepcopy(prior_cases)
    too_many.append(copy.deepcopy(too_many[0]))
    _assert_strict_manifest_exact_set(
        strict.build_artifact_bundle(too_many, prior_bundle=prior)
    )
    too_few = copy.deepcopy(prior_cases[:3])
    _assert_strict_manifest_exact_set(
        strict.build_artifact_bundle(too_few, prior_bundle=prior)
    )


def test_mf1a_strict_rejects_substituted_or_reordered_id():
    strict = _mf1a_strict_module()
    manifest_cells = strict.build_manifest()["cells"]
    realization = _mf1a_strict_frozen_realization(partition_count=2)
    prior_cases = [
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            realization=copy.deepcopy(realization),
        )
        for cell in manifest_cells
    ]
    prior = strict.build_artifact_bundle(prior_cases, prior_bundle=None)
    substituted = copy.deepcopy(prior_cases)
    substituted[2]["workload"] = "mf1a_strict_spectator_embed_q8_wrong"
    _assert_strict_manifest_exact_set(
        strict.build_artifact_bundle(substituted, prior_bundle=prior)
    )
    reordered = copy.deepcopy(prior_cases)
    reordered[2], reordered[3] = reordered[3], reordered[2]
    _assert_strict_manifest_exact_set(
        strict.build_artifact_bundle(reordered, prior_bundle=prior)
    )


def _mf1a_strict_uniform_prior_cases(strict):
    manifest_cells = strict.build_manifest()["cells"]
    realization = _mf1a_strict_frozen_realization(partition_count=2)
    prior_cases = [
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            realization=copy.deepcopy(realization),
        )
        for cell in manifest_cells
    ]
    for case in prior_cases:
        case["provenance"]["implementation_revision"] = "a" * 40
    return prior_cases


def test_mf1a_strict_one_current_revision_diverges():
    strict = _mf1a_strict_module()
    prior_cases = _mf1a_strict_uniform_prior_cases(strict)
    prior = strict.build_artifact_bundle(prior_cases, prior_bundle=None)

    rc5_current = copy.deepcopy(prior_cases)
    for case in rc5_current:
        case["provenance"]["implementation_revision"] = "a" * 40
    rc5_current[2]["provenance"]["implementation_revision"] = "d" * 40
    _assert_strict_regeneration_negative(
        strict.build_artifact_bundle(rc5_current, prior_bundle=prior),
        "cases[2].provenance.implementation_revision",
    )

    all_moved_current = copy.deepcopy(prior_cases)
    revisions = ["c" * 40, "d" * 40, "c" * 40, "c" * 40]
    for case, revision in zip(all_moved_current, revisions, strict=True):
        case["provenance"]["implementation_revision"] = revision
    _assert_strict_regeneration_negative(
        strict.build_artifact_bundle(all_moved_current, prior_bundle=prior),
        "cases[0].provenance.implementation_revision",
    )


def test_mf1a_strict_one_prior_revision_diverges():
    strict = _mf1a_strict_module()
    prior_cases = _mf1a_strict_uniform_prior_cases(strict)
    for case in prior_cases:
        case["provenance"]["implementation_revision"] = "a" * 40
    prior = strict.build_artifact_bundle(prior_cases, prior_bundle=None)

    rc6_current = copy.deepcopy(prior_cases)
    prior_mut = copy.deepcopy(prior)
    prior_mut["cases"][1]["provenance"]["implementation_revision"] = "d" * 40
    _assert_strict_regeneration_negative(
        strict.build_artifact_bundle(rc6_current, prior_bundle=prior_mut),
        "cases[1].provenance.implementation_revision",
    )

    all_moved_prior = copy.deepcopy(prior)
    for case in rc6_current:
        case["provenance"]["implementation_revision"] = "c" * 40
    prior_mut = copy.deepcopy(prior)
    for index, revision in enumerate(["a" * 40, "a" * 40, "b" * 40, "a" * 40]):
        prior_mut["cases"][index]["provenance"]["implementation_revision"] = revision
    _assert_strict_regeneration_negative(
        strict.build_artifact_bundle(rc6_current, prior_bundle=prior_mut),
        "cases[0].provenance.implementation_revision",
    )


def test_mf1a_strict_one_case_non_allowlisted_path():
    strict = _mf1a_strict_module()
    prior_cases = _mf1a_strict_uniform_prior_cases(strict)
    prior = strict.build_artifact_bundle(prior_cases, prior_bundle=None)
    current = copy.deepcopy(prior_cases)
    for case in current:
        case["provenance"]["implementation_revision"] = "c" * 40
    current[1]["seed_policy"] = "structured_family_seed_20260318"
    _assert_strict_regeneration_negative(
        strict.build_artifact_bundle(current, prior_bundle=prior),
        "cases[1].seed_policy",
    )


def test_mf1a_strict_swapped_two_cases_revisions_only():
    strict = _mf1a_strict_module()
    prior_cases = _mf1a_strict_uniform_prior_cases(strict)
    prior = strict.build_artifact_bundle(prior_cases, prior_bundle=None)
    current = copy.deepcopy(prior_cases)
    for case in current:
        case["provenance"]["implementation_revision"] = "c" * 40
    prior_before = copy.deepcopy(prior)
    for index, revision in enumerate(["a" * 40, "b" * 40, "a" * 40, "a" * 40]):
        prior_before["cases"][index]["provenance"]["implementation_revision"] = revision
    _assert_strict_regeneration_negative(
        strict.build_artifact_bundle(current, prior_bundle=prior_before),
        "cases[0].provenance.implementation_revision",
    )
    prior_after = copy.deepcopy(prior_before)
    prior_after["cases"][0]["provenance"]["implementation_revision"] = "b" * 40
    prior_after["cases"][1]["provenance"]["implementation_revision"] = "a" * 40
    _assert_strict_regeneration_negative(
        strict.build_artifact_bundle(current, prior_bundle=prior_after),
        "cases[0].provenance.implementation_revision",
    )


def test_mf1a_strict_empty_allowlist_fails(monkeypatch: pytest.MonkeyPatch):
    strict = _mf1a_strict_module()
    monkeypatch.setattr(strict, "STRICT_REGENERATION_ALLOWLIST", ())
    prior_cases = _mf1a_strict_uniform_prior_cases(strict)
    prior = strict.build_artifact_bundle(prior_cases, prior_bundle=None)
    current = copy.deepcopy(prior_cases)
    for case in current:
        case["provenance"]["implementation_revision"] = "c" * 40
    _assert_strict_regeneration_negative(
        strict.build_artifact_bundle(current, prior_bundle=prior),
        "cases[0].provenance.implementation_revision",
    )


def test_mf1a_strict_length_three_allowlist_fails(monkeypatch: pytest.MonkeyPatch):
    strict = _mf1a_strict_module()
    monkeypatch.setattr(
        strict,
        "STRICT_REGENERATION_ALLOWLIST",
        strict.STRICT_REGENERATION_ALLOWLIST[:3],
    )
    prior_cases = _mf1a_strict_uniform_prior_cases(strict)
    prior = strict.build_artifact_bundle(prior_cases, prior_bundle=None)
    current = copy.deepcopy(prior_cases)
    for case in current:
        case["provenance"]["implementation_revision"] = "c" * 40
    _assert_strict_regeneration_negative(
        strict.build_artifact_bundle(current, prior_bundle=prior),
        "cases[3].provenance.implementation_revision",
    )


def test_mf1a_strict_revision_only_passes():
    strict = _mf1a_strict_module()
    prior_cases = _mf1a_strict_uniform_prior_cases(strict)
    prior = strict.build_artifact_bundle(prior_cases, prior_bundle=None)
    current = copy.deepcopy(prior_cases)
    for case in current:
        case["provenance"]["implementation_revision"] = "c" * 40
    bundle = strict.build_artifact_bundle(current, prior_bundle=prior)
    assert bundle["status"] == "pass"
    assert bundle["summary"]["first_failure"] is None
    assert bundle["regeneration"] == {
        "prior_present": True,
        "pass": True,
        "first_mismatch": None,
    }
    for index in (0, 1, 2, 3):
        failing = copy.deepcopy(current)
        failing[index]["provenance"]["provenance_pass"] = False
        failing[index]["provenance"]["implementation_revision"] = "a" * 40
        bundle = strict.build_artifact_bundle(failing, prior_bundle=None)
        assert bundle["status"] == "fail"
        assert bundle["summary"]["first_failure"] == "provenance"


def test_mf1a_strict_rejects_non_revision_value():
    strict = _mf1a_strict_module()
    prior_cases = _mf1a_strict_uniform_prior_cases(strict)
    prior = strict.build_artifact_bundle(prior_cases, prior_bundle=None)
    invalid_values = ("g" * 40, "c" * 39, "", "C" * 40)
    for value in invalid_values:
        current_cases = copy.deepcopy(prior_cases)
        current_cases[0]["provenance"]["implementation_revision"] = value
        bundle = strict.build_artifact_bundle(current_cases, prior_bundle=prior)
        _assert_strict_regeneration_negative(
            bundle, "cases[0].provenance.implementation_revision"
        )
        prior_mut = copy.deepcopy(prior)
        prior_mut["cases"][1]["provenance"]["implementation_revision"] = value
        current_cases = copy.deepcopy(prior_cases)
        bundle = strict.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
        _assert_strict_regeneration_negative(
            bundle, "cases[1].provenance.implementation_revision"
        )
    current_cases = copy.deepcopy(prior_cases)
    del current_cases[2]["provenance"]["implementation_revision"]
    bundle = strict.build_artifact_bundle(current_cases, prior_bundle=prior)
    _assert_strict_regeneration_negative(
        bundle, "cases[2].provenance.implementation_revision"
    )
    prior_mut = copy.deepcopy(prior)
    del prior_mut["cases"][2]["provenance"]["implementation_revision"]
    current_cases = copy.deepcopy(prior_cases)
    bundle = strict.build_artifact_bundle(current_cases, prior_bundle=prior_mut)
    _assert_strict_regeneration_negative(
        bundle, "cases[2].provenance.implementation_revision"
    )
    for value in invalid_values:
        current_uniform = copy.deepcopy(prior_cases)
        for case in current_uniform:
            case["provenance"]["implementation_revision"] = value
        bundle = strict.build_artifact_bundle(current_uniform, prior_bundle=prior)
        _assert_strict_regeneration_negative(
            bundle, "cases[0].provenance.implementation_revision"
        )
        prior_mut = copy.deepcopy(prior)
        for case in prior_mut["cases"]:
            case["provenance"]["implementation_revision"] = value
        current_uniform = copy.deepcopy(prior_cases)
        bundle = strict.build_artifact_bundle(current_uniform, prior_bundle=prior_mut)
        _assert_strict_regeneration_negative(
            bundle, "cases[0].provenance.implementation_revision"
        )
    current_uniform = copy.deepcopy(prior_cases)
    for case in current_uniform:
        del case["provenance"]["implementation_revision"]
    bundle = strict.build_artifact_bundle(current_uniform, prior_bundle=prior)
    _assert_strict_regeneration_negative(
        bundle, "cases[0].provenance.implementation_revision"
    )
    prior_mut = copy.deepcopy(prior)
    for case in prior_mut["cases"]:
        del case["provenance"]["implementation_revision"]
    current_uniform = copy.deepcopy(prior_cases)
    bundle = strict.build_artifact_bundle(current_uniform, prior_bundle=prior_mut)
    _assert_strict_regeneration_negative(
        bundle, "cases[0].provenance.implementation_revision"
    )


def test_mf1a_strict_finding_bands_follow_layer1():
    import json

    strict = _mf1a_strict_module()
    manifest_cells = strict.build_manifest()["cells"]
    realization = _mf1a_strict_frozen_realization(partition_count=2)
    cases = [
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            realization=copy.deepcopy(realization),
        )
        for cell in manifest_cells
    ]
    for measure, value, expected in (
        ("frobenius_norm_diff", 5e-14, "expected_range"),
        ("frobenius_norm_diff", 5e-12, "outside_expected"),
        ("frobenius_norm_diff", 5e-11, "finding"),
        ("frobenius_norm_diff", 2e-10, "qa001_fail"),
        ("lambda_min", -5e-14, "no_finding"),
        ("lambda_min", -5e-13, "finding"),
        ("lambda_min", -2e-12, "qa001_fail"),
        ("lambda_min", 3e-6, "no_finding"),
    ):
        assert strict.classify_near_threshold_measure(measure, value) == expected

    cases[0]["qa001"]["lambda_min"] = -5e-13
    bundle = strict.build_artifact_bundle(cases, prior_bundle=None)
    assert bundle["status"] == "pass"
    assert bundle["summary"]["first_failure"] is None
    assert bundle["summary"]["findings"] == [
        {
            "route": "phase31_channel_native",
            "anchor_qbits": 4,
            "workload": "phase31_local_support_q4_spectator_embedding_smoke",
            "measure": "lambda_min",
            "value": -5e-13,
            "cause_hypothesis": "lambda_min below -1e-13 Layer 1 finding band",
        }
    ]
    assert "oracle_lambda_min" not in json.dumps(bundle)
    cases_positive = [
        _mf1a_strict_synthetic_case(
            strict,
            anchor_qbits=cell["anchor_qbits"],
            workload=cell["workload"],
            realization=copy.deepcopy(realization),
        )
        for cell in manifest_cells
    ]
    cases_positive[1]["qa001"]["lambda_min"] = 3e-6
    bundle = strict.build_artifact_bundle(cases_positive, prior_bundle=None)
    assert bundle["summary"]["findings"] == []
    assert bundle["summary"]["outside_expected_markers"] == []


def test_mf1a_strict_second_field_fails():
    strict = _mf1a_strict_module()
    prior_cases = _mf1a_strict_uniform_prior_cases(strict)
    prior = strict.build_artifact_bundle(prior_cases, prior_bundle=None)
    current = copy.deepcopy(prior_cases)
    for case in current:
        case["provenance"]["implementation_revision"] = "c" * 40
    current[1]["workload"] = "mf1a_strict_spectator_embed_q6_wrong"
    bundle = strict.build_artifact_bundle(current, prior_bundle=prior)
    assert bundle["status"] == "fail"
    assert bundle["regeneration"]["pass"] is False
    assert bundle["regeneration"]["first_mismatch"] == "cases[1].workload"
    assert bundle["summary"]["first_failure"] == "manifest_exact_set"


def test_mf1a_strict_pipeline_builds_every_sibling_before_writing_any(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    strict = _mf1a_strict_module()
    fused = _mf1a_fused_module()
    hybrid = _mf1a_hybrid_module()
    from benchmarks.density_matrix.correctness_evidence import (
        mf1a_q4_baseline_validation as q4_mod,
    )

    live = validation_pipeline._CASE_SLICE_REGISTRY[0:4]
    assert len(live) == 4
    assert live[0].module is q4_mod and live[0].mf1a_sibling is True
    assert live[1].module is fused and live[1].mf1a_sibling is True
    assert live[2].module is hybrid and live[2].mf1a_sibling is True
    assert live[3].module is strict and live[3].mf1a_sibling is True

    events: list[tuple[str, str]] = []
    fake_root = tmp_path / "mf1a_fake_siblings"

    def make_fake_sibling(name: str) -> SimpleNamespace:
        module = SimpleNamespace(
            SUITE_NAME=f"correctness_evidence_mf1a_fake_{name}",
            ARTIFACT_FILENAME=f"fake_{name}_bundle.json",
            DEFAULT_OUTPUT_DIR=fake_root / name,
        )

        def build_cases() -> list:
            events.append(("build_cases", name))
            return []

        def build_artifact_bundle(_cases: list | None = None) -> dict:
            events.append(("build_artifact_bundle", name))
            return {"status": "pass", "cases": []}

        module.build_cases = build_cases
        module.build_artifact_bundle = build_artifact_bundle
        return module

    patched_registry = (
        validation_pipeline._CaseSuiteEntry(
            make_fake_sibling("a"), "build_cases", "build_artifact_bundle", mf1a_sibling=True
        ),
        validation_pipeline._CaseSuiteEntry(
            make_fake_sibling("b"), "build_cases", "build_artifact_bundle", mf1a_sibling=True
        ),
        validation_pipeline._CaseSuiteEntry(
            make_fake_sibling("c"), "build_cases", "build_artifact_bundle", mf1a_sibling=True
        ),
        validation_pipeline._CaseSuiteEntry(
            make_fake_sibling("d"), "build_cases", "build_artifact_bundle", mf1a_sibling=True
        ),
    )
    monkeypatch.setattr(validation_pipeline, "_CASE_SLICE_REGISTRY", patched_registry)
    monkeypatch.setattr(validation_pipeline, "_NULLARY_BUNDLE_REGISTRY", ())
    original_write = validation_pipeline._write_slice_bundle

    def log_write(module, bundle: dict) -> Path:
        events.append(("write", module.SUITE_NAME))
        return original_write(module, bundle)

    monkeypatch.setattr(validation_pipeline, "_write_slice_bundle", log_write)
    validation_pipeline.run_pipeline()
    first_write_index = next(
        index for index, event in enumerate(events) if event[0] == "write"
    )
    build_indices = [
        index for index, event in enumerate(events) if event[0].startswith("build")
    ]
    assert build_indices
    assert all(build_index < first_write_index for build_index in build_indices)



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
    fused_sibling_path = fake_root / "mf1a" / "fused" / _mf1a_fused_module().ARTIFACT_FILENAME
    assert fused_sibling_path.is_file()
    hybrid_sibling_path = (
        fake_root / "mf1a" / "hybrid" / _mf1a_hybrid_module().ARTIFACT_FILENAME
    )
    assert hybrid_sibling_path.is_file()
    strict_sibling_path = (
        fake_root / "mf1a" / "strict" / _mf1a_strict_module().ARTIFACT_FILENAME
    )
    assert strict_sibling_path.is_file()


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

    assert sibling_dirs == {"mf1a/q4_baseline", "mf1a/fused", "mf1a/hybrid", "mf1a/strict"}
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
