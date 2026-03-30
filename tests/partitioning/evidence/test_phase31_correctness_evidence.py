"""Regression tests for Stage-A bounded Phase 3.1 correctness-evidence sibling (P31-S10-E02)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.correctness_evidence.case_selection import (
    CORRECTNESS_EVIDENCE_CASE_KIND_CONTINUITY,
    CORRECTNESS_EVIDENCE_CASE_KIND_MICROCASE,
    build_phase31_correctness_evidence_case_contexts,
)
from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION,
    CORRECTNESS_EVIDENCE_PHASE31_PACKAGE_SCHEMA_VERSION,
    CORRECTNESS_EVIDENCE_PHASE31_SUMMARY_SCHEMA_VERSION,
)
from benchmarks.density_matrix.correctness_evidence.phase31_correctness_bundle_validation import (
    build_artifact_bundle as build_phase31_correctness_package_bundle,
)
from benchmarks.density_matrix.correctness_evidence.phase31_correctness_matrix_validation import (
    build_artifact_bundle as build_phase31_matrix_bundle,
    build_cases as build_phase31_matrix_cases,
)
from benchmarks.density_matrix.correctness_evidence.phase31_external_correctness_validation import (
    build_artifact_bundle as build_phase31_external_bundle,
    build_cases as build_phase31_external_cases,
)
from benchmarks.density_matrix.correctness_evidence.phase31_output_integrity_validation import (
    build_artifact_bundle as build_phase31_output_integrity_bundle,
    build_cases as build_phase31_output_integrity_cases,
)
from benchmarks.density_matrix.correctness_evidence.phase31_runtime_classification_validation import (
    build_artifact_bundle as build_phase31_runtime_classification_bundle,
    build_cases as build_phase31_runtime_classification_cases,
)
from benchmarks.density_matrix.correctness_evidence.phase31_sequential_correctness_validation import (
    build_artifact_bundle as build_phase31_sequential_bundle,
    build_cases as build_phase31_sequential_cases,
)
from benchmarks.density_matrix.correctness_evidence.phase31_summary_consistency_validation import (
    build_artifact_bundle as build_phase31_summary_consistency_bundle,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    build_phase31_correctness_evidence_positive_records,
    counted_supported_case,
)


def test_phase31_case_contexts_bounded_counts():
    ctxs = build_phase31_correctness_evidence_case_contexts()
    assert len(ctxs) == 6
    kinds = [c.metadata["case_kind"] for c in ctxs]
    assert kinds.count(CORRECTNESS_EVIDENCE_CASE_KIND_MICROCASE) == 4
    assert kinds.count(CORRECTNESS_EVIDENCE_CASE_KIND_CONTINUITY) == 2
    assert sum(c.metadata["external_reference_required"] for c in ctxs) == 5


def test_phase31_positive_records_fields_and_counted_supported():
    recs = build_phase31_correctness_evidence_positive_records()
    assert len(recs) == 6
    assert all(r["record_schema_version"] == CORRECTNESS_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION for r in recs)
    assert all(counted_supported_case(r) for r in recs)
    for r in recs:
        assert r["claim_surface_id"] == "phase31_bounded_mixed_motif_v1"
        assert r["representation_primary"] == "kraus_bundle"
        assert r["contains_noise"] is True
        assert r["counted_phase31_case"] is True
        assert "runtime_class" in r
        if r["case_kind"] == CORRECTNESS_EVIDENCE_CASE_KIND_MICROCASE:
            assert r["channel_invariants"] is not None
            assert r["channel_invariants"]["invariant_slice_pass"] is True
            assert r["partition_route_summary"] is None
            assert r["fused_block_support_qbits"] is not None
        else:
            assert r["channel_invariants"] is None
            assert r["partition_route_summary"] is not None
            assert r["partition_route_summary"]["partition_count"] >= 1
            assert r["fused_block_support_qbits"] is None


@pytest.mark.parametrize(
    "build_cases_fn,build_bundle_fn",
    (
        (build_phase31_matrix_cases, build_phase31_matrix_bundle),
        (build_phase31_sequential_cases, build_phase31_sequential_bundle),
        (build_phase31_external_cases, build_phase31_external_bundle),
        (build_phase31_output_integrity_cases, build_phase31_output_integrity_bundle),
        (build_phase31_runtime_classification_cases, build_phase31_runtime_classification_bundle),
    ),
    ids=(
        "phase31_matrix",
        "phase31_sequential",
        "phase31_external",
        "phase31_output_integrity",
        "phase31_runtime_classification",
    ),
)
def test_phase31_positive_slice_bundles_pass(build_cases_fn, build_bundle_fn):
    cases = build_cases_fn()
    bundle = build_bundle_fn(cases)
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == CORRECTNESS_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION


def test_phase31_correctness_matrix_summary():
    bundle = build_phase31_matrix_bundle(build_phase31_matrix_cases())
    s = bundle["summary"]
    assert s["total_cases"] == 6
    assert s["microcases"] == 4
    assert s["continuity_cases"] == 2
    assert s["structured_cases"] == 0
    assert s["external_slice_cases"] == 5


def test_phase31_external_slice_counts():
    bundle = build_phase31_external_bundle(build_phase31_external_cases())
    assert bundle["status"] == "pass"
    assert bundle["summary"]["total_cases"] == 5
    assert bundle["summary"]["microcases"] == 4
    assert bundle["summary"]["continuity_cases"] == 1


def test_phase31_correctness_package_and_summary():
    pkg = build_phase31_correctness_package_bundle()
    assert pkg["schema_version"] == CORRECTNESS_EVIDENCE_PHASE31_PACKAGE_SCHEMA_VERSION
    assert pkg["status"] == "pass"
    assert len(pkg["cases"]) == 6
    summ = build_phase31_summary_consistency_bundle()
    assert summ["status"] == "pass"
    assert summ["schema_version"] == CORRECTNESS_EVIDENCE_PHASE31_SUMMARY_SCHEMA_VERSION
    assert summ["summary"]["counted_supported_cases"] == 6
