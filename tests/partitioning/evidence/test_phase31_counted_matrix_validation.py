"""Regression tests for the Phase 3.1 counted performance matrix (P31-S11-E01)."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.case_selection import (
    build_phase31_counted_performance_case_contexts,
    build_phase31_counted_performance_inventory_cases,
)
from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION,
)
from benchmarks.density_matrix.performance_evidence.phase31_counted_matrix_validation import (
    build_artifact_bundle,
    build_cases,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_phase31_counted_performance_record,
)


def test_phase31_counted_matrix_inventory_frozen():
    inventory = build_phase31_counted_performance_inventory_cases()
    assert len(inventory) == 26
    assert sum(case["benchmark_slice"] == "phase31_structured_performance" for case in inventory) == 24
    assert sum(case["benchmark_slice"] == "phase31_control_performance" for case in inventory) == 2
    assert all(case["counted_phase31_case"] is True for case in inventory)
    assert all(case["claim_surface_id"] == "phase31_bounded_mixed_motif_v1" for case in inventory)


def test_phase31_counted_matrix_contexts_frozen():
    contexts = build_phase31_counted_performance_case_contexts()
    assert len(contexts) == 26
    assert sum(ctx.metadata["benchmark_slice"] == "phase31_structured_performance" for ctx in contexts) == 24
    assert sum(ctx.metadata["benchmark_slice"] == "phase31_control_performance" for ctx in contexts) == 2


def test_phase31_counted_matrix_record_fields():
    record = build_phase31_counted_performance_record(
        build_phase31_counted_performance_case_contexts()[0]
    )
    assert record["record_schema_version"] == PERFORMANCE_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION
    assert record["artifact_kind"] == "phase31_counted_performance_matrix_row"
    assert record["counted_phase31_case"] is True
    assert record["claim_surface_id"] == "phase31_bounded_mixed_motif_v1"
    assert record["representation_primary"] == "kraus_bundle"
    assert record["contains_noise"] is True
    assert record["runtime_class"] == "phase31_channel_native_hybrid"
    assert record["build_policy_id"] == "phase31_scalar_only_v1"
    assert record["build_flavor"] == "scalar"
    assert record["simd_enabled"] is False
    assert record["tbb_enabled"] is False
    assert record["thread_count"] == 1
    assert record["counted_claim_build"] is True
    assert record["timing_mode"] == "median_3"
    assert record["decision_class"] in {
        "phase3_sufficient",
        "phase31_justified",
        "phase31_not_justified_yet",
    }
    assert isinstance(record["hybrid_partition_route_records"], list)
    assert len(record["hybrid_partition_route_records"]) == (
        record["channel_native_partition_count"] + record["phase3_routed_partition_count"]
    )
    for key in (
        "channel_native_partition_count",
        "phase3_routed_partition_count",
        "channel_native_member_count",
        "phase3_routed_member_count",
    ):
        assert key in record
        assert record[key] >= 0
    for key in (
        "sequential_runtime_ms_samples",
        "phase3_fused_runtime_ms_samples",
        "phase31_hybrid_runtime_ms_samples",
    ):
        assert len(record[key]) == 3


def test_phase31_counted_matrix_bundle_summary():
    cases = build_cases()
    bundle = build_artifact_bundle(cases)
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == PERFORMANCE_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION
    summary = bundle["summary"]
    assert summary["total_cases"] == 26
    assert summary["primary_rows"] == 24
    assert summary["control_rows"] == 2
    assert summary["baseline_trio_present"] is True
    assert summary["route_fields_present"] is True
    assert summary["build_metadata_present"] is True
    assert summary["inventory_match"] is True
