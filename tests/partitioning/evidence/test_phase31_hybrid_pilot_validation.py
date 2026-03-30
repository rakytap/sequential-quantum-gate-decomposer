"""Regression tests for the Phase 3.1 hybrid performance pilot (P31-S09-E01)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.case_selection import (
    BENCHMARK_SLICE_PHASE31_HYBRID_PILOT,
    PHASE31_HYBRID_PILOT_WORKLOAD_ID,
    build_phase31_hybrid_pilot_case_context,
    build_phase31_performance_inventory_cases,
    iter_phase31_performance_cases,
)
from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
    PERFORMANCE_EVIDENCE_REPETITIONS,
)
from benchmarks.density_matrix.performance_evidence.phase31_hybrid_pilot_validation import (
    build_phase31_hybrid_pilot_bundle,
    build_phase31_hybrid_pilot_cases,
)
from benchmarks.density_matrix.performance_evidence.records import (
    _PHASE31_HYBRID_DECISION_CLASSES,
    _PHASE31_HYBRID_DIAGNOSIS_TAGS,
    build_phase31_hybrid_pilot_record,
)

_FROZEN_DECISION_CLASSES = frozenset(_PHASE31_HYBRID_DECISION_CLASSES)
_FROZEN_DIAGNOSIS_TAGS = frozenset(_PHASE31_HYBRID_DIAGNOSIS_TAGS)


def test_phase31_hybrid_pilot_workload_id_frozen():
    ctx = build_phase31_hybrid_pilot_case_context()
    assert ctx.descriptor_set.workload_id == PHASE31_HYBRID_PILOT_WORKLOAD_ID
    assert ctx.metadata["benchmark_slice"] == BENCHMARK_SLICE_PHASE31_HYBRID_PILOT


def test_phase31_hybrid_pilot_record_baseline_trio_and_routes():
    record = build_phase31_hybrid_pilot_record(build_phase31_hybrid_pilot_case_context())
    assert record["workload_id"] == PHASE31_HYBRID_PILOT_WORKLOAD_ID
    assert record["timing_mode"] == "median_3"
    n = PERFORMANCE_EVIDENCE_REPETITIONS
    assert len(record["sequential_runtime_ms_samples"]) == n
    assert len(record["phase3_fused_runtime_ms_samples"]) == n
    assert len(record["phase31_hybrid_runtime_ms_samples"]) == n
    assert len(record["sequential_peak_rss_kb_samples"]) == n
    assert len(record["phase3_fused_peak_rss_kb_samples"]) == n
    assert len(record["phase31_hybrid_peak_rss_kb_samples"]) == n
    for key in (
        "sequential_median_runtime_ms",
        "phase3_fused_median_runtime_ms",
        "phase31_hybrid_median_runtime_ms",
    ):
        assert record[key] > 0.0
    for key in (
        "sequential_median_peak_rss_kb",
        "phase3_fused_median_peak_rss_kb",
        "phase31_hybrid_median_peak_rss_kb",
    ):
        assert record[key] >= 0
    for key in (
        "channel_native_partition_count",
        "phase3_routed_partition_count",
        "channel_native_member_count",
        "phase3_routed_member_count",
    ):
        assert key in record
        assert record[key] >= 0
    assert record["channel_native_partition_count"] > 0
    assert record["channel_native_member_count"] > 0
    assert record["decision_class"] in _FROZEN_DECISION_CLASSES
    assert record["diagnosis_tag"] in _FROZEN_DIAGNOSIS_TAGS
    assert isinstance(record["hybrid_partition_route_records"], list)
    assert len(record["hybrid_partition_route_records"]) == record[
        "phase31_hybrid_partition_count"
    ]
    assert record["phase3_fused_internal_reference_pass"] is True
    assert record["phase31_hybrid_internal_reference_pass"] is True


def test_phase31_hybrid_pilot_decision_consistent_with_tags():
    record = build_phase31_hybrid_pilot_record(build_phase31_hybrid_pilot_case_context())
    cn = record["channel_native_partition_count"]
    fused_ms = record["phase3_fused_median_runtime_ms"]
    hybrid_ms = record["phase31_hybrid_median_runtime_ms"]
    speedup = fused_ms / hybrid_ms if hybrid_ms > 0.0 else 0.0
    positive = speedup >= 1.2
    tag = record["diagnosis_tag"]
    decision = record["decision_class"]
    if positive:
        assert tag == "phase31_positive_gain"
        assert decision == "phase31_justified"
    elif cn == 0:
        assert tag == "limited_channel_native_coverage"
        assert decision == "phase3_sufficient"
    else:
        assert tag == "hybrid_overhead_dominant"
        assert decision == "phase31_not_justified_yet"


def test_phase31_hybrid_pilot_bundle_schema():
    cases = build_phase31_hybrid_pilot_cases()
    bundle = build_phase31_hybrid_pilot_bundle(cases)
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION
    s = bundle["summary"]
    assert s["total_cases"] == 1
    assert s["pilot_workload_id"] == PHASE31_HYBRID_PILOT_WORKLOAD_ID
    assert s["pilot_case_name"] == cases[0]["case_name"]
    assert s["timing_mode"] == "median_3"
    assert s["decision_class"] == cases[0]["decision_class"]
    assert s["diagnosis_tag"] == cases[0]["diagnosis_tag"]
    assert s["channel_native_partition_count"] == cases[0]["channel_native_partition_count"]
    assert s["phase3_routed_partition_count"] == cases[0]["phase3_routed_partition_count"]


def test_phase31_performance_helpers_inventory_and_iter_regression():
    """Guards Phase 3.1 planning helpers (control branch uses DEFAULT_STRUCTURED_SEED)."""
    inventory = build_phase31_performance_inventory_cases()
    assert len(inventory) == 26
    assert any(
        c.get("benchmark_slice") == "phase31_control_performance" for c in inventory
    )
    contexts = list(iter_phase31_performance_cases())
    assert len(contexts) == 26
    assert any(
        ctx.metadata.get("benchmark_slice") == "phase31_control_performance"
        for ctx in contexts
    )
