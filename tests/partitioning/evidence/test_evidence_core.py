"""Regression tests for shared density-matrix evidence core (Story 1)."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.evidence_core import (
    RUNTIME_CORRECTNESS_BRIDGE_FIELD_NAMES,
    counted_supported_case,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    build_correctness_evidence_positive_records,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_performance_evidence_core_benchmark_records,
)

# Snapshot: sorted bridge keys must stay stable unless the evidence contract changes intentionally.
_EXPECTED_SORTED_BRIDGE_FIELD_NAMES: tuple[str, ...] = (
    "actual_fused_execution",
    "continuity_energy_error",
    "continuity_energy_pass",
    "continuity_energy_required",
    "deferred_region_count",
    "descriptor_member_count",
    "exact_output_present",
    "external_frobenius_norm_diff",
    "external_max_abs_diff",
    "external_reference_pass",
    "fallback_used",
    "frobenius_norm_diff",
    "fused_gate_count",
    "fused_region_count",
    "gate_count",
    "internal_reference_pass",
    "max_abs_diff",
    "max_partition_span",
    "noise_count",
    "output_integrity_pass",
    "parameter_routing_segment_count",
    "partition_count",
    "partition_member_counts",
    "peak_rss_kb",
    "peak_rss_mb",
    "remapped_partition_count",
    "rho_is_valid",
    "rho_is_valid_tol",
    "runtime_ms",
    "runtime_path",
    "supported_runtime_case",
    "supported_unfused_gate_count",
    "supported_unfused_region_count",
    "trace_deviation",
)


def test_runtime_correctness_bridge_field_names_stable():
    assert tuple(sorted(RUNTIME_CORRECTNESS_BRIDGE_FIELD_NAMES)) == _EXPECTED_SORTED_BRIDGE_FIELD_NAMES


def test_counted_supported_matches_between_pipelines_when_reference_available():
    correctness_by_workload = {
        r["workload_id"]: r for r in build_correctness_evidence_positive_records()
    }
    for perf in build_performance_evidence_core_benchmark_records():
        if not perf["correctness_evidence_reference_available"]:
            continue
        ce = correctness_by_workload[perf["workload_id"]]
        ce_counted = counted_supported_case(ce)
        perf_counted = counted_supported_case(perf)
        assert ce_counted == perf_counted
        assert perf_counted == perf["counted_supported_benchmark_case"]


def test_bridge_fields_match_correctness_when_reference_available():
    correctness_by_workload = {
        r["workload_id"]: r for r in build_correctness_evidence_positive_records()
    }
    for perf in build_performance_evidence_core_benchmark_records():
        if not perf["correctness_evidence_reference_available"]:
            continue
        ce = correctness_by_workload[perf["workload_id"]]
        for field in RUNTIME_CORRECTNESS_BRIDGE_FIELD_NAMES:
            assert perf[field] == ce[field], field


def test_counted_supported_case_synthetic_examples():
    base = {
        "supported_runtime_case": True,
        "internal_reference_pass": True,
        "output_integrity_pass": True,
        "continuity_energy_required": False,
        "continuity_energy_pass": True,
        "external_reference_required": False,
        "external_reference_pass": True,
    }
    assert counted_supported_case(base) is True

    assert counted_supported_case({**base, "supported_runtime_case": False}) is False
    assert counted_supported_case({**base, "continuity_energy_required": True, "continuity_energy_pass": False}) is False
    assert counted_supported_case(
        {**base, "external_reference_required": True, "external_reference_pass": False}
    ) is False
