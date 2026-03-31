from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import statistics
from typing import Any

from benchmarks.density_matrix.evidence_core import (
    RUNTIME_CORRECTNESS_BRIDGE_FIELD_NAMES,
    build_runtime_correctness_bridge_fields,
    counted_supported_case,
)
from benchmarks.density_matrix.correctness_evidence.common import build_selected_candidate
from benchmarks.density_matrix.partitioned_runtime.common import (
    build_density_comparison_metrics,
    execute_fused_with_reference,
)
from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
    PERFORMANCE_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION,
    PERFORMANCE_EVIDENCE_REFERENCE_BACKEND_EXTERNAL,
    PERFORMANCE_EVIDENCE_REFERENCE_BACKEND_INTERNAL,
    PERFORMANCE_EVIDENCE_REPETITIONS,
    PERFORMANCE_EVIDENCE_STATUS_COUNTED,
    PERFORMANCE_EVIDENCE_STATUS_DIAGNOSIS_ONLY,
    PERFORMANCE_EVIDENCE_STATUS_EXCLUDED,
    build_correctness_reference_index,
    build_phase31_counted_build_metadata,
    measure_sequential_density_reference,
)
from benchmarks.density_matrix.performance_evidence.case_selection import (
    build_performance_evidence_case_contexts,
    build_phase31_counted_performance_case_contexts,
)
from squander.partitioning.noisy_runtime import (
    execute_partitioned_density_channel_native_hybrid,
    execute_partitioned_density_fused,
)


def _base_record(case_context) -> dict:
    metadata = case_context.metadata
    descriptor_set = case_context.descriptor_set
    selected_candidate = build_selected_candidate()
    return {
        "record_schema_version": PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
        "candidate_schema_version": selected_candidate["candidate_schema_version"],
        "candidate_id": selected_candidate["candidate_id"],
        "planner_family": selected_candidate["planner_family"],
        "planner_variant": selected_candidate["planner_variant"],
        "planner_settings": dict(selected_candidate["planner_settings"]),
        "max_partition_qubits": selected_candidate["max_partition_qubits"],
        "planner_calibration_selected_candidate_id": selected_candidate["selected_candidate_id"],
        "planner_calibration_claim_selection_schema_version": selected_candidate[
            "claim_selection_schema_version"
        ],
        "planner_calibration_claim_selection_rule": selected_candidate["claim_selection_rule"],
        "case_name": metadata["case_name"],
        "case_kind": metadata["case_kind"],
        "benchmark_slice": metadata["benchmark_slice"],
        "representative_review_case": bool(metadata["representative_review_case"]),
        "review_group_id": metadata["review_group_id"],
        "requested_mode": descriptor_set.requested_mode,
        "source_type": descriptor_set.source_type,
        "workload_id": descriptor_set.workload_id,
        "qbit_num": descriptor_set.qbit_num,
        "parameter_count": descriptor_set.parameter_count,
        "family_name": metadata["family_name"],
        "noise_pattern": metadata["noise_pattern"],
        "seed": metadata["seed"],
        "topology": metadata["topology"],
        "planning_time_ms": metadata["planning_time_ms"],
        "external_reference_required": bool(metadata["external_reference_required"]),
        "reference_backend_internal": PERFORMANCE_EVIDENCE_REFERENCE_BACKEND_INTERNAL,
        "reference_backend_external": (
            PERFORMANCE_EVIDENCE_REFERENCE_BACKEND_EXTERNAL
            if metadata["external_reference_required"]
            else None
        ),
        "benchmark_matrix_pass": True,
    }


def _base_phase31_record(case_context) -> dict:
    record = _base_record(case_context)
    metadata = case_context.metadata
    record["record_schema_version"] = PERFORMANCE_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION
    record.update(
        {
            "claim_surface_id": metadata["claim_surface_id"],
            "representation_primary": metadata["representation_primary"],
            "contains_noise": bool(metadata["contains_noise"]),
            "counted_phase31_case": bool(metadata["counted_phase31_case"]),
            "fused_block_support_qbits": metadata.get("fused_block_support_qbits"),
            "runtime_class": "phase31_channel_native_hybrid",
            **build_phase31_counted_build_metadata(),
        }
    )
    return record


def _measure_review_timings(case_context) -> dict:
    descriptor_set = case_context.descriptor_set
    parameters = case_context.parameters

    sequential_runtime_ms_samples: list[float] = []
    fused_runtime_ms_samples: list[float] = []
    sequential_peak_rss_kb_samples: list[int] = []
    fused_peak_rss_kb_samples: list[int] = []

    for _ in range(PERFORMANCE_EVIDENCE_REPETITIONS):
        sequential_measurement = measure_sequential_density_reference(
            descriptor_set, parameters
        )
        fused_result = execute_partitioned_density_fused(descriptor_set, parameters)
        sequential_runtime_ms_samples.append(sequential_measurement.runtime_ms)
        fused_runtime_ms_samples.append(fused_result.runtime_ms)
        sequential_peak_rss_kb_samples.append(sequential_measurement.peak_rss_kb)
        fused_peak_rss_kb_samples.append(fused_result.peak_rss_kb)

    sequential_median_runtime_ms = float(statistics.median(sequential_runtime_ms_samples))
    fused_median_runtime_ms = float(statistics.median(fused_runtime_ms_samples))
    sequential_median_peak_rss_kb = int(statistics.median(sequential_peak_rss_kb_samples))
    fused_median_peak_rss_kb = int(statistics.median(fused_peak_rss_kb_samples))
    speedup = (
        sequential_median_runtime_ms / fused_median_runtime_ms
        if fused_median_runtime_ms > 0.0
        else 0.0
    )
    memory_reduction = (
        (sequential_median_peak_rss_kb - fused_median_peak_rss_kb)
        / sequential_median_peak_rss_kb
        if sequential_median_peak_rss_kb > 0
        else 0.0
    )

    return {
        "timing_mode": "median_3",
        "sequential_runtime_ms_samples": sequential_runtime_ms_samples,
        "fused_runtime_ms_samples": fused_runtime_ms_samples,
        "sequential_peak_rss_kb_samples": sequential_peak_rss_kb_samples,
        "fused_peak_rss_kb_samples": fused_peak_rss_kb_samples,
        "sequential_median_runtime_ms": sequential_median_runtime_ms,
        "fused_median_runtime_ms": fused_median_runtime_ms,
        "sequential_median_peak_rss_kb": sequential_median_peak_rss_kb,
        "fused_median_peak_rss_kb": fused_median_peak_rss_kb,
        "speedup": speedup,
        "memory_reduction": memory_reduction,
    }


def _diagnosis_reasons(record: dict) -> list[str]:
    reasons: list[str] = []
    if not record["actual_fused_execution"]:
        reasons.append("no_real_fused_coverage")
    if record["deferred_region_count"] > 0:
        reasons.append("limited_fused_coverage_due_to_noise_boundaries")
    if record["supported_unfused_region_count"] > 0:
        reasons.append("supported_islands_left_unfused")
    if record["representative_review_case"]:
        if record["fused_median_runtime_ms"] is not None and (
            record["fused_median_runtime_ms"] >= record["sequential_median_runtime_ms"]
        ):
            reasons.append("fused_runtime_slower_than_sequential_reference")
        if record["fused_median_peak_rss_kb"] is not None and (
            record["fused_median_peak_rss_kb"] >= record["sequential_median_peak_rss_kb"]
        ):
            reasons.append("no_peak_memory_reduction_against_sequential_reference")
    if not reasons:
        reasons.append("no_representative_threshold_gain_observed")
    return reasons


def _apply_correctness_evidence_reference_fields(
    record: dict, correctness_evidence_reference: dict
) -> None:
    for field in RUNTIME_CORRECTNESS_BRIDGE_FIELD_NAMES:
        record[field] = correctness_evidence_reference[field]


def performance_evidence_counted_supported_case(record: dict) -> bool:
    return counted_supported_case(record)


def build_performance_evidence_core_benchmark_record(case_context) -> dict:
    record = _base_record(case_context)
    correctness_evidence_reference = build_correctness_reference_index().get(
        record["workload_id"]
    )
    sequential_measurement = measure_sequential_density_reference(
        case_context.descriptor_set, case_context.parameters
    )
    if correctness_evidence_reference is not None:
        _apply_correctness_evidence_reference_fields(record, correctness_evidence_reference)
    else:
        fused_result, reference_density, density_metrics = execute_fused_with_reference(
            case_context.descriptor_set, case_context.parameters
        )
        record.update(
            build_runtime_correctness_bridge_fields(
                case_context,
                fused_result,
                reference_density,
                density_metrics,
                external_reference_required=record["external_reference_required"],
            )
        )

    record.update(
        {
            "sequential_runtime_ms_single": sequential_measurement.runtime_ms,
            "sequential_peak_rss_kb_single": sequential_measurement.peak_rss_kb,
            "sequential_trace_deviation": sequential_measurement.trace_deviation,
            "sequential_rho_is_valid": sequential_measurement.rho_is_valid,
            "correctness_evidence_reference_available": correctness_evidence_reference is not None,
            "correctness_evidence_counted_reference_available": (
                False
                if correctness_evidence_reference is None
                else counted_supported_case(correctness_evidence_reference)
            ),
        }
    )

    counted_supported = performance_evidence_counted_supported_case(record)
    record["counted_supported_benchmark_case"] = counted_supported

    benchmark_status = PERFORMANCE_EVIDENCE_STATUS_COUNTED
    if not counted_supported:
        benchmark_status = PERFORMANCE_EVIDENCE_STATUS_EXCLUDED

    record.update(
        {
            "timing_mode": "single_run",
            "sequential_runtime_ms_samples": None,
            "fused_runtime_ms_samples": None,
            "sequential_peak_rss_kb_samples": None,
            "fused_peak_rss_kb_samples": None,
            "sequential_median_runtime_ms": None,
            "fused_median_runtime_ms": None,
            "sequential_median_peak_rss_kb": None,
            "fused_median_peak_rss_kb": None,
            "speedup": None,
            "memory_reduction": None,
            "positive_threshold_pass": False,
            "diagnosis_reasons": [],
            "diagnosis_only_case": False,
            "benchmark_status": benchmark_status,
            "exclusion_reason": (
                None
                if benchmark_status != PERFORMANCE_EVIDENCE_STATUS_EXCLUDED
                else "performance_evidence_counted_supported_gate_failed"
            ),
        }
    )
    return record


@lru_cache(maxsize=1)
def _build_performance_evidence_core_benchmark_records_cached() -> tuple[dict, ...]:
    return tuple(
        build_performance_evidence_core_benchmark_record(case_context)
        for case_context in build_performance_evidence_case_contexts()
    )


def build_performance_evidence_core_benchmark_records() -> list[dict]:
    return deepcopy(list(_build_performance_evidence_core_benchmark_records_cached()))


def _augment_review_fields(case_context, core_record: dict) -> dict:
    record = dict(core_record)
    if not record["representative_review_case"]:
        return record

    record.update(_measure_review_timings(case_context))
    positive_threshold_pass = bool(
        record["counted_supported_benchmark_case"]
        and record["actual_fused_execution"]
        and (
            (record["speedup"] is not None and record["speedup"] >= 1.2)
            or (
                record["memory_reduction"] is not None
                and record["memory_reduction"] >= 0.15
            )
        )
    )
    diagnosis_reasons = (
        _diagnosis_reasons(record)
        if record["counted_supported_benchmark_case"] and not positive_threshold_pass
        else []
    )
    benchmark_status = record["benchmark_status"]
    if (
        benchmark_status != PERFORMANCE_EVIDENCE_STATUS_EXCLUDED
        and not positive_threshold_pass
        and diagnosis_reasons
    ):
        benchmark_status = PERFORMANCE_EVIDENCE_STATUS_DIAGNOSIS_ONLY

    record.update(
        {
            "positive_threshold_pass": positive_threshold_pass,
            "diagnosis_reasons": diagnosis_reasons,
            "diagnosis_only_case": benchmark_status == PERFORMANCE_EVIDENCE_STATUS_DIAGNOSIS_ONLY,
            "benchmark_status": benchmark_status,
        }
    )
    return record


def build_performance_evidence_benchmark_record(case_context) -> dict:
    return _augment_review_fields(
        case_context, build_performance_evidence_core_benchmark_record(case_context)
    )


@lru_cache(maxsize=1)
def _build_performance_evidence_benchmark_records_cached() -> tuple[dict, ...]:
    contexts_by_workload_id = {
        case_context.metadata["workload_id"]: case_context
        for case_context in build_performance_evidence_case_contexts()
    }
    return tuple(
        _augment_review_fields(
            contexts_by_workload_id[core_record["workload_id"]],
            dict(core_record),
        )
        for core_record in _build_performance_evidence_core_benchmark_records_cached()
    )


def build_performance_evidence_benchmark_records() -> list[dict]:
    return deepcopy(list(_build_performance_evidence_benchmark_records_cached()))


_PHASE31_HYBRID_PHASE3_ROUTED_CLASSES = frozenset(
    {"phase3_unitary_island_fused", "phase3_supported_unfused"}
)

_PHASE31_HYBRID_DECISION_CLASSES = frozenset(
    {"phase3_sufficient", "phase31_justified", "phase31_not_justified_yet"}
)

_PHASE31_HYBRID_DIAGNOSIS_TAGS = frozenset(
    {
        "phase31_positive_gain",
        "limited_channel_native_coverage",
        "hybrid_overhead_dominant",
    }
)


def _prefixed_runtime_bridge_fields(prefix: str, bridge: dict) -> dict:
    return {f"{prefix}{key}": value for key, value in bridge.items()}


def _phase31_decision_priority(decision_class: str) -> int:
    if decision_class == "phase31_justified":
        return 0
    if decision_class == "phase31_not_justified_yet":
        return 1
    return 2


def _hybrid_route_coverage(runtime_result, descriptor_set) -> dict[str, Any]:
    channel_native_partition_count = 0
    phase3_routed_partition_count = 0
    channel_native_member_count = 0
    phase3_routed_member_count = 0
    member_counts = descriptor_set.partition_member_counts
    partition_route_records: list[dict[str, Any]] = []
    for rec in runtime_result.partitions:
        pidx = rec.partition_index
        n_members = int(member_counts[pidx])
        cls = rec.partition_runtime_class
        partition_route_records.append(rec.to_dict(descriptor_set))
        if cls == "phase31_channel_native":
            channel_native_partition_count += 1
            channel_native_member_count += n_members
        elif cls in _PHASE31_HYBRID_PHASE3_ROUTED_CLASSES:
            phase3_routed_partition_count += 1
            phase3_routed_member_count += n_members
    return {
        "channel_native_partition_count": channel_native_partition_count,
        "phase3_routed_partition_count": phase3_routed_partition_count,
        "channel_native_member_count": channel_native_member_count,
        "phase3_routed_member_count": phase3_routed_member_count,
        "hybrid_partition_route_records": partition_route_records,
    }


def _phase31_hybrid_pilot_decision(
    *,
    channel_native_partition_count: int,
    phase3_fused_median_runtime_ms: float,
    phase31_hybrid_median_runtime_ms: float,
) -> tuple[str, str]:
    """Return (decision_class, diagnosis_tag) per P31-S09-E01 mapping.

    Positive gain uses hybrid vs Phase-3-fused wall-clock speedup only. Peak-RSS
    samples remain on the record for observability but are not used here: they
    come from process-wide ``ru_maxrss`` and are order-biased across sequential
    runs in the same process.
    """
    hybrid_ms = phase31_hybrid_median_runtime_ms
    fused_ms = phase3_fused_median_runtime_ms
    hybrid_vs_phase3_speedup = fused_ms / hybrid_ms if hybrid_ms > 0.0 else 0.0
    positive_gain = hybrid_vs_phase3_speedup >= 1.2
    if positive_gain:
        return "phase31_justified", "phase31_positive_gain"
    if channel_native_partition_count == 0:
        return "phase3_sufficient", "limited_channel_native_coverage"
    return "phase31_not_justified_yet", "hybrid_overhead_dominant"


def _measure_phase31_hybrid_timings(case_context) -> dict[str, Any]:
    descriptor_set = case_context.descriptor_set
    parameters = case_context.parameters

    sequential_runtime_ms_samples: list[float] = []
    phase3_fused_runtime_ms_samples: list[float] = []
    phase31_hybrid_runtime_ms_samples: list[float] = []
    sequential_peak_rss_kb_samples: list[int] = []
    phase3_fused_peak_rss_kb_samples: list[int] = []
    phase31_hybrid_peak_rss_kb_samples: list[int] = []

    last_reference_density = None
    last_fused_result = None
    last_hybrid_result = None

    for _ in range(PERFORMANCE_EVIDENCE_REPETITIONS):
        sequential_measurement = measure_sequential_density_reference(
            descriptor_set, parameters
        )
        fused_result = execute_partitioned_density_fused(descriptor_set, parameters)
        hybrid_result = execute_partitioned_density_channel_native_hybrid(
            descriptor_set, parameters
        )
        sequential_runtime_ms_samples.append(sequential_measurement.runtime_ms)
        phase3_fused_runtime_ms_samples.append(fused_result.runtime_ms)
        phase31_hybrid_runtime_ms_samples.append(hybrid_result.runtime_ms)
        sequential_peak_rss_kb_samples.append(sequential_measurement.peak_rss_kb)
        phase3_fused_peak_rss_kb_samples.append(fused_result.peak_rss_kb)
        phase31_hybrid_peak_rss_kb_samples.append(hybrid_result.peak_rss_kb)
        last_reference_density = sequential_measurement.density_matrix
        last_fused_result = fused_result
        last_hybrid_result = hybrid_result

    assert last_reference_density is not None
    assert last_fused_result is not None
    assert last_hybrid_result is not None

    fused_metrics = build_density_comparison_metrics(
        last_fused_result.density_matrix, last_reference_density
    )
    hybrid_metrics = build_density_comparison_metrics(
        last_hybrid_result.density_matrix, last_reference_density
    )

    sequential_median_runtime_ms = float(statistics.median(sequential_runtime_ms_samples))
    phase3_fused_median_runtime_ms = float(
        statistics.median(phase3_fused_runtime_ms_samples)
    )
    phase31_hybrid_median_runtime_ms = float(
        statistics.median(phase31_hybrid_runtime_ms_samples)
    )
    sequential_median_peak_rss_kb = int(statistics.median(sequential_peak_rss_kb_samples))
    phase3_fused_median_peak_rss_kb = int(
        statistics.median(phase3_fused_peak_rss_kb_samples)
    )
    phase31_hybrid_median_peak_rss_kb = int(
        statistics.median(phase31_hybrid_peak_rss_kb_samples)
    )

    return {
        "timing_mode": "median_3",
        "sequential_runtime_ms_samples": sequential_runtime_ms_samples,
        "phase3_fused_runtime_ms_samples": phase3_fused_runtime_ms_samples,
        "phase31_hybrid_runtime_ms_samples": phase31_hybrid_runtime_ms_samples,
        "sequential_peak_rss_kb_samples": sequential_peak_rss_kb_samples,
        "phase3_fused_peak_rss_kb_samples": phase3_fused_peak_rss_kb_samples,
        "phase31_hybrid_peak_rss_kb_samples": phase31_hybrid_peak_rss_kb_samples,
        "sequential_median_runtime_ms": sequential_median_runtime_ms,
        "phase3_fused_median_runtime_ms": phase3_fused_median_runtime_ms,
        "phase31_hybrid_median_runtime_ms": phase31_hybrid_median_runtime_ms,
        "sequential_median_peak_rss_kb": sequential_median_peak_rss_kb,
        "phase3_fused_median_peak_rss_kb": phase3_fused_median_peak_rss_kb,
        "phase31_hybrid_median_peak_rss_kb": phase31_hybrid_median_peak_rss_kb,
        "last_reference_density": last_reference_density,
        "last_fused_result": last_fused_result,
        "last_hybrid_result": last_hybrid_result,
        "fused_metrics": fused_metrics,
        "hybrid_metrics": hybrid_metrics,
    }


def _measure_phase31_hybrid_pilot_timings(case_context) -> dict[str, Any]:
    descriptor_set = case_context.descriptor_set
    parameters = case_context.parameters

    sequential_runtime_ms_samples: list[float] = []
    phase3_fused_runtime_ms_samples: list[float] = []
    phase31_hybrid_runtime_ms_samples: list[float] = []
    sequential_peak_rss_kb_samples: list[int] = []
    phase3_fused_peak_rss_kb_samples: list[int] = []
    phase31_hybrid_peak_rss_kb_samples: list[int] = []

    last_reference_density = None
    last_fused_result = None
    last_hybrid_result = None

    for _ in range(PERFORMANCE_EVIDENCE_REPETITIONS):
        sequential_measurement = measure_sequential_density_reference(
            descriptor_set, parameters
        )
        fused_result = execute_partitioned_density_fused(descriptor_set, parameters)
        hybrid_result = execute_partitioned_density_channel_native_hybrid(
            descriptor_set, parameters
        )
        sequential_runtime_ms_samples.append(sequential_measurement.runtime_ms)
        phase3_fused_runtime_ms_samples.append(fused_result.runtime_ms)
        phase31_hybrid_runtime_ms_samples.append(hybrid_result.runtime_ms)
        sequential_peak_rss_kb_samples.append(sequential_measurement.peak_rss_kb)
        phase3_fused_peak_rss_kb_samples.append(fused_result.peak_rss_kb)
        phase31_hybrid_peak_rss_kb_samples.append(hybrid_result.peak_rss_kb)
        last_reference_density = sequential_measurement.density_matrix
        last_fused_result = fused_result
        last_hybrid_result = hybrid_result

    assert last_reference_density is not None
    assert last_fused_result is not None
    assert last_hybrid_result is not None

    fused_metrics = build_density_comparison_metrics(
        last_fused_result.density_matrix, last_reference_density
    )
    hybrid_metrics = build_density_comparison_metrics(
        last_hybrid_result.density_matrix, last_reference_density
    )

    sequential_median_runtime_ms = float(statistics.median(sequential_runtime_ms_samples))
    phase3_fused_median_runtime_ms = float(
        statistics.median(phase3_fused_runtime_ms_samples)
    )
    phase31_hybrid_median_runtime_ms = float(
        statistics.median(phase31_hybrid_runtime_ms_samples)
    )
    sequential_median_peak_rss_kb = int(statistics.median(sequential_peak_rss_kb_samples))
    phase3_fused_median_peak_rss_kb = int(
        statistics.median(phase3_fused_peak_rss_kb_samples)
    )
    phase31_hybrid_median_peak_rss_kb = int(
        statistics.median(phase31_hybrid_peak_rss_kb_samples)
    )

    return {
        "timing_mode": "median_3",
        "sequential_runtime_ms_samples": sequential_runtime_ms_samples,
        "phase3_fused_runtime_ms_samples": phase3_fused_runtime_ms_samples,
        "phase31_hybrid_runtime_ms_samples": phase31_hybrid_runtime_ms_samples,
        "sequential_peak_rss_kb_samples": sequential_peak_rss_kb_samples,
        "phase3_fused_peak_rss_kb_samples": phase3_fused_peak_rss_kb_samples,
        "phase31_hybrid_peak_rss_kb_samples": phase31_hybrid_peak_rss_kb_samples,
        "sequential_median_runtime_ms": sequential_median_runtime_ms,
        "phase3_fused_median_runtime_ms": phase3_fused_median_runtime_ms,
        "phase31_hybrid_median_runtime_ms": phase31_hybrid_median_runtime_ms,
        "sequential_median_peak_rss_kb": sequential_median_peak_rss_kb,
        "phase3_fused_median_peak_rss_kb": phase3_fused_median_peak_rss_kb,
        "phase31_hybrid_median_peak_rss_kb": phase31_hybrid_median_peak_rss_kb,
        "last_reference_density": last_reference_density,
        "last_fused_result": last_fused_result,
        "last_hybrid_result": last_hybrid_result,
        "fused_metrics": fused_metrics,
        "hybrid_metrics": hybrid_metrics,
    }


def build_phase31_hybrid_pilot_record(case_context) -> dict[str, Any]:
    """One benchmark row: sequential, Phase 3 fused, and hybrid timings plus route and decision fields."""
    record = _base_record(case_context)
    timings = _measure_phase31_hybrid_pilot_timings(case_context)
    fused_bridge = build_runtime_correctness_bridge_fields(
        case_context,
        timings["last_fused_result"],
        timings["last_reference_density"],
        timings["fused_metrics"],
        external_reference_required=record["external_reference_required"],
    )
    hybrid_bridge = build_runtime_correctness_bridge_fields(
        case_context,
        timings["last_hybrid_result"],
        timings["last_reference_density"],
        timings["hybrid_metrics"],
        external_reference_required=record["external_reference_required"],
    )

    route = _hybrid_route_coverage(
        timings["last_hybrid_result"], case_context.descriptor_set
    )
    decision_class, diagnosis_tag = _phase31_hybrid_pilot_decision(
        channel_native_partition_count=route["channel_native_partition_count"],
        phase3_fused_median_runtime_ms=timings["phase3_fused_median_runtime_ms"],
        phase31_hybrid_median_runtime_ms=timings["phase31_hybrid_median_runtime_ms"],
    )

    if decision_class not in _PHASE31_HYBRID_DECISION_CLASSES:
        raise AssertionError("unexpected decision_class {!r}".format(decision_class))
    if diagnosis_tag not in _PHASE31_HYBRID_DIAGNOSIS_TAGS:
        raise AssertionError("unexpected diagnosis_tag {!r}".format(diagnosis_tag))

    record.update(
        {
            "artifact_kind": "phase31_hybrid_pilot",
            "timing_mode": timings["timing_mode"],
            "sequential_runtime_ms_samples": timings["sequential_runtime_ms_samples"],
            "phase3_fused_runtime_ms_samples": timings["phase3_fused_runtime_ms_samples"],
            "phase31_hybrid_runtime_ms_samples": timings["phase31_hybrid_runtime_ms_samples"],
            "sequential_peak_rss_kb_samples": timings["sequential_peak_rss_kb_samples"],
            "phase3_fused_peak_rss_kb_samples": timings["phase3_fused_peak_rss_kb_samples"],
            "phase31_hybrid_peak_rss_kb_samples": timings["phase31_hybrid_peak_rss_kb_samples"],
            "sequential_median_runtime_ms": timings["sequential_median_runtime_ms"],
            "phase3_fused_median_runtime_ms": timings["phase3_fused_median_runtime_ms"],
            "phase31_hybrid_median_runtime_ms": timings["phase31_hybrid_median_runtime_ms"],
            "sequential_median_peak_rss_kb": timings["sequential_median_peak_rss_kb"],
            "phase3_fused_median_peak_rss_kb": timings["phase3_fused_median_peak_rss_kb"],
            "phase31_hybrid_median_peak_rss_kb": timings["phase31_hybrid_median_peak_rss_kb"],
            "decision_class": decision_class,
            "diagnosis_tag": diagnosis_tag,
            **route,
        }
    )
    record.update(_prefixed_runtime_bridge_fields("phase3_fused_", fused_bridge))
    record.update(_prefixed_runtime_bridge_fields("phase31_hybrid_", hybrid_bridge))
    return record


def build_phase31_counted_performance_record(case_context) -> dict[str, Any]:
    """One counted Phase 3.1 matrix row with baseline trio and route coverage."""
    record = _base_phase31_record(case_context)
    timings = _measure_phase31_hybrid_timings(case_context)
    fused_bridge = build_runtime_correctness_bridge_fields(
        case_context,
        timings["last_fused_result"],
        timings["last_reference_density"],
        timings["fused_metrics"],
        external_reference_required=record["external_reference_required"],
    )
    hybrid_bridge = build_runtime_correctness_bridge_fields(
        case_context,
        timings["last_hybrid_result"],
        timings["last_reference_density"],
        timings["hybrid_metrics"],
        external_reference_required=record["external_reference_required"],
    )
    route = _hybrid_route_coverage(
        timings["last_hybrid_result"], case_context.descriptor_set
    )
    decision_class, diagnosis_tag = _phase31_hybrid_pilot_decision(
        channel_native_partition_count=route["channel_native_partition_count"],
        phase3_fused_median_runtime_ms=timings["phase3_fused_median_runtime_ms"],
        phase31_hybrid_median_runtime_ms=timings["phase31_hybrid_median_runtime_ms"],
    )
    if decision_class not in _PHASE31_HYBRID_DECISION_CLASSES:
        raise AssertionError("unexpected decision_class {!r}".format(decision_class))
    if diagnosis_tag not in _PHASE31_HYBRID_DIAGNOSIS_TAGS:
        raise AssertionError("unexpected diagnosis_tag {!r}".format(diagnosis_tag))

    record.update(
        {
            "artifact_kind": "phase31_counted_performance_matrix_row",
            "timing_mode": timings["timing_mode"],
            "sequential_runtime_ms_samples": timings["sequential_runtime_ms_samples"],
            "phase3_fused_runtime_ms_samples": timings["phase3_fused_runtime_ms_samples"],
            "phase31_hybrid_runtime_ms_samples": timings["phase31_hybrid_runtime_ms_samples"],
            "sequential_peak_rss_kb_samples": timings["sequential_peak_rss_kb_samples"],
            "phase3_fused_peak_rss_kb_samples": timings["phase3_fused_peak_rss_kb_samples"],
            "phase31_hybrid_peak_rss_kb_samples": timings["phase31_hybrid_peak_rss_kb_samples"],
            "sequential_median_runtime_ms": timings["sequential_median_runtime_ms"],
            "phase3_fused_median_runtime_ms": timings["phase3_fused_median_runtime_ms"],
            "phase31_hybrid_median_runtime_ms": timings["phase31_hybrid_median_runtime_ms"],
            "sequential_median_peak_rss_kb": timings["sequential_median_peak_rss_kb"],
            "phase3_fused_median_peak_rss_kb": timings["phase3_fused_median_peak_rss_kb"],
            "phase31_hybrid_median_peak_rss_kb": timings["phase31_hybrid_median_peak_rss_kb"],
            "decision_class": decision_class,
            "diagnosis_tag": diagnosis_tag,
            **route,
        }
    )
    record.update(_prefixed_runtime_bridge_fields("phase3_fused_", fused_bridge))
    record.update(_prefixed_runtime_bridge_fields("phase31_hybrid_", hybrid_bridge))
    return record


def build_phase31_break_even_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for record in records:
        table.append(
            {
                "case_name": record["case_name"],
                "workload_id": record["workload_id"],
                "family_name": record["family_name"],
                "qbit_num": record["qbit_num"],
                "noise_pattern": record["noise_pattern"],
                "seed": record["seed"],
                "benchmark_slice": record["benchmark_slice"],
                "decision_class": record["decision_class"],
                "diagnosis_tag": record["diagnosis_tag"],
                "sequential_median_runtime_ms": record["sequential_median_runtime_ms"],
                "phase3_fused_median_runtime_ms": record["phase3_fused_median_runtime_ms"],
                "phase31_hybrid_median_runtime_ms": record["phase31_hybrid_median_runtime_ms"],
                "sequential_median_peak_rss_kb": record["sequential_median_peak_rss_kb"],
                "phase3_fused_median_peak_rss_kb": record["phase3_fused_median_peak_rss_kb"],
                "phase31_hybrid_median_peak_rss_kb": record["phase31_hybrid_median_peak_rss_kb"],
                "channel_native_partition_count": record["channel_native_partition_count"],
                "phase3_routed_partition_count": record["phase3_routed_partition_count"],
                "channel_native_member_count": record["channel_native_member_count"],
                "phase3_routed_member_count": record["phase3_routed_member_count"],
                "phase31_hybrid_internal_reference_pass": record[
                    "phase31_hybrid_internal_reference_pass"
                ],
            }
        )
    table.sort(
        key=lambda row: (
            _phase31_decision_priority(row["decision_class"]),
            row["case_name"],
        )
    )
    return table


def build_phase31_decision_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    break_even_table = build_phase31_break_even_table(records)
    case_names = [row["case_name"] for row in break_even_table]
    decision_class_counts = {
        decision_class: sum(
            row["decision_class"] == decision_class for row in break_even_table
        )
        for decision_class in sorted(_PHASE31_HYBRID_DECISION_CLASSES)
    }
    diagnosis_tag_counts = {
        diagnosis_tag: sum(
            row["diagnosis_tag"] == diagnosis_tag for row in break_even_table
        )
        for diagnosis_tag in sorted(_PHASE31_HYBRID_DIAGNOSIS_TAGS)
    }
    representative_rows = [row for row in break_even_table if row["seed"] == 20260318]
    review_ready_case_table = [
        {
            "case_name": row["case_name"],
            "family_name": row["family_name"],
            "qbit_num": row["qbit_num"],
            "noise_pattern": row["noise_pattern"],
            "decision_class": row["decision_class"],
            "diagnosis_tag": row["diagnosis_tag"],
            "channel_native_partition_count": row["channel_native_partition_count"],
            "phase3_routed_partition_count": row["phase3_routed_partition_count"],
            "phase3_fused_median_runtime_ms": row["phase3_fused_median_runtime_ms"],
            "phase31_hybrid_median_runtime_ms": row["phase31_hybrid_median_runtime_ms"],
        }
        for row in break_even_table
    ]
    justification_map = {
        row["case_name"]: {
            "decision_class": row["decision_class"],
            "diagnosis_tag": row["diagnosis_tag"],
            "channel_native_partition_count": row["channel_native_partition_count"],
            "phase3_routed_partition_count": row["phase3_routed_partition_count"],
        }
        for row in break_even_table
    }
    return {
        "total_cases": len(break_even_table),
        "inventory_match": len(case_names) == len(set(case_names)),
        "decision_vocabulary": sorted(_PHASE31_HYBRID_DECISION_CLASSES),
        "diagnosis_vocabulary": sorted(_PHASE31_HYBRID_DIAGNOSIS_TAGS),
        "decision_class_counts": decision_class_counts,
        "diagnosis_tag_counts": diagnosis_tag_counts,
        "phase3_sufficient_rows": decision_class_counts["phase3_sufficient"],
        "phase31_justified_rows": decision_class_counts["phase31_justified"],
        "phase31_not_justified_yet_rows": decision_class_counts[
            "phase31_not_justified_yet"
        ],
        "break_even_table": break_even_table,
        "justification_map": justification_map,
        "review_ready_case_table": review_ready_case_table,
        "representative_review_cases": [
            row["case_name"] for row in representative_rows
        ],
    }


@lru_cache(maxsize=1)
def _build_phase31_counted_performance_records_cached() -> tuple[dict[str, Any], ...]:
    return tuple(
        build_phase31_counted_performance_record(case_context)
        for case_context in build_phase31_counted_performance_case_contexts()
    )


def build_phase31_counted_performance_records() -> list[dict[str, Any]]:
    return deepcopy(list(_build_phase31_counted_performance_records_cached()))
