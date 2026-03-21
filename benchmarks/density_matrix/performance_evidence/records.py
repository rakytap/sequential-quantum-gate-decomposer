from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import statistics

from benchmarks.density_matrix.partitioned_runtime.common import (
    PHASE3_RUNTIME_DENSITY_TOL,
    PHASE3_RUNTIME_ENERGY_TOL,
    build_density_comparison_metrics,
    density_energy,
    execute_fused_with_reference,
)
from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
    PERFORMANCE_EVIDENCE_REFERENCE_BACKEND_EXTERNAL,
    PERFORMANCE_EVIDENCE_REFERENCE_BACKEND_INTERNAL,
    PERFORMANCE_EVIDENCE_REPETITIONS,
    PERFORMANCE_EVIDENCE_STATUS_COUNTED,
    PERFORMANCE_EVIDENCE_STATUS_DIAGNOSIS_ONLY,
    PERFORMANCE_EVIDENCE_STATUS_EXCLUDED,
    build_correctness_reference_index,
    build_selected_candidate,
    measure_sequential_density_reference,
)
from benchmarks.density_matrix.performance_evidence.case_selection import (
    build_performance_evidence_case_contexts,
)
from benchmarks.density_matrix.planner_calibration.calibration_records import (
    execute_qiskit_density_reference,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    counted_supported_case,
)
from squander.partitioning.noisy_runtime import execute_partitioned_density_fused


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
        "entry_route": descriptor_set.entry_route,
        "workload_family": descriptor_set.workload_family,
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
    for field in (
        "runtime_schema_version",
        "planner_schema_version",
        "descriptor_schema_version",
        "runtime_path",
        "supported_runtime_case",
        "fallback_used",
        "exact_output_present",
        "runtime_ms",
        "peak_rss_kb",
        "peak_rss_mb",
        "partition_count",
        "descriptor_member_count",
        "gate_count",
        "noise_count",
        "max_partition_span",
        "partition_member_counts",
        "remapped_partition_count",
        "parameter_routing_segment_count",
        "fused_region_count",
        "supported_unfused_region_count",
        "deferred_region_count",
        "fused_gate_count",
        "supported_unfused_gate_count",
        "actual_fused_execution",
        "frobenius_norm_diff",
        "max_abs_diff",
        "internal_reference_pass",
        "trace_deviation",
        "rho_is_valid",
        "rho_is_valid_tol",
        "output_integrity_pass",
        "continuity_energy_required",
        "continuity_energy_error",
        "continuity_energy_pass",
        "external_frobenius_norm_diff",
        "external_max_abs_diff",
        "external_reference_pass",
    ):
        record[field] = correctness_evidence_reference[field]


def performance_evidence_counted_supported_case(record: dict) -> bool:
    if not record["supported_runtime_case"]:
        return False
    if not record["internal_reference_pass"]:
        return False
    if not record["output_integrity_pass"]:
        return False
    if record["continuity_energy_required"] and not record["continuity_energy_pass"]:
        return False
    if record["external_reference_required"] and not record["external_reference_pass"]:
        return False
    return True


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
        runtime_payload = fused_result.to_dict(include_density_matrix=False)

        internal_reference_pass = (
            density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            and density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            and fused_result.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL
            and fused_result.rho_is_valid
        )
        output_integrity_pass = (
            fused_result.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL
            and fused_result.rho_is_valid
        )

        continuity_energy_required = case_context.hamiltonian is not None
        continuity_energy_error = None
        continuity_energy_pass = True
        if continuity_energy_required:
            runtime_energy_real, _ = density_energy(
                case_context.hamiltonian, fused_result.density_matrix_numpy()
            )
            reference_energy_real, _ = density_energy(
                case_context.hamiltonian, reference_density.to_numpy()
            )
            continuity_energy_error = float(abs(runtime_energy_real - reference_energy_real))
            continuity_energy_pass = continuity_energy_error <= PHASE3_RUNTIME_ENERGY_TOL

        external_metrics = None
        external_reference_pass = True
        if record["external_reference_required"]:
            aer_density = execute_qiskit_density_reference(
                case_context.descriptor_set, case_context.parameters
            )
            external_metrics = build_density_comparison_metrics(
                fused_result.density_matrix, aer_density
            )
            external_reference_pass = (
                external_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
                and external_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            )

        record.update(
            {
                "runtime_schema_version": runtime_payload["runtime_schema_version"],
                "planner_schema_version": runtime_payload["planner_schema_version"],
                "descriptor_schema_version": runtime_payload["descriptor_schema_version"],
                "runtime_path": runtime_payload["runtime_path"],
                "supported_runtime_case": (
                    runtime_payload["summary"]["exact_output_present"]
                    and not runtime_payload["summary"]["fallback_used"]
                ),
                "fallback_used": runtime_payload["summary"]["fallback_used"],
                "exact_output_present": runtime_payload["summary"]["exact_output_present"],
                "runtime_ms": runtime_payload["summary"]["runtime_ms"],
                "peak_rss_kb": runtime_payload["summary"]["peak_rss_kb"],
                "peak_rss_mb": float(runtime_payload["summary"]["peak_rss_kb"]) / 1024.0,
                "partition_count": runtime_payload["summary"]["partition_count"],
                "descriptor_member_count": runtime_payload["summary"][
                    "descriptor_member_count"
                ],
                "gate_count": runtime_payload["summary"]["gate_count"],
                "noise_count": runtime_payload["summary"]["noise_count"],
                "max_partition_span": runtime_payload["summary"]["max_partition_span"],
                "partition_member_counts": runtime_payload["summary"][
                    "partition_member_counts"
                ],
                "remapped_partition_count": runtime_payload["summary"][
                    "remapped_partition_count"
                ],
                "parameter_routing_segment_count": runtime_payload["summary"][
                    "parameter_routing_segment_count"
                ],
                "fused_region_count": runtime_payload["summary"]["fused_region_count"],
                "supported_unfused_region_count": runtime_payload["summary"][
                    "supported_unfused_region_count"
                ],
                "deferred_region_count": runtime_payload["summary"][
                    "deferred_region_count"
                ],
                "fused_gate_count": runtime_payload["summary"]["fused_gate_count"],
                "supported_unfused_gate_count": runtime_payload["summary"][
                    "supported_unfused_gate_count"
                ],
                "actual_fused_execution": runtime_payload["summary"][
                    "actual_fused_execution"
                ],
                "frobenius_norm_diff": density_metrics["frobenius_norm_diff"],
                "max_abs_diff": density_metrics["max_abs_diff"],
                "internal_reference_pass": internal_reference_pass,
                "trace_deviation": fused_result.trace_deviation,
                "rho_is_valid": fused_result.rho_is_valid,
                "rho_is_valid_tol": runtime_payload["summary"]["rho_is_valid_tol"],
                "output_integrity_pass": output_integrity_pass,
                "continuity_energy_required": continuity_energy_required,
                "continuity_energy_error": continuity_energy_error,
                "continuity_energy_pass": continuity_energy_pass,
                "external_frobenius_norm_diff": (
                    None
                    if external_metrics is None
                    else external_metrics["frobenius_norm_diff"]
                ),
                "external_max_abs_diff": (
                    None if external_metrics is None else external_metrics["max_abs_diff"]
                ),
                "external_reference_pass": external_reference_pass,
            }
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
