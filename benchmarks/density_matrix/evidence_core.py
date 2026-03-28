"""Shared evidence helpers for correctness and performance pipelines."""

from __future__ import annotations

from typing import Any

from benchmarks.density_matrix.partitioned_runtime.common import (
    PHASE3_RUNTIME_DENSITY_TOL,
    PHASE3_RUNTIME_ENERGY_TOL,
    build_density_comparison_metrics,
    density_energy,
)
from benchmarks.density_matrix.planner_calibration.calibration_records import (
    execute_qiskit_density_reference,
)

# Fields copied from correctness positive records into performance benchmark records
# (via reference index or fused fallback). Must match build_runtime_correctness_bridge_fields keys.
RUNTIME_CORRECTNESS_BRIDGE_FIELD_NAMES: tuple[str, ...] = (
    "runtime_path",
    "supported_runtime_case",
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
)


def counted_supported_case(record: dict[str, Any]) -> bool:
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


def build_runtime_correctness_bridge_fields(
    case_context,
    runtime_result,
    reference_density,
    density_metrics: dict[str, Any],
    *,
    external_reference_required: bool,
    runtime_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize runtime summary, density comparison, and integrity fields shared by evidence records."""
    if runtime_payload is None:
        runtime_payload = runtime_result.to_dict(include_density_matrix=False)

    internal_reference_pass = (
        density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        and density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        and runtime_result.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL
        and runtime_result.rho_is_valid
    )
    output_integrity_pass = (
        runtime_result.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL
        and runtime_result.rho_is_valid
    )

    continuity_energy_required = case_context.hamiltonian is not None
    continuity_energy_error = None
    continuity_energy_pass = True
    if continuity_energy_required:
        runtime_energy_real, _ = density_energy(
            case_context.hamiltonian, runtime_result.density_matrix_numpy()
        )
        reference_energy_real, _ = density_energy(
            case_context.hamiltonian, reference_density.to_numpy()
        )
        continuity_energy_error = float(abs(runtime_energy_real - reference_energy_real))
        continuity_energy_pass = continuity_energy_error <= PHASE3_RUNTIME_ENERGY_TOL

    external_metrics = None
    external_reference_pass = True
    if external_reference_required:
        aer_density = execute_qiskit_density_reference(
            case_context.descriptor_set, case_context.parameters
        )
        external_metrics = build_density_comparison_metrics(
            runtime_result.density_matrix, aer_density
        )
        external_reference_pass = (
            external_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            and external_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        )

    fields: dict[str, Any] = {
        "runtime_path": runtime_payload["runtime_path"],
        "supported_runtime_case": runtime_payload["summary"]["exact_output_present"],
        "exact_output_present": runtime_payload["summary"]["exact_output_present"],
        "runtime_ms": runtime_payload["summary"]["runtime_ms"],
        "peak_rss_kb": runtime_payload["summary"]["peak_rss_kb"],
        "peak_rss_mb": float(runtime_payload["summary"]["peak_rss_kb"]) / 1024.0,
        "partition_count": runtime_payload["summary"]["partition_count"],
        "descriptor_member_count": runtime_payload["summary"]["descriptor_member_count"],
        "gate_count": runtime_payload["summary"]["gate_count"],
        "noise_count": runtime_payload["summary"]["noise_count"],
        "max_partition_span": runtime_payload["summary"]["max_partition_span"],
        "partition_member_counts": runtime_payload["summary"]["partition_member_counts"],
        "remapped_partition_count": runtime_payload["summary"]["remapped_partition_count"],
        "parameter_routing_segment_count": runtime_payload["summary"][
            "parameter_routing_segment_count"
        ],
        "fused_region_count": runtime_payload["summary"]["fused_region_count"],
        "supported_unfused_region_count": runtime_payload["summary"][
            "supported_unfused_region_count"
        ],
        "deferred_region_count": runtime_payload["summary"]["deferred_region_count"],
        "fused_gate_count": runtime_payload["summary"]["fused_gate_count"],
        "supported_unfused_gate_count": runtime_payload["summary"][
            "supported_unfused_gate_count"
        ],
        "actual_fused_execution": runtime_payload["summary"]["actual_fused_execution"],
        "frobenius_norm_diff": density_metrics["frobenius_norm_diff"],
        "max_abs_diff": density_metrics["max_abs_diff"],
        "internal_reference_pass": internal_reference_pass,
        "trace_deviation": runtime_result.trace_deviation,
        "rho_is_valid": runtime_result.rho_is_valid,
        "rho_is_valid_tol": runtime_payload["summary"]["rho_is_valid_tol"],
        "output_integrity_pass": output_integrity_pass,
        "continuity_energy_required": continuity_energy_required,
        "continuity_energy_error": continuity_energy_error,
        "continuity_energy_pass": continuity_energy_pass,
        "external_frobenius_norm_diff": (
            None if external_metrics is None else external_metrics["frobenius_norm_diff"]
        ),
        "external_max_abs_diff": (
            None if external_metrics is None else external_metrics["max_abs_diff"]
        ),
        "external_reference_pass": external_reference_pass,
    }
    if set(fields) != set(RUNTIME_CORRECTNESS_BRIDGE_FIELD_NAMES):
        missing = set(RUNTIME_CORRECTNESS_BRIDGE_FIELD_NAMES) - set(fields)
        extra = set(fields) - set(RUNTIME_CORRECTNESS_BRIDGE_FIELD_NAMES)
        raise AssertionError(f"bridge field keys mismatch: missing={missing!r} extra={extra!r}")
    return fields
