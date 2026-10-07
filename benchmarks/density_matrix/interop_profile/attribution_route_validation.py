#!/usr/bin/env python3
"""Validate M-F5a task-4 width-4 attribution route bundles (no O / no lower twin)."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from benchmarks.density_matrix.interop_profile.interop_bundle_validation import (
    THROUGHPUT_DIVISOR_REQUIRED,
    Z_95,
)

SUITE_ID_TASK4_ROUTES = "interop_attribution_routes_task4_w4_v1"
QBIT_WIDTH_TASK4 = 4
OPERATION_COUNT_TASK4 = 12
PARAMETER_COUNT_TASK4 = 18
GATE_COUNT_TASK4 = 9
NOISE_COUNT_TASK4 = 3
WORKLOAD_LABEL_TASK4 = "phase2_xxz_hea_q4_continuity"

HYBRID_PARTITION_RUNTIME_CLASSES_ANCHOR: tuple[str, ...] = (
    "phase31_channel_native",
    "phase3_unitary_island_fused",
    "phase31_channel_native",
    "phase3_unitary_island_fused",
    "phase3_unitary_island_fused",
)

ROUTE_IDS_REQUIRED = ("R-base", "R-fused", "R-strict", "R-hybrid")
TIMED_ROUTE_IDS = ("R-base", "R-fused", "R-hybrid")
R_STRICT_STATUS_REFUSED = "handback_refused"
R_STRICT_HANDABACK_SHA256_PREFIX = "98eec857"
R_STRICT_RAISE_CODE_W4 = "channel_native_noise_presence"

R_STRICT_REFUSAL_ALLOWED_KEYS = frozenset(
    {"route_id", "entry_symbol", "apply_label", "status", "reason"}
)

R_STRICT_FORBIDDEN_TIMING_KEYS = frozenset(
    {
        "samples",
        "orchestration",
        "apply_component",
        "throughput",
        "mean_O",
        "upper_bound_95_O",
        "overhead",
        "t_lower_ns",
        "t_public_ns",
    }
)

COUNTED_ATTRIBUTION_WARMUP_CALLS = 50
COUNTED_ATTRIBUTION_SAMPLES_PER_ROUTE = 1000

COUNTED_ATTRIBUTION_PROVENANCE_COMMAND = (
    "taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    "OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output "
    "python benchmarks/density_matrix/interop_profile/validation_pipeline.py "
    "--attribution-routes --width 4 --output "
    "benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w4.json"
)

FORBIDDEN_SHIPPED_CLAIM_PHRASES = (
    "four-route shipped",
    "req-004 met",
    "speedup",
)

FORBIDDEN_OVERHEAD_KEYS = frozenset(
    {
        "mean_O",
        "upper_bound_95_O",
        "median_O",
        "min_O",
        "max_O",
        "overhead",
        "t_lower_ns",
        "t_public_ns",
        "harness_timer_flag",
    }
)

FORBIDDEN_CLAIM_PHRASES = (
    "qa-007 met",
    "reduction shipped",
    "reduction justified",
    "reduction taken",
    "m-f5a complete",
    "attribution routes profiled",
    "a4 kill",
    "hold-the-line",
)

E1_ORACLE_REQUIRED_PHRASE = "exists only to label C++ apply_to for the E-VQE diagnosis"

COUNTED_E_VQE_ARTIFACT_NAMES = (
    "interop_profile_bundle.json",
    "interop_profile_bundle_w6.json",
    "interop_profile_bundle_w8.json",
)


def _require_finite(value: float, field: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {value!r}")


def one_sided_upper_bound_route(values: Sequence[float]) -> tuple[float, float]:
    arr = [float(value) for value in values]
    mean = float(sum(arr) / len(arr))
    if len(arr) < 2:
        return mean, mean
    variance = sum((value - mean) ** 2 for value in arr) / (len(arr) - 1)
    std = math.sqrt(variance)
    bound = mean + Z_95 * std / math.sqrt(len(arr))
    return mean, bound


def _reject_overhead_ratio_fields(bundle: Mapping[str, Any]) -> None:
    serialized = json.dumps(bundle, sort_keys=True)
    if "QA-007 met" in serialized:
        raise ValueError('attribution bundle must not contain "QA-007 met"')
    if '"O_i"' in serialized or '"mean_O"' in serialized:
        raise ValueError("attribution bundle must not publish an overhead ratio O")

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in FORBIDDEN_OVERHEAD_KEYS:
                    raise ValueError(
                        f"attribution bundle must not contain overhead field {key!r}"
                    )
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(bundle)


def _reject_forbidden_claims(label_blob: str) -> None:
    blob_lower = label_blob.lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        if phrase in blob_lower:
            raise ValueError(f"forbidden claim phrase {phrase!r} in attribution metadata")
    for phrase in FORBIDDEN_SHIPPED_CLAIM_PHRASES:
        if phrase in blob_lower:
            raise ValueError(f"forbidden claim phrase {phrase!r} in attribution metadata")


def _reject_r_oracle_in_attribution_bundle(bundle: Mapping[str, Any], label_blob: str) -> None:
    if "r-oracle" in label_blob.lower():
        raise ValueError("R-oracle diagnosis rows are excluded from attribution route bundles")
    for row in bundle.get("rows") or []:
        if row.get("route_id") == "R-oracle":
            raise ValueError("R-oracle row is forbidden in attribution route bundles")


def _reject_numeric_values_in_r_strict_row(node: Any, path: str) -> None:
    if isinstance(node, bool):
        raise ValueError(f"R-strict refusal row must not contain numbers at {path}")
    if isinstance(node, (int, float)):
        raise ValueError(f"R-strict refusal row must not contain numbers at {path}")
    if isinstance(node, str):
        return
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, bool) or isinstance(key, (int, float)):
                raise ValueError(f"R-strict refusal row must not contain numbers at {path}.key")
            _reject_numeric_values_in_r_strict_row(value, f"{path}.{key}")
        return
    if isinstance(node, list):
        for index, item in enumerate(node):
            _reject_numeric_values_in_r_strict_row(item, f"{path}[{index}]")
        return
    if node is None:
        return
    raise ValueError(f"R-strict refusal row field {path} must be a string")


def _validate_r_strict_refusal_row(row: Mapping[str, Any]) -> None:
    if row.get("status") != R_STRICT_STATUS_REFUSED:
        raise ValueError("R-strict row must have status handback_refused")
    extra_keys = set(row.keys()) - R_STRICT_REFUSAL_ALLOWED_KEYS
    if extra_keys:
        raise ValueError(f"R-strict refusal row must not carry keys {sorted(extra_keys)!r}")
    forbidden_present = R_STRICT_FORBIDDEN_TIMING_KEYS.intersection(row.keys())
    if forbidden_present:
        raise ValueError(
            f"R-strict refusal row must not carry timing fields {sorted(forbidden_present)!r}"
        )
    for key in sorted(R_STRICT_REFUSAL_ALLOWED_KEYS):
        value = row.get(key)
        if type(value) is not str:
            raise ValueError(f"R-strict refusal row field {key!r} must be a string")
    if row.get("entry_symbol") != "execute_partitioned_density_channel_native":
        raise ValueError("R-strict refusal row entry_symbol must be the strict entry")
    if row.get("apply_label") != "numpy Kraus":
        raise ValueError("R-strict refusal row apply_label must be numpy Kraus")
    _reject_numeric_values_in_r_strict_row(dict(row), "R-strict")
    reason = row.get("reason", "")
    if R_STRICT_RAISE_CODE_W4 not in reason:
        raise ValueError("R-strict refusal reason must cite channel_native_noise_presence")
    if R_STRICT_HANDABACK_SHA256_PREFIX not in reason:
        raise ValueError("R-strict refusal reason must cite STEP_4A_HANDBACK 98eec857")


def _reject_timed_r_strict_row(row: Mapping[str, Any]) -> None:
    if row.get("route_id") != "R-strict":
        return
    if row.get("status") == R_STRICT_STATUS_REFUSED:
        return
    raise ValueError("R-strict row must be a refusal row with no timings")


def _validate_hybrid_partition_record(row: Mapping[str, Any]) -> None:
    classes = row.get("partition_runtime_classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("R-hybrid row must record partition_runtime_classes from execution")
    if tuple(str(value) for value in classes) != HYBRID_PARTITION_RUNTIME_CLASSES_ANCHOR:
        raise ValueError("R-hybrid partition_runtime_classes do not match the continuity anchor")
    apply_label = str(row.get("apply_label", ""))
    if apply_label == "the executed class":
        raise ValueError("R-hybrid apply_label must be derived from partition_runtime_classes")
    if "phase31_channel_native" not in apply_label or "phase3_unitary_island_fused" not in apply_label:
        raise ValueError("R-hybrid apply_label must summarize partition_runtime_classes")


def _validate_metric_block(block: Mapping[str, Any], name: str) -> tuple[float, float]:
    if "mean_ns" not in block or "upper_bound_95_ns" not in block:
        raise ValueError(f"{name} mean_ns and upper_bound_95_ns are required")
    mean_ns = float(block["mean_ns"])
    bound_ns = float(block["upper_bound_95_ns"])
    _require_finite(mean_ns, f"{name}.mean_ns")
    _require_finite(bound_ns, f"{name}.upper_bound_95_ns")
    return mean_ns, bound_ns


def _validate_row_samples(
    row: Mapping[str, Any],
    *,
    recomputed_orch_mean: float,
    recomputed_orch_bound: float,
    recomputed_apply_mean: float,
    recomputed_apply_bound: float,
    recomputed_tp_mean: float,
    recomputed_tp_bound: float,
) -> None:
    orch_block = row.get("orchestration") or {}
    apply_block = row.get("apply_component") or {}
    throughput = row.get("throughput") or {}

    mean_orch, bound_orch = _validate_metric_block(orch_block, "orchestration")
    mean_apply, bound_apply = _validate_metric_block(apply_block, "apply_component")

    if abs(mean_orch - recomputed_orch_mean) > 1e-9:
        raise ValueError("orchestration.mean_ns does not match samples")
    if abs(bound_orch - recomputed_orch_bound) > 1e-6:
        raise ValueError("orchestration.upper_bound_95_ns does not match samples")
    if abs(mean_apply - recomputed_apply_mean) > 1e-9:
        raise ValueError("apply_component.mean_ns does not match samples")
    if abs(bound_apply - recomputed_apply_bound) > 1e-6:
        raise ValueError("apply_component.upper_bound_95_ns does not match samples")

    if throughput.get("divisor") != THROUGHPUT_DIVISOR_REQUIRED:
        raise ValueError(f"throughput divisor must be {THROUGHPUT_DIVISOR_REQUIRED}")
    if "mean_ns_per_op" not in throughput or "upper_bound_95_ns_per_op" not in throughput:
        raise ValueError("throughput mean and upper bound are required")
    if abs(float(throughput["mean_ns_per_op"]) - recomputed_tp_mean) > 1e-12:
        raise ValueError("throughput.mean_ns_per_op does not match samples")
    if abs(float(throughput["upper_bound_95_ns_per_op"]) - recomputed_tp_bound) > 1e-9:
        raise ValueError("throughput.upper_bound_95_ns_per_op does not match samples")


def _validate_counted_attribution_provenance(bundle: Mapping[str, Any]) -> None:
    provenance = bundle.get("provenance")
    if provenance is None:
        return
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance block must be a mapping when present")
    command = provenance.get("command")
    if command != COUNTED_ATTRIBUTION_PROVENANCE_COMMAND:
        raise ValueError("provenance.command must match the task-4 counted attribution command")
    if provenance.get("warmup_calls") != COUNTED_ATTRIBUTION_WARMUP_CALLS:
        raise ValueError(
            f"provenance.warmup_calls must be {COUNTED_ATTRIBUTION_WARMUP_CALLS}"
        )
    thread_env = provenance.get("thread_env") or {}
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if thread_env.get(key) != "1":
            raise ValueError(f"provenance.thread_env.{key} must be '1'")


def validate_attribution_route_bundle(bundle: Mapping[str, Any]) -> None:
    """Raise ValueError when a task-4 attribution bundle violates the contract."""
    _reject_overhead_ratio_fields(bundle)

    if bundle.get("suite") != SUITE_ID_TASK4_ROUTES:
        raise ValueError(f"suite id must be {SUITE_ID_TASK4_ROUTES}")
    if bundle.get("qbit_num") != QBIT_WIDTH_TASK4:
        raise ValueError(f"attribution bundle width must be {QBIT_WIDTH_TASK4}")
    if bundle.get("milestone_counted") is not False:
        raise ValueError("milestone_counted must be false for attribution route rows")
    if bundle.get("workload_label") != WORKLOAD_LABEL_TASK4:
        raise ValueError(f"workload_label must be {WORKLOAD_LABEL_TASK4!r}")

    bridge = bundle.get("bridge") or {}
    if bridge.get("parameter_count") != PARAMETER_COUNT_TASK4:
        raise ValueError(f"bridge.parameter_count must be {PARAMETER_COUNT_TASK4}")
    if bridge.get("operation_count") != OPERATION_COUNT_TASK4:
        raise ValueError(f"bridge.operation_count must be {OPERATION_COUNT_TASK4}")
    if bridge.get("gate_count") != GATE_COUNT_TASK4:
        raise ValueError(f"bridge.gate_count must be {GATE_COUNT_TASK4}")
    if bridge.get("noise_count") != NOISE_COUNT_TASK4:
        raise ValueError(f"bridge.noise_count must be {NOISE_COUNT_TASK4}")

    label_blob = str(bundle.get("claim_boundary", "")) + str(bundle.get("labels", ""))
    _reject_forbidden_claims(label_blob)
    _reject_r_oracle_in_attribution_bundle(bundle, label_blob)
    _validate_counted_attribution_provenance(bundle)

    rows = bundle.get("rows")
    if not isinstance(rows, list) or len(rows) != len(ROUTE_IDS_REQUIRED):
        raise ValueError("rows must list exactly four attribution routes")

    seen_ids: set[str] = set()
    for row in rows:
        route_id = row.get("route_id")
        if route_id not in ROUTE_IDS_REQUIRED:
            raise ValueError(f"unknown route_id {route_id!r}")
        if route_id in seen_ids:
            raise ValueError(f"duplicate route_id {route_id!r}")
        seen_ids.add(route_id)

        if not row.get("apply_label"):
            raise ValueError("apply_label is required on each route row")
        if not row.get("entry_symbol"):
            raise ValueError("entry_symbol is required on each route row")

        _reject_timed_r_strict_row(row)
        if route_id == "R-strict":
            _validate_r_strict_refusal_row(row)
            continue

        samples = row.get("samples") or []
        min_samples = 2
        if bundle.get("provenance") is not None:
            min_samples = COUNTED_ATTRIBUTION_SAMPLES_PER_ROUTE
        if len(samples) < min_samples:
            raise ValueError("each timed route row needs sufficient timing samples")
        if len(samples) < 2:
            raise ValueError("each timed route row needs at least two timing samples")

        orch_values: list[float] = []
        apply_values: list[float] = []
        throughput_values: list[float] = []
        for idx, sample in enumerate(samples):
            orch_ns = int(sample["orchestration_ns"])
            apply_ns = int(sample["apply_component_ns"])
            if orch_ns < 0 or apply_ns <= 0:
                raise ValueError(f"samples[{idx}] orchestration_ns and apply_component_ns invalid")
            orch_values.append(float(orch_ns))
            apply_values.append(float(apply_ns))
            throughput_values.append(apply_ns / THROUGHPUT_DIVISOR_REQUIRED)

        mean_orch, bound_orch = one_sided_upper_bound_route(orch_values)
        mean_apply, bound_apply = one_sided_upper_bound_route(apply_values)
        mean_tp, bound_tp = one_sided_upper_bound_route(throughput_values)
        _validate_row_samples(
            row,
            recomputed_orch_mean=mean_orch,
            recomputed_orch_bound=bound_orch,
            recomputed_apply_mean=mean_apply,
            recomputed_apply_bound=bound_apply,
            recomputed_tp_mean=mean_tp,
            recomputed_tp_bound=bound_tp,
        )
        if route_id == "R-hybrid":
            _validate_hybrid_partition_record(row)

    if seen_ids != set(ROUTE_IDS_REQUIRED):
        raise ValueError("rows must include R-base, R-fused, R-strict, and R-hybrid")


def validate_r_oracle_claim_boundary(claim_boundary: str) -> None:
    """Require the ADR-F5A-001 E1 sentence on any R-oracle diagnosis row."""
    if E1_ORACLE_REQUIRED_PHRASE not in claim_boundary:
        raise ValueError("R-oracle row requires the E1 claim-boundary sentence")


def resolve_attribution_output_path(output: Any, *, width: int = QBIT_WIDTH_TASK4) -> Any:
    """Refuse committed E-VQE bundle filenames before any route timing."""
    from pathlib import Path

    resolved = Path(output)
    if resolved.name in COUNTED_E_VQE_ARTIFACT_NAMES:
        raise ValueError(
            f"attribution routes must not write counted E-VQE bundle {resolved.name!r}"
        )
    if width != QBIT_WIDTH_TASK4:
        raise ValueError(
            f"--attribution-routes supports width {QBIT_WIDTH_TASK4} only; got {width}"
        )
    return resolved


def refuse_evqe_output_with_routes_name(output: Any) -> None:
    """Refuse route artifact names on the E-VQE path (no --attribution-routes)."""
    from pathlib import Path

    resolved = Path(output) if output is not None else None
    if resolved is not None and "_routes_" in resolved.name:
        raise ValueError(
            f"E-VQE interop path must not write attribution route bundle {resolved.name!r}"
        )
