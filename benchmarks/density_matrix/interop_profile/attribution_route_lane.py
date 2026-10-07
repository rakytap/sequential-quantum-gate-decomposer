#!/usr/bin/env python3
"""M-F5a task-4: width-4 attribution route tracer (orchestration + apply, no O)."""

from __future__ import annotations

import time
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
    COUNTED_ATTRIBUTION_PROVENANCE_COMMAND,
    COUNTED_ATTRIBUTION_SAMPLES_PER_ROUTE,
    COUNTED_ATTRIBUTION_WARMUP_CALLS,
    OPERATION_COUNT_TASK4,
    PARAMETER_COUNT_TASK4,
    R_STRICT_HANDABACK_SHA256_PREFIX,
    R_STRICT_RAISE_CODE_W4,
    R_STRICT_STATUS_REFUSED,
    SUITE_ID_TASK4_ROUTES,
    TIMED_ROUTE_IDS,
    WORKLOAD_LABEL_TASK4,
)
from benchmarks.density_matrix.interop_profile.interop_lane import (
    THROUGHPUT_DIVISOR,
    build_task_evaluator,
)
from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters
from squander.density_matrix import DensityMatrix, NoisyCircuit
from squander.partitioning.noisy_planner import build_phase3_continuity_partition_descriptor_set
from squander.partitioning.noisy_runtime import (
    NoisyRuntimeExecutionResult,
    execute_partitioned_density,
    execute_partitioned_density_channel_native,
    execute_partitioned_density_channel_native_hybrid,
    execute_partitioned_density_fused,
)
from squander.partitioning.noisy_runtime_errors import NoisyRuntimeValidationError

RouteExecutor = Callable[..., NoisyRuntimeExecutionResult]

ROUTE_IDS_ORDER = ["R-base", "R-fused", "R-strict", "R-hybrid"]

ROUTE_TABLE: dict[str, dict[str, Any]] = {
    "R-base": {
        "entry_symbol": "execute_partitioned_density",
        "apply_label": "C++ NoisyCircuit.apply_to",
        "executor": lambda descriptor_set, parameters: execute_partitioned_density(
            descriptor_set, parameters, allow_fusion=False
        ),
    },
    "R-fused": {
        "entry_symbol": "execute_partitioned_density_fused",
        "apply_label": "the executed apply",
        "executor": lambda descriptor_set, parameters: execute_partitioned_density_fused(
            descriptor_set, parameters
        ),
    },
    "R-strict": {
        "entry_symbol": "execute_partitioned_density_channel_native",
        "apply_label": "numpy Kraus",
        "executor": lambda descriptor_set, parameters: execute_partitioned_density_channel_native(
            descriptor_set, parameters
        ),
    },
    "R-hybrid": {
        "entry_symbol": "execute_partitioned_density_channel_native_hybrid",
        "apply_label": "",  # filled from result.partitions after execution
        "executor": lambda descriptor_set, parameters: execute_partitioned_density_channel_native_hybrid(
            descriptor_set, parameters
        ),
    },
}


def build_width4_attribution_anchor() -> tuple[Any, Any, Mapping[str, Any]]:
    """Build the width-4 continuity descriptor from the task-1 interop evaluator."""
    vqe, _hamiltonian = build_task_evaluator(4)
    bridge = vqe.describe_density_bridge()
    if int(bridge["operation_count"]) != OPERATION_COUNT_TASK4:
        raise ValueError("anchor operation_count mismatch")
    if vqe.get_Parameter_Num() != PARAMETER_COUNT_TASK4:
        raise ValueError("anchor parameter_count mismatch")
    descriptor_set = build_phase3_continuity_partition_descriptor_set(
        vqe, max_partition_qubits=2
    )
    if descriptor_set.workload_id != WORKLOAD_LABEL_TASK4:
        raise ValueError("unexpected workload_id on continuity descriptor")
    return vqe, descriptor_set, bridge


def _format_hybrid_apply_label(partition_runtime_classes: list[str]) -> str:
    counts = Counter(partition_runtime_classes)
    parts = [f"{count}× {runtime_class}" for runtime_class, count in sorted(counts.items())]
    return "; ".join(parts)


def _partition_runtime_classes(result: NoisyRuntimeExecutionResult) -> list[str]:
    return [str(partition.partition_runtime_class) for partition in result.partitions]


@contextmanager
def _apply_primitive_timer(route_id: str) -> Iterator[dict[str, int]]:
    """Harness-side wrap of real apply primitives only (never runtime_ms)."""
    from squander.partitioning import noisy_runtime_channel_native as channel_native_mod

    accumulator = {"apply_ns": 0, "apply_primitive_calls": 0}
    restored: list[tuple[Any, str, Any]] = []

    def _wrap_callable(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start_ns = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                accumulator["apply_ns"] += time.perf_counter_ns() - start_ns
                accumulator["apply_primitive_calls"] += 1

        return wrapped

    def _patch(target: Any, attribute: str) -> None:
        original = getattr(target, attribute)
        setattr(target, attribute, _wrap_callable(original))
        restored.append((target, attribute, original))

    try:
        if route_id in ("R-base", "R-fused"):
            _patch(NoisyCircuit, "apply_to")
        if route_id in ("R-fused", "R-hybrid"):
            _patch(DensityMatrix, "apply_local_unitary")
        if route_id == "R-hybrid":
            _patch(channel_native_mod, "_apply_kraus_bundle")
        yield accumulator
    finally:
        for target, attribute, original in reversed(restored):
            setattr(target, attribute, original)


def _time_route_sample(
    route_id: str,
    executor: RouteExecutor,
    descriptor_set: Any,
    parameters: np.ndarray,
) -> tuple[dict[str, int], NoisyRuntimeExecutionResult | None]:
    result: NoisyRuntimeExecutionResult | None = None
    with _apply_primitive_timer(route_id) as accumulator:
        start_ns = time.perf_counter_ns()
        try:
            result = executor(descriptor_set, parameters)
        except NoisyRuntimeValidationError as exc:
            raise ValueError(
                f"attribution route handback for {route_id}: {exc}"
            ) from exc
        total_ns = time.perf_counter_ns() - start_ns
    if route_id == "R-strict":
        apply_ns = 0
        apply_primitive_calls = 0
    else:
        apply_ns = int(accumulator["apply_ns"])
        apply_primitive_calls = int(accumulator["apply_primitive_calls"])
        if apply_primitive_calls <= 0:
            raise ValueError(
                f"attribution route {route_id} recorded zero apply primitive calls"
            )
        if apply_ns <= 0:
            raise ValueError(f"attribution route {route_id} recorded zero apply time")
        if apply_ns >= total_ns:
            raise ValueError(
                f"attribution route {route_id} apply time must be strictly below wall time"
            )
    orchestration_ns = int(total_ns - apply_ns)
    return (
        {
            "orchestration_ns": orchestration_ns,
            "apply_component_ns": apply_ns,
            "apply_primitive_calls": apply_primitive_calls,
        },
        result,
    )


def build_r_strict_refusal_row(
    descriptor_set: Any,
    parameters: np.ndarray,
) -> dict[str, Any]:
    """Record the live R-strict raise as an ADR-F5A-010 refusal row (no numbers)."""
    spec = ROUTE_TABLE["R-strict"]
    try:
        spec["executor"](descriptor_set, parameters)
    except NoisyRuntimeValidationError as exc:
        code = exc.first_unsupported_condition
        if code != R_STRICT_RAISE_CODE_W4:
            raise ValueError(
                f"attribution route handback for R-strict: unexpected raise code {code!r}"
            ) from exc
        reason = (
            f"STEP_4A_HANDBACK {R_STRICT_HANDABACK_SHA256_PREFIX}; "
            f"{code}; live raise from "
            f"{spec['entry_symbol']}: {exc}"
        )
        return {
            "route_id": "R-strict",
            "entry_symbol": spec["entry_symbol"],
            "apply_label": spec["apply_label"],
            "status": R_STRICT_STATUS_REFUSED,
            "reason": reason,
        }
    raise ValueError(
        "attribution route handback for R-strict: execute_partitioned_density_channel_native returned"
    )


def build_route_row(
    route_id: str,
    descriptor_set: Any,
    parameters: np.ndarray,
    *,
    sample_count: int,
) -> dict[str, Any]:
    if route_id not in ROUTE_TABLE:
        raise ValueError(f"unsupported route_id {route_id!r}")
    if route_id == "R-strict":
        raise ValueError("R-strict must be recorded as a refusal row, not a timed row")
    spec = ROUTE_TABLE[route_id]
    samples: list[dict[str, int]] = []
    last_result: NoisyRuntimeExecutionResult | None = None
    for _ in range(sample_count):
        sample, result = _time_route_sample(
            route_id, spec["executor"], descriptor_set, parameters
        )
        samples.append(sample)
        if result is not None:
            last_result = result

    orch_values = [float(sample["orchestration_ns"]) for sample in samples]
    apply_values = [float(sample["apply_component_ns"]) for sample in samples]
    throughput_values = [value / THROUGHPUT_DIVISOR for value in apply_values]

    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        one_sided_upper_bound_route,
    )

    mean_orch, bound_orch = one_sided_upper_bound_route(orch_values)
    mean_apply, bound_apply = one_sided_upper_bound_route(apply_values)
    mean_tp, bound_tp = one_sided_upper_bound_route(throughput_values)

    apply_label = spec["apply_label"]
    partition_runtime_classes: list[str] | None = None
    if route_id == "R-hybrid":
        if last_result is None:
            raise ValueError("R-hybrid execution did not return a runtime result")
        partition_runtime_classes = _partition_runtime_classes(last_result)
        apply_label = _format_hybrid_apply_label(partition_runtime_classes)

    row: dict[str, Any] = {
        "route_id": route_id,
        "entry_symbol": spec["entry_symbol"],
        "apply_label": apply_label,
        "samples": samples,
        "orchestration": {
            "mean_ns": mean_orch,
            "upper_bound_95_ns": bound_orch,
        },
        "apply_component": {
            "mean_ns": mean_apply,
            "upper_bound_95_ns": bound_apply,
        },
        "throughput": {
            "divisor": THROUGHPUT_DIVISOR,
            "mean_ns_per_op": mean_tp,
            "upper_bound_95_ns_per_op": bound_tp,
        },
    }
    if partition_runtime_classes is not None:
        row["partition_runtime_classes"] = partition_runtime_classes
    return row


def _assert_single_thread_env() -> dict[str, str]:
    from benchmarks.density_matrix.interop_profile.interop_lane import _required_thread_env

    thread_env = _required_thread_env()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if thread_env.get(key, "") != "1":
            raise RuntimeError(f"{key} must be 1 for counted attribution routes")
    return thread_env


def run_counted_attribution_bundle() -> dict[str, Any]:
    """Build the width-4 counted attribution bundle (50 warm-up + 1000 per timed route)."""
    from benchmarks.density_matrix.interop_profile.interop_lane import (
        _pin_lowest_allowed_cpu,
        capture_provenance,
    )

    thread_env = _assert_single_thread_env()
    cpu_id = _pin_lowest_allowed_cpu()
    provenance = capture_provenance(COUNTED_ATTRIBUTION_PROVENANCE_COMMAND)
    provenance = {
        **provenance,
        "warmup_calls": COUNTED_ATTRIBUTION_WARMUP_CALLS,
        "thread_env": thread_env,
        "affinity_cpu": cpu_id,
        "estimator": "arithmetic_mean",
        "bound": "one_sided_95_orchestration_and_apply",
    }

    _vqe, descriptor_set, bridge = build_width4_attribution_anchor()
    param_count = _vqe.get_Parameter_Num()
    parameters = build_initial_parameters(param_count)

    rows: list[dict[str, Any]] = []
    for route_id in TIMED_ROUTE_IDS:
        spec = ROUTE_TABLE[route_id]
        for _ in range(COUNTED_ATTRIBUTION_WARMUP_CALLS):
            _time_route_sample(route_id, spec["executor"], descriptor_set, parameters)
        row = build_route_row(
            route_id,
            descriptor_set,
            parameters,
            sample_count=COUNTED_ATTRIBUTION_SAMPLES_PER_ROUTE,
        )
        rows.append(row)
    rows.append(build_r_strict_refusal_row(descriptor_set, parameters))

    claim_boundary = (
        "task-4 counted attribution routes; milestone_counted=false; "
        "R-hybrid does not wrap NoisyCircuit.apply_to (N-1)"
    )
    bundle = {
        "suite": SUITE_ID_TASK4_ROUTES,
        "qbit_num": 4,
        "milestone_counted": False,
        "workload_label": WORKLOAD_LABEL_TASK4,
        "claim_boundary": claim_boundary,
        "labels": "width-4 attribution routes; QA-007 withheld on routes",
        "clean_start": provenance["clean_start"],
        "provenance": provenance,
        "bridge": {
            "parameter_count": param_count,
            "operation_count": int(bridge["operation_count"]),
            "gate_count": int(bridge["gate_count"]),
            "noise_count": int(bridge["noise_count"]),
            "source_type": bridge.get("source_type"),
        },
        "rows": sorted(rows, key=lambda row: ROUTE_IDS_ORDER.index(row["route_id"])),
    }
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    validate_attribution_route_bundle(bundle)
    return bundle


def run_attribution_route_tracer_bundle(
    *,
    sample_count: int = 3,
    claim_boundary: str = "task-4 attribution tracer; milestone_counted=false",
    labels: str = "width-4 attribution routes; QA-007 withheld on routes",
) -> dict[str, Any]:
    """Build a four-route attribution bundle without publishing O."""
    _vqe, descriptor_set, bridge = build_width4_attribution_anchor()
    param_count = _vqe.get_Parameter_Num()
    parameters = build_initial_parameters(param_count)
    rows: list[dict[str, Any]] = []
    for route_id in TIMED_ROUTE_IDS:
        rows.append(
            build_route_row(route_id, descriptor_set, parameters, sample_count=sample_count)
        )
    rows.append(build_r_strict_refusal_row(descriptor_set, parameters))
    rows.sort(key=lambda row: ROUTE_IDS_ORDER.index(row["route_id"]))
    return {
        "suite": SUITE_ID_TASK4_ROUTES,
        "qbit_num": 4,
        "milestone_counted": False,
        "workload_label": WORKLOAD_LABEL_TASK4,
        "claim_boundary": claim_boundary,
        "labels": labels,
        "bridge": {
            "parameter_count": param_count,
            "operation_count": int(bridge["operation_count"]),
            "gate_count": int(bridge["gate_count"]),
            "noise_count": int(bridge["noise_count"]),
            "source_type": bridge.get("source_type"),
        },
        "rows": rows,
    }
