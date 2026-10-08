#!/usr/bin/env python3
"""Validate M-F5a task-4 width-4 attribution route bundles (no O / no lower twin)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.density_matrix.interop_profile.interop_bundle_validation import (
    THROUGHPUT_DIVISOR_REQUIRED,
    THROUGHPUT_DIVISOR_W6_REQUIRED,
    THROUGHPUT_DIVISOR_W8_REQUIRED,
    Z_95,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ATTRIBUTION_ARTIFACT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "interop_profile"
)

ATTRIBUTION_ROUTES_ALLOWED_WIDTHS = (4, 6, 8)
ATTRIBUTION_WIDTH_REFUSAL_MESSAGE = "attribution routes allow widths 4, 6, and 8 only"

ROUTES_ARTIFACT_NAME_W4 = "interop_profile_bundle_routes_w4.json"
ROUTES_ARTIFACT_NAME_W6 = "interop_profile_bundle_routes_w6.json"
ROUTES_ARTIFACT_NAME_W8 = "interop_profile_bundle_routes_w8.json"
ROUTES_ARTIFACT_BY_WIDTH = {
    4: ROUTES_ARTIFACT_NAME_W4,
    6: ROUTES_ARTIFACT_NAME_W6,
    8: ROUTES_ARTIFACT_NAME_W8,
}
ALL_ROUTES_ARTIFACT_NAMES = frozenset(ROUTES_ARTIFACT_BY_WIDTH.values())

_IMPLEMENTATION_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_DENSITY_MATRIX_CPP_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SUITE_ID_TASK4_ROUTES = "interop_attribution_routes_task4_w4_v1"
SUITE_ID_TASK5_ROUTES_W6 = "interop_attribution_routes_task5_w6_v1"
SUITE_ID_TASK5_ROUTES_W8 = "interop_attribution_routes_task5_w8_v1"
QBIT_WIDTH_TASK4 = 4
OPERATION_COUNT_TASK4 = 12
PARAMETER_COUNT_TASK4 = 18
GATE_COUNT_TASK4 = 9
NOISE_COUNT_TASK4 = 3
WORKLOAD_LABEL_TASK4 = "phase2_xxz_hea_q4_continuity"

OPERATION_COUNT_TASK6 = 18
PARAMETER_COUNT_TASK6 = 30
GATE_COUNT_TASK6 = 15
NOISE_COUNT_TASK6 = 3
WORKLOAD_LABEL_TASK6 = "phase2_xxz_hea_q6_continuity"
PARTITION_COUNT_TASK6 = 7

OPERATION_COUNT_TASK8 = 24
PARAMETER_COUNT_TASK8 = 42
GATE_COUNT_TASK8 = 21
NOISE_COUNT_TASK8 = 3
WORKLOAD_LABEL_TASK8 = "phase2_xxz_hea_q8_continuity"
PARTITION_COUNT_TASK8 = 9

HYBRID_PARTITION_RUNTIME_CLASSES_ANCHOR: tuple[str, ...] = (
    "phase31_channel_native",
    "phase3_unitary_island_fused",
    "phase31_channel_native",
    "phase3_unitary_island_fused",
    "phase3_unitary_island_fused",
)

HYBRID_PARTITION_RUNTIME_CLASSES_W6: tuple[str, ...] = (
    "phase31_channel_native",
    "phase3_unitary_island_fused",
    "phase31_channel_native",
    "phase3_unitary_island_fused",
    "phase3_unitary_island_fused",
    "phase3_unitary_island_fused",
    "phase3_unitary_island_fused",
)

HYBRID_PARTITION_RUNTIME_CLASSES_W8: tuple[str, ...] = (
    "phase31_channel_native",
    "phase3_unitary_island_fused",
    "phase31_channel_native",
    "phase3_unitary_island_fused",
    "phase3_unitary_island_fused",
    "phase3_unitary_island_fused",
    "phase3_unitary_island_fused",
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

COUNTED_ATTRIBUTION_PROVENANCE_COMMAND_W6 = (
    "taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    "OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output "
    "python benchmarks/density_matrix/interop_profile/validation_pipeline.py "
    "--attribution-routes --width 6 --output "
    "benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w6.json"
)

COUNTED_ATTRIBUTION_PROVENANCE_COMMAND_W8 = (
    "taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    "OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output "
    "python benchmarks/density_matrix/interop_profile/validation_pipeline.py "
    "--attribution-routes --width 8 --output "
    "benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w8.json"
)


@dataclass(frozen=True)
class AttributionWidthProfile:
    qbit_num: int
    suite_id: str
    workload_label: str
    operation_count: int
    parameter_count: int
    gate_count: int
    noise_count: int
    partition_count: int
    throughput_divisor: int
    hybrid_partition_classes: tuple[str, ...]
    counted_provenance_command: str
    routes_artifact_name: str
    record_density_matrix_cpp_sha256: bool


_ATTRIBUTION_WIDTH_PROFILES: dict[int, AttributionWidthProfile] = {
    4: AttributionWidthProfile(
        qbit_num=4,
        suite_id=SUITE_ID_TASK4_ROUTES,
        workload_label=WORKLOAD_LABEL_TASK4,
        operation_count=OPERATION_COUNT_TASK4,
        parameter_count=PARAMETER_COUNT_TASK4,
        gate_count=GATE_COUNT_TASK4,
        noise_count=NOISE_COUNT_TASK4,
        partition_count=5,
        throughput_divisor=THROUGHPUT_DIVISOR_REQUIRED,
        hybrid_partition_classes=HYBRID_PARTITION_RUNTIME_CLASSES_ANCHOR,
        counted_provenance_command=COUNTED_ATTRIBUTION_PROVENANCE_COMMAND,
        routes_artifact_name=ROUTES_ARTIFACT_NAME_W4,
        record_density_matrix_cpp_sha256=False,
    ),
    6: AttributionWidthProfile(
        qbit_num=6,
        suite_id=SUITE_ID_TASK5_ROUTES_W6,
        workload_label=WORKLOAD_LABEL_TASK6,
        operation_count=OPERATION_COUNT_TASK6,
        parameter_count=PARAMETER_COUNT_TASK6,
        gate_count=GATE_COUNT_TASK6,
        noise_count=NOISE_COUNT_TASK6,
        partition_count=PARTITION_COUNT_TASK6,
        throughput_divisor=THROUGHPUT_DIVISOR_W6_REQUIRED,
        hybrid_partition_classes=HYBRID_PARTITION_RUNTIME_CLASSES_W6,
        counted_provenance_command=COUNTED_ATTRIBUTION_PROVENANCE_COMMAND_W6,
        routes_artifact_name=ROUTES_ARTIFACT_NAME_W6,
        record_density_matrix_cpp_sha256=True,
    ),
    8: AttributionWidthProfile(
        qbit_num=8,
        suite_id=SUITE_ID_TASK5_ROUTES_W8,
        workload_label=WORKLOAD_LABEL_TASK8,
        operation_count=OPERATION_COUNT_TASK8,
        parameter_count=PARAMETER_COUNT_TASK8,
        gate_count=GATE_COUNT_TASK8,
        noise_count=NOISE_COUNT_TASK8,
        partition_count=PARTITION_COUNT_TASK8,
        throughput_divisor=THROUGHPUT_DIVISOR_W8_REQUIRED,
        hybrid_partition_classes=HYBRID_PARTITION_RUNTIME_CLASSES_W8,
        counted_provenance_command=COUNTED_ATTRIBUTION_PROVENANCE_COMMAND_W8,
        routes_artifact_name=ROUTES_ARTIFACT_NAME_W8,
        record_density_matrix_cpp_sha256=True,
    ),
}


def attribution_width_profile(qbit_num: int) -> AttributionWidthProfile:
    try:
        return _ATTRIBUTION_WIDTH_PROFILES[qbit_num]
    except (KeyError, TypeError):
        raise ValueError(ATTRIBUTION_WIDTH_REFUSAL_MESSAGE) from None

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


def _validate_hybrid_partition_record(
    row: Mapping[str, Any],
    *,
    expected_classes: tuple[str, ...],
) -> None:
    classes = row.get("partition_runtime_classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("R-hybrid row must record partition_runtime_classes from execution")
    if tuple(str(value) for value in classes) != expected_classes:
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
    throughput_divisor: int,
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

    if throughput.get("divisor") != throughput_divisor:
        raise ValueError(f"throughput divisor must be {throughput_divisor}")
    if "mean_ns_per_op" not in throughput or "upper_bound_95_ns_per_op" not in throughput:
        raise ValueError("throughput mean and upper bound are required")
    if abs(float(throughput["mean_ns_per_op"]) - recomputed_tp_mean) > 1e-12:
        raise ValueError("throughput.mean_ns_per_op does not match samples")
    if abs(float(throughput["upper_bound_95_ns_per_op"]) - recomputed_tp_bound) > 1e-9:
        raise ValueError("throughput.upper_bound_95_ns_per_op does not match samples")


def _validate_apply_primitive_calls(sample: Mapping[str, Any], index: int) -> None:
    calls = sample.get("apply_primitive_calls")
    if calls is None:
        raise ValueError(f"samples[{index}].apply_primitive_calls is required")
    if type(calls) is not int:
        raise ValueError(f"samples[{index}].apply_primitive_calls must be an int")
    if calls <= 0:
        raise ValueError(f"samples[{index}].apply_primitive_calls must be greater than 0")


def _validate_counted_attribution_provenance(
    bundle: Mapping[str, Any],
    profile: AttributionWidthProfile,
) -> None:
    provenance = bundle.get("provenance")
    if provenance is None:
        return
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance block must be a mapping when present")

    if bundle.get("clean_start") is not True:
        raise ValueError("clean_start must be true for a counted attribution bundle")
    if provenance.get("clean_start") is not True:
        raise ValueError("provenance.clean_start must be true for a counted attribution bundle")
    dirty_paths = provenance.get("dirty_paths")
    if dirty_paths is None:
        raise ValueError("provenance.dirty_paths is required for a counted attribution bundle")
    if dirty_paths != []:
        raise ValueError("provenance.dirty_paths must be empty for a counted attribution bundle")

    affinity_cpu = provenance.get("affinity_cpu")
    if affinity_cpu is None:
        raise ValueError("provenance.affinity_cpu is required for a counted attribution bundle")
    if type(affinity_cpu) is not int or affinity_cpu != 0:
        raise ValueError("provenance.affinity_cpu must be 0 for a counted attribution bundle")

    if provenance.get("estimator") != "arithmetic_mean":
        raise ValueError("provenance.estimator must be arithmetic_mean")
    if provenance.get("bound") != "one_sided_95_orchestration_and_apply":
        raise ValueError(
            "provenance.bound must be one_sided_95_orchestration_and_apply"
        )

    implementation_revision = provenance.get("implementation_revision")
    if implementation_revision is None or implementation_revision == "":
        raise ValueError("provenance.implementation_revision is required")
    if not _IMPLEMENTATION_REVISION_RE.fullmatch(str(implementation_revision)):
        raise ValueError("provenance.implementation_revision must be 40 lowercase hex")

    extension_identities = provenance.get("extension_identities")
    if extension_identities is None:
        raise ValueError("provenance.extension_identities is required")
    if not isinstance(extension_identities, list) or not extension_identities:
        raise ValueError("provenance.extension_identities must be a non-empty list")

    if profile.record_density_matrix_cpp_sha256:
        sha_field = provenance.get("density_matrix_cpp_sha256")
        if sha_field is None or sha_field == "":
            raise ValueError("provenance.density_matrix_cpp_sha256 is required")
        if not _DENSITY_MATRIX_CPP_SHA256_RE.fullmatch(str(sha_field)):
            raise ValueError(
                "provenance.density_matrix_cpp_sha256 must be 64 lowercase hex"
            )

    command = provenance.get("command")
    if command != profile.counted_provenance_command:
        raise ValueError("provenance.command must match the counted attribution command")
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
    """Raise ValueError when an attribution route bundle violates the contract."""
    _reject_overhead_ratio_fields(bundle)

    try:
        profile = attribution_width_profile(bundle.get("qbit_num"))
    except ValueError:
        raise ValueError(ATTRIBUTION_WIDTH_REFUSAL_MESSAGE) from None

    if bundle.get("suite") != profile.suite_id:
        raise ValueError(f"suite id must be {profile.suite_id}")
    if bundle.get("milestone_counted") is not False:
        raise ValueError("milestone_counted must be false for attribution route rows")
    if bundle.get("workload_label") != profile.workload_label:
        raise ValueError(f"workload_label must be {profile.workload_label!r}")

    bridge = bundle.get("bridge") or {}
    if bridge.get("parameter_count") != profile.parameter_count:
        raise ValueError(f"bridge.parameter_count must be {profile.parameter_count}")
    if bridge.get("operation_count") != profile.operation_count:
        raise ValueError(f"bridge.operation_count must be {profile.operation_count}")
    if bridge.get("gate_count") != profile.gate_count:
        raise ValueError(f"bridge.gate_count must be {profile.gate_count}")
    if bridge.get("noise_count") != profile.noise_count:
        raise ValueError(f"bridge.noise_count must be {profile.noise_count}")

    label_blob = str(bundle.get("claim_boundary", "")) + str(bundle.get("labels", ""))
    _reject_forbidden_claims(label_blob)
    _reject_r_oracle_in_attribution_bundle(bundle, label_blob)
    _validate_counted_attribution_provenance(bundle, profile)
    counted_bundle = bundle.get("provenance") is not None

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
        if counted_bundle:
            min_samples = COUNTED_ATTRIBUTION_SAMPLES_PER_ROUTE
        if len(samples) < min_samples:
            raise ValueError("each timed route row needs sufficient timing samples")
        if len(samples) < 2:
            raise ValueError("each timed route row needs at least two timing samples")
        if counted_bundle and len(samples) != COUNTED_ATTRIBUTION_SAMPLES_PER_ROUTE:
            raise ValueError("each timed route row needs sufficient timing samples")

        orch_values: list[float] = []
        apply_values: list[float] = []
        throughput_values: list[float] = []
        for idx, sample in enumerate(samples):
            orch_ns = int(sample["orchestration_ns"])
            apply_ns = int(sample["apply_component_ns"])
            if orch_ns < 0 or apply_ns <= 0:
                raise ValueError(f"samples[{idx}] orchestration_ns and apply_component_ns invalid")
            if counted_bundle:
                _validate_apply_primitive_calls(sample, idx)
            orch_values.append(float(orch_ns))
            apply_values.append(float(apply_ns))
            throughput_values.append(apply_ns / profile.throughput_divisor)

        mean_orch, bound_orch = one_sided_upper_bound_route(orch_values)
        mean_apply, bound_apply = one_sided_upper_bound_route(apply_values)
        mean_tp, bound_tp = one_sided_upper_bound_route(throughput_values)
        _validate_row_samples(
            row,
            throughput_divisor=profile.throughput_divisor,
            recomputed_orch_mean=mean_orch,
            recomputed_orch_bound=bound_orch,
            recomputed_apply_mean=mean_apply,
            recomputed_apply_bound=bound_apply,
            recomputed_tp_mean=mean_tp,
            recomputed_tp_bound=bound_tp,
        )
        if route_id == "R-hybrid":
            _validate_hybrid_partition_record(
                row, expected_classes=profile.hybrid_partition_classes
            )

    if seen_ids != set(ROUTE_IDS_REQUIRED):
        raise ValueError("rows must include R-base, R-fused, R-strict, and R-hybrid")


def validate_r_oracle_claim_boundary(claim_boundary: str) -> None:
    """Require the ADR-F5A-001 E1 sentence on any R-oracle diagnosis row."""
    if E1_ORACLE_REQUIRED_PHRASE not in claim_boundary:
        raise ValueError("R-oracle row requires the E1 claim-boundary sentence")


def _path_inside_attribution_artifacts(resolved: Path) -> bool:
    artifact_dir = DEFAULT_ATTRIBUTION_ARTIFACT_DIR.resolve()
    return artifact_dir == resolved.parent or artifact_dir in resolved.parents


def resolve_attribution_output_path(output: Any, *, width: int = QBIT_WIDTH_TASK4) -> Path:
    """Refuse illegal attribution-route output paths before any route timing."""
    try:
        profile = attribution_width_profile(width)
    except ValueError:
        raise ValueError(ATTRIBUTION_WIDTH_REFUSAL_MESSAGE) from None

    path = Path(output)
    resolved = path.resolve()
    name = path.name

    if name in COUNTED_E_VQE_ARTIFACT_NAMES:
        raise ValueError(
            f"attribution routes must not write counted E-VQE bundle {name!r}"
        )
    for other_width, other_name in ROUTES_ARTIFACT_BY_WIDTH.items():
        if other_width != profile.qbit_num and name == other_name:
            raise ValueError(
                f"width {profile.qbit_num} must not write attribution routes bundle {other_name!r}"
            )

    if _path_inside_attribution_artifacts(resolved):
        if resolved.name != profile.routes_artifact_name:
            raise ValueError(
                f"width {profile.qbit_num} must write only {profile.routes_artifact_name!r} "
                f"under {DEFAULT_ATTRIBUTION_ARTIFACT_DIR}"
            )
    return resolved


def qa008_route_categorical_exact(
    committed: Mapping[str, Any],
    regenerated: Mapping[str, Any],
) -> None:
    """QA-008 fitness: categorical leaves must match exactly (task-5 §4)."""

    def is_run_identity_path(path: tuple[str, ...]) -> bool:
        if len(path) < 2 or path[0] != "provenance":
            return False
        if path[1] == "implementation_revision":
            return True
        if path[1] == "extension_identities" and len(path) == 4 and path[3] == "sha256":
            return True
        if path[1] == "density_matrix_cpp_sha256":
            return True
        return False

    def is_timing_path(path: tuple[str, ...]) -> bool:
        if not path or path[0] != "rows" or len(path) < 4:
            return False
        if path[2] == "samples" and path[-1] in ("orchestration_ns", "apply_component_ns"):
            return True
        if path[2] in ("orchestration", "apply_component") and path[3] in (
            "mean_ns",
            "upper_bound_95_ns",
        ):
            return True
        if path[2] == "throughput" and path[3] in (
            "mean_ns_per_op",
            "upper_bound_95_ns_per_op",
        ):
            return True
        return False

    def walk(left: Any, right: Any, path: tuple[str, ...]) -> None:
        if is_timing_path(path) or is_run_identity_path(path):
            return
        if type(left) is not type(right):
            raise ValueError(f"QA-008 categorical mismatch at {'.'.join(path)}")
        if isinstance(left, dict):
            left_keys = set(left.keys())
            right_keys = set(right.keys())
            if left_keys != right_keys:
                raise ValueError(f"QA-008 key set mismatch at {'.'.join(path)}")
            for key in sorted(left_keys):
                walk(left[key], right[key], path + (str(key),))
            return
        if isinstance(left, list):
            if len(left) != len(right):
                raise ValueError(f"QA-008 list length mismatch at {'.'.join(path)}")
            for index, (l_item, r_item) in enumerate(zip(left, right)):
                walk(l_item, r_item, path + (str(index),))
            return
        if left != right:
            raise ValueError(f"QA-008 categorical mismatch at {'.'.join(path)}")

    walk(committed, regenerated, ())


def refuse_evqe_output_with_routes_name(output: Any) -> None:
    """Refuse route artifact names on the E-VQE path (no --attribution-routes)."""
    from pathlib import Path

    resolved = Path(output) if output is not None else None
    if resolved is not None and "_routes_" in resolved.name:
        raise ValueError(
            f"E-VQE interop path must not write attribution route bundle {resolved.name!r}"
        )
