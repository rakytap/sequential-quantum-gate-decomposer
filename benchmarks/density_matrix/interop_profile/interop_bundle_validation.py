#!/usr/bin/env python3
"""Validate M-F5a interop profile bundles (task-1 and task-2 tracer rows)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

COUNTED_PAIRS_REQUIRED = 1000
WARMUP_PAIRS_REQUIRED = 50
QBIT_WIDTH_REQUIRED = 4
QBIT_WIDTH_W6_REQUIRED = 6
THROUGHPUT_DIVISOR_REQUIRED = 3072
THROUGHPUT_DIVISOR_W6_REQUIRED = 73728
OPERATION_COUNT_W6_REQUIRED = 18
PARTITION_REL_TOL = 0.01
PARTITION_ABS_TOL_NS = 1000
Z_95 = 1.644854
QA008_MEAN_O_ABSOLUTE_MARGIN = 0.02
MEAN_COMPONENT_ABS_TOL_NS = 0.5
SPIKE_ABS_WRAPPER_THRESHOLD_NS = 20000

# Optimization_Problem_Batch is excluded from the counted inventory because
# Optimization_Interface::optimization_problem_batched dispatches through the
# virtual optimization_problem override (density path exists) while QA-007
# freezes on a single scalar energy, not a batch array (mini-spec §3, F-1).
INTEROP_BATCH_EXCLUSION_NOTE = (
    "Optimization_Problem_Batch excluded: virtual dispatch to "
    "optimization_problem, not a separate density batch branch; QA-007 scalar."
)

FORBIDDEN_IMPLEMENTATION_PREFIXES = (
    "squander/src-cpp/density_matrix/",
    "squander/partitioning/",
    "docs/density_matrix_project/archive/",
    "docs/specs/",
)

FORBIDDEN_PUBLIC_ENERGY_SYMBOLS = (
    "Optimization_Problem_Batch",
    "Optimization_Problem_Grad",
    "harness_density_lower_energy",
)
FORBIDDEN_ROW_LABELS = ("R-oracle", "R-base", "R-fused", "R-strict", "R-hybrid")

FORBIDDEN_W6_CLAIM_PHRASES = (
    "A4 kill",
    "hold-the-line",
    "reduction taken",
)

PROVENANCE_REQUIRED_KEYS = (
    "implementation_revision",
    "clean_start",
    "dirty_paths",
    "command",
    "host",
    "cpu_model",
    "compiler",
    "environment",
    "dependencies",
    "extension_identities",
    "provenance_pass",
)


@dataclass(frozen=True)
class InteropBundleProfile:
    qbit_width: int
    throughput_divisor: int
    suite_id: str
    row_label: str
    provenance_command: str | None = None
    operation_count: int | None = None
    require_w6_overhead_fields: bool = False
    check_w6_forbidden_claims: bool = False


PROFILE_WIDTH_4 = InteropBundleProfile(
    qbit_width=QBIT_WIDTH_REQUIRED,
    throughput_divisor=THROUGHPUT_DIVISOR_REQUIRED,
    suite_id="interop_profile_task1_evqe_4q_v1",
    row_label="task-1",
)

PROFILE_WIDTH_6 = InteropBundleProfile(
    qbit_width=QBIT_WIDTH_W6_REQUIRED,
    throughput_divisor=THROUGHPUT_DIVISOR_W6_REQUIRED,
    suite_id="interop_profile_task2_evqe_6q_v1",
    row_label="task-2",
    operation_count=OPERATION_COUNT_W6_REQUIRED,
    require_w6_overhead_fields=True,
    check_w6_forbidden_claims=True,
)


def _require_finite(value: float, field: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {value!r}")


def _one_sided_upper_bound(values: Sequence[float]) -> tuple[float, float]:
    arr = [float(value) for value in values]
    mean = float(sum(arr) / len(arr))
    if len(arr) < 2:
        return mean, mean
    variance = sum((value - mean) ** 2 for value in arr) / (len(arr) - 1)
    std = math.sqrt(variance)
    bound = mean + Z_95 * std / math.sqrt(len(arr))
    return mean, bound


def np_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def spike_count_abs_wrapper_ns_above_20000(
    samples: Sequence[Mapping[str, Any]],
) -> int:
    count = 0
    for sample in samples:
        delta = abs(int(sample["t_public_ns"]) - int(sample["t_lower_ns"]))
        if delta > SPIKE_ABS_WRAPPER_THRESHOLD_NS:
            count += 1
    return count


def assert_mean_o_within_margin(
    recorded_mean: float,
    reference_mean: float,
    *,
    margin: float = QA008_MEAN_O_ABSOLUTE_MARGIN,
) -> None:
    if abs(recorded_mean - reference_mean) > margin:
        raise ValueError(
            f"mean O margin exceeded: |{recorded_mean} - {reference_mean}| > {margin}"
        )


def _sample_components(sample: Mapping[str, Any]) -> dict[str, int]:
    t_public = int(sample["t_public_ns"])
    t_lower = int(sample["t_lower_ns"])
    subtimes = sample["subtimes_ns"]
    support_outer, construct, lowering, apply_to, contraction, teardown = (
        int(subtimes[0]),
        int(subtimes[1]),
        int(subtimes[2]),
        int(subtimes[3]),
        int(subtimes[4]),
        int(subtimes[5]),
    )
    allocate_build = support_outer + construct + lowering + teardown
    return {
        "wrapper_ns": t_public - t_lower,
        "allocate_build_ns": allocate_build,
        "apply_to_ns": apply_to,
        "contraction_ns": contraction,
    }


def validate_interop_implementation_paths(changed_paths: Sequence[str]) -> None:
    """Reject implementation diffs that touch forbidden trees (ET-5)."""
    for path in changed_paths:
        normalized = path.replace("\\", "/")
        for prefix in FORBIDDEN_IMPLEMENTATION_PREFIXES:
            if normalized.startswith(prefix) or f"/{prefix}" in normalized:
                raise ValueError(
                    f"interop implementation touched forbidden path: {path!r}"
                )


def _validate_provenance(provenance: Mapping[str, Any], profile: InteropBundleProfile) -> None:
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance block is required")

    for key in PROVENANCE_REQUIRED_KEYS:
        if key not in provenance:
            raise ValueError(f"provenance.{key} is required")

    if provenance.get("clean_start") is not True:
        raise ValueError("provenance.clean_start must be true for a counted bundle")

    if provenance.get("provenance_pass") is not True:
        raise ValueError("provenance.provenance_pass must be true")

    if profile.provenance_command is not None:
        command = provenance.get("command")
        if command != profile.provenance_command:
            raise ValueError("provenance.command must match the width-6 counted command")
        if "--width 6" not in str(command):
            raise ValueError("provenance.command must include --width 6")

    identities = provenance.get("extension_identities") or []
    if len(identities) < 2:
        raise ValueError("provenance.extension_identities must include libqgd and wrapper")

    if not provenance.get("host"):
        raise ValueError("provenance.host is required")
    if not provenance.get("cpu_model"):
        raise ValueError("provenance.cpu_model is required")

    compiler = provenance.get("compiler") or {}
    if not compiler.get("executable") or not compiler.get("version_line"):
        raise ValueError("provenance.compiler executable and version_line are required")


def _validate_forbidden_labels(label_blob: str) -> None:
    for forbidden in FORBIDDEN_ROW_LABELS:
        if forbidden in label_blob:
            raise ValueError(f"attribution route label {forbidden!r} is forbidden")

    for forbidden in FORBIDDEN_PUBLIC_ENERGY_SYMBOLS:
        if forbidden in label_blob:
            raise ValueError(f"forbidden timed entry {forbidden!r} in bundle metadata")


def _label_contains_forbidden_w6_phrase(label_blob: str, phrase: str) -> bool:
    if phrase != "reduction taken":
        return phrase in label_blob
    start = 0
    while True:
        pos = label_blob.find(phrase, start)
        if pos == -1:
            return False
        if pos >= 3 and label_blob[pos - 3 : pos] == "no ":
            start = pos + 1
            continue
        return True


def _validate_w6_forbidden_claims(bundle: Mapping[str, Any]) -> None:
    serialized = json.dumps(bundle, sort_keys=True)
    if "QA-007 met" in serialized:
        raise ValueError('interop bundle must not contain "QA-007 met"')

    label_blob = str(bundle.get("claim_boundary", "")) + str(bundle.get("labels", ""))
    for phrase in FORBIDDEN_W6_CLAIM_PHRASES:
        if _label_contains_forbidden_w6_phrase(label_blob, phrase):
            raise ValueError(f"forbidden claim phrase {phrase!r} in bundle metadata")


def _validate_interop_bundle(bundle: Mapping[str, Any], profile: InteropBundleProfile) -> None:
    if not profile.check_w6_forbidden_claims:
        serialized = json.dumps(bundle, sort_keys=True)
        if "QA-007 met" in serialized:
            raise ValueError('interop bundle must not contain "QA-007 met"')
    else:
        _validate_w6_forbidden_claims(bundle)

    if bundle.get("suite") != profile.suite_id:
        raise ValueError(f"suite id must be {profile.suite_id}")

    if bundle.get("qbit_num") != profile.qbit_width:
        raise ValueError(f"interop row width must be {profile.qbit_width} qubits")

    if bundle.get("warmup_pairs") != WARMUP_PAIRS_REQUIRED:
        raise ValueError(f"warmup_pairs must be {WARMUP_PAIRS_REQUIRED}")

    counted_pairs = bundle.get("counted_pairs")
    if counted_pairs != COUNTED_PAIRS_REQUIRED:
        raise ValueError(f"counted_pairs must be {COUNTED_PAIRS_REQUIRED}")

    if bundle.get("clean_start") is not True:
        raise ValueError("clean_start must be true for a counted bundle")

    _validate_provenance(bundle.get("provenance") or {}, profile)

    if bundle.get("harness_timer_flag") is not True:
        raise ValueError("harness_timer_flag must be true for counted pairs")

    if profile.operation_count is not None:
        if bundle.get("operation_count") != profile.operation_count:
            raise ValueError(f"operation_count must be {profile.operation_count}")

    protocol = bundle.get("protocol") or {}
    if protocol.get("pairing") != "paired_not_interleaved":
        raise ValueError("protocol.pairing must be paired_not_interleaved")
    if protocol.get("counted_pairs") != COUNTED_PAIRS_REQUIRED:
        raise ValueError("protocol.counted_pairs must be 1000")

    estimator = bundle.get("estimator") or {}
    if estimator.get("name") != "arithmetic_mean_O_i":
        raise ValueError("estimator.name must be arithmetic_mean_O_i")
    if estimator.get("no_sample_dropped") is not True:
        raise ValueError("estimator.no_sample_dropped must be true")

    workload = bundle.get("workload") or {}
    for key in ("hamiltonian_nnz", "hamiltonian_csr_sha256", "entry", "ansatz"):
        if key not in workload:
            raise ValueError(f"workload.{key} is required")

    qa008 = bundle.get("qa008") or {}
    if qa008.get("categorical_labels_exact") is not True:
        raise ValueError("qa008.categorical_labels_exact must be true")
    if qa008.get("mean_O_absolute_margin") != QA008_MEAN_O_ABSOLUTE_MARGIN:
        raise ValueError("qa008.mean_O_absolute_margin must be 0.02")

    components = bundle.get("components") or {}
    for key in (
        "mean_wrapper_ns",
        "mean_allocate_build_ns",
        "mean_apply_to_ns",
        "mean_contraction_ns",
    ):
        if key not in components:
            raise ValueError(f"components.{key} is required")

    label_blob = str(bundle.get("claim_boundary", "")) + str(bundle.get("labels", ""))
    _validate_forbidden_labels(label_blob)

    throughput = bundle.get("throughput") or {}
    if throughput.get("divisor") != profile.throughput_divisor:
        raise ValueError(
            f"throughput divisor must be {profile.throughput_divisor} for this row"
        )
    if "mean_ns_per_op" not in throughput or "upper_bound_95_ns_per_op" not in throughput:
        raise ValueError("throughput mean and one-sided 95% bound are required")

    overhead = bundle.get("overhead") or {}
    if "mean_O" not in overhead or "upper_bound_95_O" not in overhead:
        raise ValueError("overhead mean_O and upper_bound_95_O are required")

    if profile.require_w6_overhead_fields:
        for key in ("min_O", "max_O", "median_O", "spike_count_abs_wrapper_ns_above_20000"):
            if key not in overhead:
                raise ValueError(f"overhead.{key} is required for width 6")

    samples: Sequence[Mapping[str, Any]] = bundle.get("samples") or []
    if len(samples) != COUNTED_PAIRS_REQUIRED:
        raise ValueError("samples length must match counted_pairs")

    o_values: list[float] = []
    throughput_values: list[float] = []
    wrapper_values: list[float] = []
    allocate_values: list[float] = []
    apply_values: list[float] = []
    contraction_values: list[float] = []

    for idx, sample in enumerate(samples):
        t_public = float(sample["t_public_ns"])
        t_lower = float(sample["t_lower_ns"])
        _require_finite(t_public, f"samples[{idx}].t_public_ns")
        _require_finite(t_lower, f"samples[{idx}].t_lower_ns")
        if t_public <= 0:
            raise ValueError(f"samples[{idx}].t_public_ns must be positive")

        o_i = (t_public - t_lower) / t_public
        _require_finite(o_i, f"samples[{idx}] overhead ratio")
        o_values.append(o_i)

        sub = sample.get("subtimes_ns")
        if sub is None or len(sub) != 6:
            raise ValueError(f"samples[{idx}].subtimes_ns must have six entries")

        recomputed = _sample_components(sample)
        wrapper_values.append(recomputed["wrapper_ns"])
        allocate_values.append(recomputed["allocate_build_ns"])
        apply_values.append(recomputed["apply_to_ns"])
        contraction_values.append(recomputed["contraction_ns"])
        throughput_values.append(recomputed["apply_to_ns"] / profile.throughput_divisor)

        support_outer, construct, lowering, apply_to, contraction, teardown = (
            int(sub[0]),
            int(sub[1]),
            int(sub[2]),
            int(sub[3]),
            int(sub[4]),
            int(sub[5]),
        )
        allocate_build = support_outer + construct + lowering + teardown
        inner_sum = allocate_build + apply_to + contraction
        partition_tol = max(PARTITION_ABS_TOL_NS, PARTITION_REL_TOL * t_lower)
        if abs(inner_sum - t_lower) > partition_tol:
            raise ValueError(
                f"samples[{idx}] partition mismatch: |subtimes - t_lower| "
                f"exceeds {partition_tol} ns"
            )

    mean_o, bound_o = _one_sided_upper_bound(o_values)
    if abs(float(overhead["mean_O"]) - mean_o) > 1e-12:
        raise ValueError("overhead.mean_O does not match samples")
    if abs(float(overhead["upper_bound_95_O"]) - bound_o) > 1e-9:
        raise ValueError("overhead.upper_bound_95_O does not match samples")

    if profile.require_w6_overhead_fields:
        if abs(float(overhead["min_O"]) - min(o_values)) > 1e-12:
            raise ValueError("overhead.min_O does not match samples")
        if abs(float(overhead["max_O"]) - max(o_values)) > 1e-12:
            raise ValueError("overhead.max_O does not match samples")
        if abs(float(overhead["median_O"]) - float(np_median(o_values))) > 1e-12:
            raise ValueError("overhead.median_O does not match samples")
        expected_spike = spike_count_abs_wrapper_ns_above_20000(samples)
        if int(overhead["spike_count_abs_wrapper_ns_above_20000"]) != expected_spike:
            raise ValueError("overhead.spike_count_abs_wrapper_ns_above_20000 mismatch")

    mean_tp, bound_tp = _one_sided_upper_bound(throughput_values)
    if abs(float(throughput["mean_ns_per_op"]) - mean_tp) > 1e-12:
        raise ValueError("throughput.mean_ns_per_op does not match samples")
    if abs(float(throughput["upper_bound_95_ns_per_op"]) - bound_tp) > 1e-9:
        raise ValueError("throughput.upper_bound_95_ns_per_op does not match samples")

    if abs(float(components["mean_wrapper_ns"]) - np_mean(wrapper_values)) > MEAN_COMPONENT_ABS_TOL_NS:
        raise ValueError("components.mean_wrapper_ns does not match samples")
    if abs(float(components["mean_allocate_build_ns"]) - np_mean(allocate_values)) > MEAN_COMPONENT_ABS_TOL_NS:
        raise ValueError("components.mean_allocate_build_ns does not match samples")
    if abs(float(components["mean_apply_to_ns"]) - np_mean(apply_values)) > MEAN_COMPONENT_ABS_TOL_NS:
        raise ValueError("components.mean_apply_to_ns does not match samples")
    if abs(float(components["mean_contraction_ns"]) - np_mean(contraction_values)) > MEAN_COMPONENT_ABS_TOL_NS:
        raise ValueError("components.mean_contraction_ns does not match samples")

    if bundle.get("milestone_counted") is not False:
        raise ValueError(f"milestone_counted must be false for the {profile.row_label} tracer row")


def np_median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def validate_interop_bundle(bundle: Mapping[str, Any]) -> None:
    """Raise ValueError when the bundle violates the task-1 interop contract."""
    _validate_interop_bundle(bundle, PROFILE_WIDTH_4)


def validate_interop_bundle_w6(bundle: Mapping[str, Any]) -> None:
    """Raise ValueError when the bundle violates the task-2 width-6 contract."""
    from benchmarks.density_matrix.interop_profile.interop_lane import (
        COUNTED_REGENERATION_COMMAND_W6,
    )

    profile = InteropBundleProfile(
        qbit_width=QBIT_WIDTH_W6_REQUIRED,
        throughput_divisor=THROUGHPUT_DIVISOR_W6_REQUIRED,
        suite_id=PROFILE_WIDTH_6.suite_id,
        row_label=PROFILE_WIDTH_6.row_label,
        provenance_command=COUNTED_REGENERATION_COMMAND_W6,
        operation_count=OPERATION_COUNT_W6_REQUIRED,
        require_w6_overhead_fields=True,
        check_w6_forbidden_claims=True,
    )
    _validate_interop_bundle(bundle, profile)
