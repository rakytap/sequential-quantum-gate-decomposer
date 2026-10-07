#!/usr/bin/env python3
"""Validate M-F5a interop profile bundles (task-1 tracer row)."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

COUNTED_PAIRS_REQUIRED = 1000
WARMUP_PAIRS_REQUIRED = 50
QBIT_WIDTH_REQUIRED = 4
THROUGHPUT_DIVISOR_REQUIRED = 3072
PARTITION_REL_TOL = 0.01
PARTITION_ABS_TOL_NS = 1000
Z_95 = 1.644854
QA008_MEAN_O_ABSOLUTE_MARGIN = 0.02
MEAN_COMPONENT_ABS_TOL_NS = 0.5

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
    """Reject C1 fix bundles that touch forbidden trees (ET-5)."""
    for path in changed_paths:
        normalized = path.replace("\\", "/")
        for prefix in FORBIDDEN_IMPLEMENTATION_PREFIXES:
            if normalized.startswith(prefix) or f"/{prefix}" in normalized:
                raise ValueError(
                    f"interop implementation touched forbidden path: {path!r}"
                )


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance block is required")

    for key in PROVENANCE_REQUIRED_KEYS:
        if key not in provenance:
            raise ValueError(f"provenance.{key} is required")

    if provenance.get("clean_start") is not True:
        raise ValueError("provenance.clean_start must be true for a counted bundle")

    if provenance.get("provenance_pass") is not True:
        raise ValueError("provenance.provenance_pass must be true")

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


def validate_interop_bundle(bundle: Mapping[str, Any]) -> None:
    """Raise ValueError when the bundle violates the task-1 interop contract."""

    serialized = json.dumps(bundle, sort_keys=True)
    if "QA-007 met" in serialized:
        raise ValueError('interop bundle must not contain "QA-007 met"')

    if bundle.get("qbit_num") != QBIT_WIDTH_REQUIRED:
        raise ValueError("interop row width must be 4 qubits for task-1")

    if bundle.get("warmup_pairs") != WARMUP_PAIRS_REQUIRED:
        raise ValueError(f"warmup_pairs must be {WARMUP_PAIRS_REQUIRED}")

    counted_pairs = bundle.get("counted_pairs")
    if counted_pairs != COUNTED_PAIRS_REQUIRED:
        raise ValueError(f"counted_pairs must be {COUNTED_PAIRS_REQUIRED}")

    if bundle.get("clean_start") is not True:
        raise ValueError("clean_start must be true for a counted bundle")

    _validate_provenance(bundle.get("provenance") or {})

    if bundle.get("harness_timer_flag") is not True:
        raise ValueError("harness_timer_flag must be true for counted pairs")

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
    for forbidden in FORBIDDEN_ROW_LABELS:
        if forbidden in label_blob:
            raise ValueError(f"attribution route label {forbidden!r} is forbidden in task-1")

    for forbidden in FORBIDDEN_PUBLIC_ENERGY_SYMBOLS:
        if forbidden in label_blob:
            raise ValueError(f"forbidden timed entry {forbidden!r} in bundle metadata")

    throughput = bundle.get("throughput") or {}
    if throughput.get("divisor") != THROUGHPUT_DIVISOR_REQUIRED:
        raise ValueError("throughput divisor must be 3072 for the 4-qubit HEA cell")
    if "mean_ns_per_op" not in throughput or "upper_bound_95_ns_per_op" not in throughput:
        raise ValueError("throughput mean and one-sided 95% bound are required")

    overhead = bundle.get("overhead") or {}
    if "mean_O" not in overhead or "upper_bound_95_O" not in overhead:
        raise ValueError("overhead mean_O and upper_bound_95_O are required")

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
        throughput_values.append(recomputed["apply_to_ns"] / THROUGHPUT_DIVISOR_REQUIRED)

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

    mean_tp, bound_tp = _one_sided_upper_bound(throughput_values)
    if abs(float(throughput["mean_ns_per_op"]) - mean_tp) > 1e-12:
        raise ValueError("throughput.mean_ns_per_op does not match samples")
    if abs(float(throughput["upper_bound_95_ns_per_op"]) - bound_tp) > 1e-9:
        raise ValueError("throughput.upper_bound_95_ns_per_op does not match samples")

    if abs(float(components["mean_wrapper_ns"]) - float(np_mean(wrapper_values))) > MEAN_COMPONENT_ABS_TOL_NS:
        raise ValueError("components.mean_wrapper_ns does not match samples")
    if abs(float(components["mean_allocate_build_ns"]) - float(np_mean(allocate_values))) > MEAN_COMPONENT_ABS_TOL_NS:
        raise ValueError("components.mean_allocate_build_ns does not match samples")
    if abs(float(components["mean_apply_to_ns"]) - float(np_mean(apply_values))) > MEAN_COMPONENT_ABS_TOL_NS:
        raise ValueError("components.mean_apply_to_ns does not match samples")
    if abs(float(components["mean_contraction_ns"]) - float(np_mean(contraction_values))) > MEAN_COMPONENT_ABS_TOL_NS:
        raise ValueError("components.mean_contraction_ns does not match samples")

    if bundle.get("milestone_counted") is not False:
        raise ValueError("milestone_counted must be false for the task-1 tracer row")


def np_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
