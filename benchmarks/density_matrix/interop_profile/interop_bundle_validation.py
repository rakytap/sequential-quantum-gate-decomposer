#!/usr/bin/env python3
"""Validate M-F5a interop profile bundles (task-1 tracer row)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

COUNTED_PAIRS_REQUIRED = 1000
WARMUP_PAIRS_REQUIRED = 50
QBIT_WIDTH_REQUIRED = 4
THROUGHPUT_DIVISOR_REQUIRED = 3072
PARTITION_REL_TOL = 0.01
PARTITION_ABS_TOL_NS = 1000
FORBIDDEN_PUBLIC_ENERGY_SYMBOLS = (
    "Optimization_Problem_Batch",
    "Optimization_Problem_Grad",
    "harness_density_lower_energy",
)
FORBIDDEN_ROW_LABELS = ("R-oracle", "R-base", "R-fused", "R-strict", "R-hybrid")


def _require_finite(value: float, field: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {value!r}")


def validate_interop_bundle(bundle: Mapping[str, Any]) -> None:
    """Raise ValueError when the bundle violates the task-1 interop contract."""

    if bundle.get("qbit_num") != QBIT_WIDTH_REQUIRED:
        raise ValueError("interop row width must be 4 qubits for task-1")

    if bundle.get("warmup_pairs") != WARMUP_PAIRS_REQUIRED:
        raise ValueError(f"warmup_pairs must be {WARMUP_PAIRS_REQUIRED}")

    counted_pairs = bundle.get("counted_pairs")
    if counted_pairs != COUNTED_PAIRS_REQUIRED:
        raise ValueError(f"counted_pairs must be {COUNTED_PAIRS_REQUIRED}")

    samples: Sequence[Mapping[str, Any]] = bundle.get("samples") or []
    if len(samples) != COUNTED_PAIRS_REQUIRED:
        raise ValueError("samples length must match counted_pairs")

    label_blob = str(bundle.get("claim_boundary", "")) + str(bundle.get("labels", ""))
    if "QA-007 met" in label_blob:
        raise ValueError('interop bundle must not contain "QA-007 met"')

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

    for idx, sample in enumerate(samples):
        t_public = float(sample["t_public_ns"])
        t_lower = float(sample["t_lower_ns"])
        _require_finite(t_public, f"samples[{idx}].t_public_ns")
        _require_finite(t_lower, f"samples[{idx}].t_lower_ns")
        if t_public <= 0:
            raise ValueError(f"samples[{idx}].t_public_ns must be positive")

        o_i = (t_public - t_lower) / t_public
        _require_finite(o_i, f"samples[{idx}] overhead ratio")

        sub = sample.get("subtimes_ns")
        if sub is None or len(sub) != 6:
            raise ValueError(f"samples[{idx}].subtimes_ns must have six entries")

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

    if bundle.get("milestone_counted") is not False:
        raise ValueError("milestone_counted must be false for the task-1 tracer row")
