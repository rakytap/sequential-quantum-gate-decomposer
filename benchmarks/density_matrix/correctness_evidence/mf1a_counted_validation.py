#!/usr/bin/env python3
"""M-F1a counted manifest exactness evidence (16-cell ADR-F1A-001 denominator)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from benchmarks.density_matrix.correctness_evidence.common import DEFAULT_OUTPUT_ROOT
from benchmarks.density_matrix.correctness_evidence import mf1a_baseline_validation as mf1a_baseline
from benchmarks.density_matrix.correctness_evidence import mf1a_fused_validation as mf1a_fused
from benchmarks.density_matrix.correctness_evidence import mf1a_hybrid_validation as mf1a_hybrid
from benchmarks.density_matrix.correctness_evidence import (
    mf1a_q4_baseline_validation as mf1a_q4,
)
from benchmarks.density_matrix.correctness_evidence import mf1a_strict_validation as mf1a_strict
from benchmarks.density_matrix.correctness_evidence.mf1a_baseline_validation import (
    _derive_summary_findings,
    classify_near_threshold_measure,
)
from benchmarks.density_matrix.correctness_evidence.mf1a_q4_baseline_validation import (
    REGENERATION_COMMAND,
    _QA001_REGENERATION_TOLERANCES,
    _QA001_VALUE_KEYS,
    capture_provenance,
    MF1A_QA001_LAMBDA_MIN_FLOOR,
    MF1A_QA001_MATRIX_TOL,
)

SUITE_NAME = "correctness_evidence_mf1a_counted_bundle_v1"
MANIFEST_SCHEMA_VERSION = "correctness_evidence_mf1a_counted_manifest_v1"
BUNDLE_SCHEMA_VERSION = "correctness_evidence_mf1a_counted_bundle_v1"
ARTIFACT_FILENAME = "mf1a_counted_bundle.json"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "mf1a" / "counted"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / ARTIFACT_FILENAME

COMPLETENESS_CLAIM = False

CLAIM_BOUNDARY = (
    "Counted M-F1a denominator evidence for the ADR-F1A-001 route-by-anchor set: "
    "partitioned_density_descriptor_baseline, "
    "partitioned_density_descriptor_fused_unitary_islands, phase31_channel_native, "
    "and phase31_channel_native_hybrid at anchors 4, 6, 8, and 10 with "
    "max_partition_qubits 2, each against execute_sequential_density_reference under "
    "QA-001. No complete M-F1a, state-vector, external-protocol, Aer, energy, or "
    "frozen-matrix claim."
)

_COUNTED_MANIFEST_CELLS: tuple[dict[str, Any], ...] = (
    {
        "route": "partitioned_density_descriptor_baseline",
        "anchor_qbits": 4,
        "workload": "phase2_xxz_hea_q4_continuity",
        "max_partition_qubits": 2,
        "seed_policy": "deterministic_workload_no_random_seed",
        "parameter_count": 18,
    },
    {
        "route": "partitioned_density_descriptor_baseline",
        "anchor_qbits": 6,
        "workload": "phase2_xxz_hea_q6_continuity",
        "max_partition_qubits": 2,
        "seed_policy": "deterministic_workload_no_random_seed",
        "parameter_count": 30,
    },
    {
        "route": "partitioned_density_descriptor_baseline",
        "anchor_qbits": 8,
        "workload": "phase2_xxz_hea_q8_continuity",
        "max_partition_qubits": 2,
        "seed_policy": "deterministic_workload_no_random_seed",
        "parameter_count": 42,
    },
    {
        "route": "partitioned_density_descriptor_baseline",
        "anchor_qbits": 10,
        "workload": "phase2_xxz_hea_q10_continuity",
        "max_partition_qubits": 2,
        "seed_policy": "deterministic_workload_no_random_seed",
        "parameter_count": 54,
    },
    {
        "route": "partitioned_density_descriptor_fused_unitary_islands",
        "anchor_qbits": 4,
        "workload": "phase2_xxz_hea_q4_continuity",
        "max_partition_qubits": 2,
        "seed_policy": "deterministic_workload_no_random_seed",
        "parameter_count": 18,
    },
    {
        "route": "partitioned_density_descriptor_fused_unitary_islands",
        "anchor_qbits": 6,
        "workload": "phase2_xxz_hea_q6_continuity",
        "max_partition_qubits": 2,
        "seed_policy": "deterministic_workload_no_random_seed",
        "parameter_count": 30,
    },
    {
        "route": "partitioned_density_descriptor_fused_unitary_islands",
        "anchor_qbits": 8,
        "workload": "layered_nearest_neighbor_q8_sparse_seed20260318",
        "max_partition_qubits": 2,
        "seed_policy": "structured_family_seed_20260318",
        "parameter_count": 72,
    },
    {
        "route": "partitioned_density_descriptor_fused_unitary_islands",
        "anchor_qbits": 10,
        "workload": "layered_nearest_neighbor_q10_sparse_seed20260318",
        "max_partition_qubits": 2,
        "seed_policy": "structured_family_seed_20260318",
        "parameter_count": 120,
    },
    {
        "route": "phase31_channel_native",
        "anchor_qbits": 4,
        "workload": "phase31_local_support_q4_spectator_embedding_smoke",
        "max_partition_qubits": 2,
        "seed_policy": "deterministic_workload_no_random_seed",
        "parameter_count": 18,
    },
    {
        "route": "phase31_channel_native",
        "anchor_qbits": 6,
        "workload": "mf1a_strict_spectator_embed_q6",
        "max_partition_qubits": 2,
        "seed_policy": "deterministic_workload_no_random_seed",
        "parameter_count": 27,
    },
    {
        "route": "phase31_channel_native",
        "anchor_qbits": 8,
        "workload": "mf1a_strict_spectator_embed_q8",
        "max_partition_qubits": 2,
        "seed_policy": "deterministic_workload_no_random_seed",
        "parameter_count": 36,
    },
    {
        "route": "phase31_channel_native",
        "anchor_qbits": 10,
        "workload": "mf1a_strict_spectator_embed_q10",
        "max_partition_qubits": 2,
        "seed_policy": "deterministic_workload_no_random_seed",
        "parameter_count": 45,
    },
    {
        "route": "phase31_channel_native_hybrid",
        "anchor_qbits": 4,
        "workload": "phase2_xxz_hea_q4_continuity",
        "max_partition_qubits": 2,
        "seed_policy": "deterministic_workload_no_random_seed",
        "parameter_count": 18,
    },
    {
        "route": "phase31_channel_native_hybrid",
        "anchor_qbits": 6,
        "workload": "phase2_xxz_hea_q6_continuity",
        "max_partition_qubits": 2,
        "seed_policy": "deterministic_workload_no_random_seed",
        "parameter_count": 30,
    },
    {
        "route": "phase31_channel_native_hybrid",
        "anchor_qbits": 8,
        "workload": "phase31_pair_repeat_q8_dense_seed20260318",
        "max_partition_qubits": 2,
        "seed_policy": "structured_family_seed_20260318",
        "parameter_count": 138,
    },
    {
        "route": "phase31_channel_native_hybrid",
        "anchor_qbits": 10,
        "workload": "phase31_pair_repeat_q10_dense_seed20260318",
        "max_partition_qubits": 2,
        "seed_policy": "structured_family_seed_20260318",
        "parameter_count": 228,
    },
)

COUNTED_REGENERATION_ALLOWLIST = tuple(
    f"cases[{index}].provenance.implementation_revision" for index in range(16)
)

_ROW_GATE_MODULES = (
    (mf1a_q4,)
    + (mf1a_baseline,) * 3
    + (mf1a_fused,) * 4
    + (mf1a_strict,) * 4
    + (mf1a_hybrid,) * 4
)

_NO_PRIOR = object()


def _is_full_git_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _implementation_revisions_uniform(cases: list[dict[str, Any]]) -> bool:
    revisions = [
        case.get("provenance", {}).get("implementation_revision") for case in cases
    ]
    return len(revisions) == 16 and len(set(revisions)) == 1


def _allowlisted_revision_difference(
    path: str,
    current: Any,
    prior: Any,
    *,
    current_cases: list[dict[str, Any]],
    prior_cases: list[dict[str, Any]],
) -> bool:
    return (
        path in COUNTED_REGENERATION_ALLOWLIST
        and _is_full_git_revision(current)
        and _is_full_git_revision(prior)
        and current != prior
        and _implementation_revisions_uniform(current_cases)
        and _implementation_revisions_uniform(prior_cases)
    )


def _manifest_projection(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "route": case["route"],
        "anchor_qbits": case["anchor_qbits"],
        "workload": case["workload"],
        "max_partition_qubits": case["planner_setting"]["max_partition_qubits"],
        "seed_policy": case["seed_policy"],
        "parameter_count": len(case["parameters"]),
    }


def build_manifest() -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "cells": [dict(cell) for cell in _COUNTED_MANIFEST_CELLS],
    }


def validate_manifest_cells(cells: list[dict[str, Any]]) -> None:
    if cells != list(_COUNTED_MANIFEST_CELLS):
        raise ValueError("M-F1a counted manifest must match the sixteen frozen cells exactly")


def _restamp_case(case: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    stamped = dict(case)
    stamped["milestone_counted"] = bool(
        provenance["clean_start"] and provenance["provenance_pass"]
    )
    stamped["completeness_claim"] = COMPLETENESS_CLAIM
    stamped["claim_boundary"] = CLAIM_BOUNDARY
    return stamped


def build_cases(*, provenance: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    run_provenance = capture_provenance() if provenance is None else provenance
    q4_cases = mf1a_q4.build_cases(provenance=run_provenance)
    baseline_cases = mf1a_baseline.build_cases(provenance=run_provenance)
    fused_cases = mf1a_fused.build_cases(provenance=run_provenance)
    strict_cases = mf1a_strict.build_cases(provenance=run_provenance)
    hybrid_cases = mf1a_hybrid.build_cases(provenance=run_provenance)
    ordered = (
        q4_cases
        + baseline_cases
        + fused_cases
        + strict_cases
        + hybrid_cases
    )
    if len(ordered) != 16:
        raise ValueError("Counted build_cases expected sixteen sibling records")
    return [_restamp_case(case, run_provenance) for case in ordered]


def _load_prior_bundle() -> dict[str, Any] | None:
    if not DEFAULT_OUTPUT_PATH.exists():
        return None
    try:
        return json.loads(DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": BUNDLE_SCHEMA_VERSION, "invalid_prior_bundle": True}


def _case_has_allowlisted_revision_only_difference(
    index: int,
    current_case: dict[str, Any],
    prior_case: dict[str, Any],
    *,
    current_cases: list[dict[str, Any]],
    prior_cases: list[dict[str, Any]],
) -> bool:
    path = f"cases[{index}].provenance.implementation_revision"
    current_value = current_case.get("provenance", {}).get("implementation_revision")
    prior_value = prior_case.get("provenance", {}).get("implementation_revision")
    if current_value == prior_value:
        return False
    return _allowlisted_revision_difference(
        path,
        current_value,
        prior_value,
        current_cases=current_cases,
        prior_cases=prior_cases,
    )


def _regeneration_case_result(
    index: int,
    current_case: dict[str, Any],
    prior_case: dict[str, Any],
    *,
    current_cases: list[dict[str, Any]],
    prior_cases: list[dict[str, Any]],
) -> dict[str, Any] | None:
    prefix = f"cases[{index}]"
    exact_paths = (
        "record_schema_version",
        "manifest_schema_version",
        "route",
        "anchor_qbits",
        "workload",
        "planner_setting",
        "parameters",
        "seed_policy",
        "realization",
        "milestone_counted",
        "completeness_claim",
        "claim_boundary",
    )
    for key in exact_paths:
        if current_case.get(key) != prior_case.get(key):
            return {
                "prior_present": True,
                "pass": False,
                "first_mismatch": f"{prefix}.{key}",
            }
    for key in ("extension_identities", "input_artifact_identities"):
        if current_case["provenance"].get(key) != prior_case.get("provenance", {}).get(
            key
        ):
            return {
                "prior_present": True,
                "pass": False,
                "first_mismatch": f"{prefix}.provenance.{key}",
            }
    for key in (
        "implementation_revision",
        "clean_start",
        "dirty_paths",
        "command",
        "environment",
        "dependencies",
        "provenance_pass",
    ):
        path = f"{prefix}.provenance.{key}"
        current_value = current_case["provenance"].get(key)
        prior_value = prior_case.get("provenance", {}).get(key)
        if _allowlisted_revision_difference(
            path,
            current_value,
            prior_value,
            current_cases=current_cases,
            prior_cases=prior_cases,
        ):
            continue
        if current_value != prior_value:
            return {
                "prior_present": True,
                "pass": False,
                "first_mismatch": path,
            }
    current_qa001_categorical = {
        key: value
        for key, value in current_case["qa001"].items()
        if key not in _QA001_VALUE_KEYS
    }
    prior_qa001_categorical = {
        key: value
        for key, value in prior_case.get("qa001", {}).items()
        if key not in _QA001_VALUE_KEYS
    }
    if current_qa001_categorical != prior_qa001_categorical:
        return {
            "prior_present": True,
            "pass": False,
            "first_mismatch": f"{prefix}.qa001.categorical",
        }
    for key in _QA001_VALUE_KEYS:
        current = current_case["qa001"].get(key)
        prior = prior_case.get("qa001", {}).get(key)
        if (
            current is None
            or prior is None
            or not np.isfinite([current, prior]).all()
            or abs(current - prior) > _QA001_REGENERATION_TOLERANCES[key]
        ):
            return {
                "prior_present": True,
                "pass": False,
                "first_mismatch": f"{prefix}.qa001.{key}",
            }
    return None


def _regeneration_result(
    current_cases: list[dict[str, Any]],
    prior_bundle: dict[str, Any] | None,
    *,
    exact_set_pass: bool,
) -> dict[str, Any]:
    if prior_bundle is None:
        return {"prior_present": False, "pass": True, "first_mismatch": None}
    prior_cases = prior_bundle.get("cases")
    if (
        prior_bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or not isinstance(prior_cases, list)
        or len(prior_cases) != 16
    ):
        return {
            "prior_present": True,
            "pass": False,
            "first_mismatch": "bundle_structure",
        }
    for index, (current_case, prior_case) in enumerate(
        zip(current_cases, prior_cases, strict=True)
    ):
        mismatch = _regeneration_case_result(
            index,
            current_case,
            prior_case,
            current_cases=current_cases,
            prior_cases=prior_cases,
        )
        if mismatch is None:
            continue
        if exact_set_pass:
            return mismatch
        revision_path = f"cases[{index}].provenance.implementation_revision"
        if (
            _case_has_allowlisted_revision_only_difference(
                index,
                current_case,
                prior_case,
                current_cases=current_cases,
                prior_cases=prior_cases,
            )
            and mismatch["first_mismatch"] != revision_path
        ):
            return mismatch
        if mismatch["first_mismatch"] == revision_path:
            continue
        return {
            "prior_present": True,
            "pass": False,
            "first_mismatch": "manifest_exact_set",
        }
    return {"prior_present": True, "pass": True, "first_mismatch": None}


def build_artifact_bundle(
    cases: list[dict[str, Any]],
    *,
    prior_bundle: dict[str, Any] | None | object = _NO_PRIOR,
    non_counted_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = build_manifest()
    try:
        projected = [_manifest_projection(case) for case in cases]
        validate_manifest_cells(projected)
        exact_set_pass = len(cases) == 16 and projected == list(_COUNTED_MANIFEST_CELLS)
    except (KeyError, ValueError, TypeError):
        exact_set_pass = False

    route_pass = bool(
        exact_set_pass
        and all(
            gate_module._route_realization_pass(case)
            for case, gate_module in zip(cases, _ROW_GATE_MODULES, strict=True)
        )
    )
    qa001_pass = bool(
        exact_set_pass
        and all(case.get("qa001", {}).get("qa001_pass") for case in cases)
    )
    provenance_pass = bool(
        exact_set_pass
        and cases
        and all(case["provenance"]["provenance_pass"] for case in cases)
    )
    selected_prior = _load_prior_bundle() if prior_bundle is _NO_PRIOR else prior_bundle
    if len(cases) == 16:
        regeneration = _regeneration_result(
            cases, selected_prior, exact_set_pass=exact_set_pass
        )
    else:
        regeneration = {
            "prior_present": selected_prior is not None,
            "pass": False,
            "first_mismatch": "manifest_exact_set",
        }
    gates = (
        ("manifest_exact_set", exact_set_pass),
        ("route_realization", route_pass),
        ("qa001", qa001_pass),
        ("provenance", provenance_pass),
        ("regeneration", regeneration["pass"]),
    )
    first_failure = next((name for name, passed in gates if not passed), None)
    status = "pass" if first_failure is None else "fail"
    finding_summary = _derive_summary_findings(cases) if exact_set_pass else {}
    summary: dict[str, Any] = {
        "total_cases": len(cases),
        "qa001_passes": sum(
            bool(case.get("qa001", {}).get("qa001_pass")) for case in cases
        ),
        "milestone_counted_cases": (
            sum(1 for case in cases if case.get("milestone_counted") is True)
            if exact_set_pass
            else 0
        ),
        "completeness_claim": COMPLETENESS_CLAIM,
        "first_failure": first_failure,
    }
    summary.update(finding_summary)
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "suite_name": SUITE_NAME,
        "status": status,
        "manifest": manifest,
        "tolerances": {
            "frobenius_norm_diff_max": MF1A_QA001_MATRIX_TOL,
            "max_abs_diff_max": MF1A_QA001_MATRIX_TOL,
            "trace_abs_deviation_max": MF1A_QA001_MATRIX_TOL,
            "lambda_min_floor": MF1A_QA001_LAMBDA_MIN_FLOOR,
        },
        "summary": summary,
        "regeneration": regeneration,
        "non_counted_context": non_counted_context or {},
        "cases": cases,
    }
