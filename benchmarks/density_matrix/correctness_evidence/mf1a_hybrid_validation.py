#!/usr/bin/env python3
"""M-F1a provisional hybrid-route exactness evidence (four non-counted cells)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.density_matrix.correctness_evidence.common import DEFAULT_OUTPUT_ROOT
from benchmarks.density_matrix.correctness_evidence.mf1a_q4_baseline_validation import (
    REGENERATION_COMMAND,
    _QA001_REGENERATION_TOLERANCES,
    _QA001_VALUE_KEYS,
    capture_provenance,
    evaluate_mf1a_qa001,
    MF1A_QA001_LAMBDA_MIN_FLOOR,
    MF1A_QA001_MATRIX_TOL,
)
from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters
from benchmarks.density_matrix.planner_surface.common import build_phase2_continuity_vqe
from benchmarks.density_matrix.planner_surface import workloads
from squander.partitioning.noisy_planner import (
    build_phase3_continuity_partition_descriptor_set,
)
from squander.partitioning.noisy_runtime import (
    PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID,
    execute_partitioned_density_channel_native_hybrid,
    execute_sequential_density_reference,
)

SUITE_NAME = "correctness_evidence_mf1a_hybrid_bundle_v1"
MANIFEST_SCHEMA_VERSION = "correctness_evidence_mf1a_hybrid_manifest_v1"
RECORD_SCHEMA_VERSION = "correctness_evidence_mf1a_hybrid_case_v1"
BUNDLE_SCHEMA_VERSION = "correctness_evidence_mf1a_hybrid_bundle_v1"
ARTIFACT_FILENAME = "mf1a_hybrid_bundle.json"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "mf1a" / "hybrid"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / ARTIFACT_FILENAME
ROUTE = PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID
MAX_PARTITION_QUBITS = 2

CLAIM_BOUNDARY = (
    "Provisional hybrid-route slice evidence for phase31_channel_native_hybrid at "
    "anchors 4, 6, 8, and 10 with max_partition_qubits 2; not the frozen M-F1a "
    "milestone denominator. No complete M-F1a, external-protocol, Aer, energy, or "
    "frozen-matrix claim."
)

FROZEN_STRUCTURED_BUILDER_CALLS = (
    {
        "family_name": "phase31_pair_repeat",
        "qbit_num": 8,
        "noise_pattern": "dense",
        "seed": 20260318,
        "max_partition_qubits": MAX_PARTITION_QUBITS,
    },
    {
        "family_name": "phase31_pair_repeat",
        "qbit_num": 10,
        "noise_pattern": "dense",
        "seed": 20260318,
        "max_partition_qubits": MAX_PARTITION_QUBITS,
    },
)

_FROZEN_MANIFEST_CELLS: tuple[dict[str, Any], ...] = (
    {
        "anchor_qbits": 4,
        "workload": "phase2_xxz_hea_q4_continuity",
        "route": ROUTE,
        "max_partition_qubits": MAX_PARTITION_QUBITS,
    },
    {
        "anchor_qbits": 6,
        "workload": "phase2_xxz_hea_q6_continuity",
        "route": ROUTE,
        "max_partition_qubits": MAX_PARTITION_QUBITS,
    },
    {
        "anchor_qbits": 8,
        "workload": "phase31_pair_repeat_q8_dense_seed20260318",
        "route": ROUTE,
        "max_partition_qubits": MAX_PARTITION_QUBITS,
    },
    {
        "anchor_qbits": 10,
        "workload": "phase31_pair_repeat_q10_dense_seed20260318",
        "route": ROUTE,
        "max_partition_qubits": MAX_PARTITION_QUBITS,
    },
)

_NO_PRIOR = object()

HYBRID_REGENERATION_ALLOWLIST = (
    "cases[0].provenance.implementation_revision",
    "cases[1].provenance.implementation_revision",
    "cases[2].provenance.implementation_revision",
    "cases[3].provenance.implementation_revision",
)

_ALLOWED_PARTITION_CLASSES = frozenset(
    {
        "phase31_channel_native",
        "phase3_unitary_island_fused",
        "phase3_supported_unfused",
    }
)
_ALLOWED_ROUTE_REASONS = frozenset(
    {
        "eligible_channel_native_motif",
        "pure_unitary_partition",
        "channel_native_qubit_span",
        "channel_native_support_surface",
    }
)

_MATRIX_MEASURES = (
    "frobenius_norm_diff",
    "max_abs_diff",
    "trace_abs_deviation",
)


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
    return len(revisions) == 4 and len(set(revisions)) == 1


def _allowlisted_revision_difference(
    path: str,
    current: Any,
    prior: Any,
    *,
    current_cases: list[dict[str, Any]],
    prior_cases: list[dict[str, Any]],
) -> bool:
    allowlist = HYBRID_REGENERATION_ALLOWLIST
    return (
        path in allowlist
        and _is_full_git_revision(current)
        and _is_full_git_revision(prior)
        and current != prior
        and _implementation_revisions_uniform(current_cases)
        and _implementation_revisions_uniform(prior_cases)
    )


def classify_near_threshold_measure(measure: str, value: float | None) -> str:
    """Pure Layer 1 §10 band classifier; does not affect pass/fail."""
    if value is None or not np.isfinite(value):
        return "qa001_fail"
    if measure == "lambda_min":
        if value >= -1e-13:
            return "no_finding"
        if value >= MF1A_QA001_LAMBDA_MIN_FLOOR:
            return "finding"
        return "qa001_fail"
    if measure not in _MATRIX_MEASURES:
        raise ValueError(f"Unsupported near-threshold measure '{measure}'")
    if value <= 1e-13:
        return "expected_range"
    if value <= 1e-11:
        return "outside_expected"
    if value <= MF1A_QA001_MATRIX_TOL:
        return "finding"
    return "qa001_fail"


def _finding_record(
    *,
    case: dict[str, Any],
    measure: str,
    value: float,
    cause_hypothesis: str,
) -> dict[str, Any]:
    return {
        "route": case["route"],
        "anchor_qbits": case["anchor_qbits"],
        "workload": case["workload"],
        "measure": measure,
        "value": value,
        "cause_hypothesis": cause_hypothesis,
    }


def _derive_summary_findings(cases: list[dict[str, Any]]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    outside_expected_markers: list[dict[str, Any]] = []
    for case in cases:
        qa001 = case.get("qa001", {})
        for measure in _MATRIX_MEASURES:
            value = qa001.get(measure)
            if value is None:
                continue
            band = classify_near_threshold_measure(measure, float(value))
            if band == "outside_expected":
                outside_expected_markers.append(
                    _finding_record(
                        case=case,
                        measure=measure,
                        value=float(value),
                        cause_hypothesis="residual above 1e-13 outside expected range",
                    )
                )
            elif band == "finding":
                findings.append(
                    _finding_record(
                        case=case,
                        measure=measure,
                        value=float(value),
                        cause_hypothesis="residual above 1e-11 Layer 1 finding band",
                    )
                )
        lambda_min = qa001.get("lambda_min")
        if lambda_min is not None:
            band = classify_near_threshold_measure("lambda_min", float(lambda_min))
            if band == "finding":
                findings.append(
                    _finding_record(
                        case=case,
                        measure="lambda_min",
                        value=float(lambda_min),
                        cause_hypothesis="lambda_min below -1e-13 Layer 1 finding band",
                    )
                )
    return {
        "findings": findings,
        "outside_expected_markers": outside_expected_markers,
    }


def build_manifest() -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "cells": [dict(cell) for cell in _FROZEN_MANIFEST_CELLS],
    }


def validate_manifest_cells(cells: list[dict[str, Any]]) -> None:
    if cells != list(_FROZEN_MANIFEST_CELLS):
        raise ValueError("M-F1a hybrid manifest must match the four frozen cells exactly")


def _serialize_fused_regions(
    fused_regions: tuple[Any, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "partition_index": region.partition_index,
            "candidate_kind": region.candidate_kind,
            "classification": region.classification,
            "reason": region.reason,
            "operation_names": list(region.operation_names),
            "global_target_qbits": list(region.global_target_qbits),
        }
        for region in fused_regions
    ]


def _serialize_partitions(result) -> list[dict[str, Any]]:
    return [
        {
            "partition_index": record.partition_index,
            "partition_runtime_class": record.partition_runtime_class,
            "partition_route_reason": record.partition_route_reason,
        }
        for record in result.partitions
    ]


def _partition_route_summary(partitions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    classes = [row["partition_runtime_class"] for row in partitions]
    reasons = [row["partition_route_reason"] for row in partitions]
    return {
        "runtime_class_counts": dict(sorted(Counter(classes).items())),
        "route_reason_counts": dict(sorted(Counter(reasons).items())),
    }


def _build_continuity_descriptor(anchor_qbits: int):
    vqe, _, _ = build_phase2_continuity_vqe(anchor_qbits)
    return build_phase3_continuity_partition_descriptor_set(
        vqe, max_partition_qubits=MAX_PARTITION_QUBITS
    )


def _build_cell_descriptor(cell: dict[str, Any]):
    anchor_qbits = cell["anchor_qbits"]
    if anchor_qbits in (4, 6):
        return _build_continuity_descriptor(anchor_qbits)
    if anchor_qbits == 8:
        return workloads.build_phase31_structured_descriptor_set(
            **FROZEN_STRUCTURED_BUILDER_CALLS[0]
        )
    if anchor_qbits == 10:
        return workloads.build_phase31_structured_descriptor_set(
            **FROZEN_STRUCTURED_BUILDER_CALLS[1]
        )
    raise ValueError(f"Unsupported hybrid anchor {anchor_qbits}")


def _seed_policy_for_cell(cell: dict[str, Any]) -> str:
    if cell["anchor_qbits"] in (4, 6):
        return "deterministic_workload_no_random_seed"
    return "structured_family_seed_20260318"


def _build_realization(result, descriptor_set) -> dict[str, Any]:
    partitions = _serialize_partitions(result)
    summary = _partition_route_summary(partitions)
    channel_native_partition_count = summary["runtime_class_counts"].get(
        "phase31_channel_native", 0
    )
    return {
        "requested_path": result.requested_runtime_path,
        "realized_path": result.runtime_path,
        "partition_count": len(descriptor_set.partitions),
        "exact_output_present": result.exact_output_present,
        "channel_native_partition_count": channel_native_partition_count,
        "runtime_class_counts": summary["runtime_class_counts"],
        "route_reason_counts": summary["route_reason_counts"],
        "partitions": partitions,
        "fused_regions": _serialize_fused_regions(result.fused_regions),
    }


def build_cases(*, provenance: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    manifest = build_manifest()
    validate_manifest_cells(manifest["cells"])
    run_provenance = capture_provenance() if provenance is None else provenance
    cases: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        descriptor_set = _build_cell_descriptor(cell)
        parameters = build_initial_parameters(descriptor_set.parameter_count)
        result = execute_partitioned_density_channel_native_hybrid(
            descriptor_set, parameters
        )
        reference = execute_sequential_density_reference(descriptor_set, parameters)
        qa001 = evaluate_mf1a_qa001(result.density_matrix, reference)
        cases.append(
            {
                "record_schema_version": RECORD_SCHEMA_VERSION,
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "route": ROUTE,
                "anchor_qbits": cell["anchor_qbits"],
                "workload": cell["workload"],
                "planner_setting": {
                    "max_partition_qubits": descriptor_set.max_partition_qubits
                },
                "parameters": parameters.tolist(),
                "seed_policy": _seed_policy_for_cell(cell),
                "realization": _build_realization(result, descriptor_set),
                "qa001": qa001,
                "milestone_counted": False,
                "completeness_claim": False,
                "claim_boundary": CLAIM_BOUNDARY,
                "provenance": run_provenance,
            }
        )
    return cases


def _regions_for_partition(
    fused_regions: list[dict[str, Any]], partition_index: int
) -> list[dict[str, Any]]:
    return [
        region
        for region in fused_regions
        if region.get("partition_index") == partition_index
    ]


def _route_realization_pass(case: dict[str, Any]) -> bool:
    realization = case.get("realization", {})
    if case.get("route") != ROUTE:
        return False
    if case.get("planner_setting", {}).get("max_partition_qubits") != MAX_PARTITION_QUBITS:
        return False
    if realization.get("requested_path") != ROUTE:
        return False
    if realization.get("realized_path") != ROUTE:
        return False
    if realization.get("exact_output_present") is not True:
        return False

    partition_count = realization.get("partition_count")
    partitions = realization.get("partitions")
    fused_regions = realization.get("fused_regions")
    if not isinstance(partition_count, int) or partition_count <= 0:
        return False
    if not isinstance(partitions, list) or not isinstance(fused_regions, list):
        return False
    if len(partitions) != partition_count:
        return False

    indices = [partition.get("partition_index") for partition in partitions]
    if sorted(indices) != list(range(partition_count)):
        return False
    if len(set(indices)) != partition_count:
        return False

    for region in fused_regions:
        partition_index = region.get("partition_index")
        if (
            not isinstance(partition_index, int)
            or partition_index < 0
            or partition_index >= partition_count
        ):
            return False

    runtime_classes: list[str] = []
    route_reasons: list[str] = []
    channel_native_count = 0

    for partition in partitions:
        partition_class = partition.get("partition_runtime_class")
        reason = partition.get("partition_route_reason")
        if partition_class not in _ALLOWED_PARTITION_CLASSES:
            return False
        if reason not in _ALLOWED_ROUTE_REASONS:
            return False
        runtime_classes.append(partition_class)
        route_reasons.append(reason)
        if partition_class == "phase31_channel_native":
            channel_native_count += 1

        regions = _regions_for_partition(fused_regions, partition["partition_index"])
        motif_regions = [
            region
            for region in regions
            if region.get("candidate_kind") == "channel_native_motif"
        ]
        island_regions = [
            region
            for region in regions
            if region.get("candidate_kind") == "unitary_island"
        ]

        if partition_class == "phase31_channel_native":
            if reason != "eligible_channel_native_motif":
                return False
            if len(motif_regions) != 1:
                return False
            if motif_regions[0].get("classification") != "actually_fused":
                return False
            if island_regions:
                return False
        elif partition_class == "phase3_unitary_island_fused":
            if reason == "eligible_channel_native_motif":
                return False
            if not island_regions or not any(
                region.get("classification") == "actually_fused"
                for region in island_regions
            ):
                return False
            if motif_regions:
                return False
        elif partition_class == "phase3_supported_unfused":
            if reason == "eligible_channel_native_motif":
                return False
            if any(region.get("classification") == "actually_fused" for region in regions):
                return False
            if not any(
                region.get("classification")
                in ("supported_but_unfused", "deferred_or_unsupported_candidate")
                for region in regions
            ):
                return False

    if channel_native_count < 1:
        return False
    if realization.get("channel_native_partition_count") != channel_native_count:
        return False

    expected = _partition_route_summary(partitions)
    if realization.get("runtime_class_counts") != expected["runtime_class_counts"]:
        return False
    if realization.get("route_reason_counts") != expected["route_reason_counts"]:
        return False
    return True


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
        or len(prior_cases) != 4
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
        validate_manifest_cells(
            [
                {
                    "anchor_qbits": case["anchor_qbits"],
                    "workload": case["workload"],
                    "route": case["route"],
                    "max_partition_qubits": case["planner_setting"][
                        "max_partition_qubits"
                    ],
                }
                for case in cases
            ]
        )
        exact_set_pass = len(cases) == 4
    except (KeyError, ValueError):
        exact_set_pass = False

    route_pass = bool(
        exact_set_pass and all(_route_realization_pass(case) for case in cases)
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
    if len(cases) == 4:
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
        "milestone_counted_cases": 0,
        "completeness_claim": False,
        "first_failure": first_failure,
    }
    summary.update(finding_summary)
    bundle = {
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
    return bundle
