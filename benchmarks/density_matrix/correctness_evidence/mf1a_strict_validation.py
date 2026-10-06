#!/usr/bin/env python3
"""M-F1a provisional strict-route exactness evidence (four non-counted cells)."""

from __future__ import annotations

import json
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
from benchmarks.density_matrix.planner_surface import workloads
from squander.partitioning.noisy_planner import (
    build_canonical_planner_surface_from_operation_specs,
    build_partition_descriptor_set,
)
from squander.partitioning.noisy_runtime import (
    PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    execute_partitioned_density_channel_native,
    execute_sequential_density_reference,
)

SUITE_NAME = "correctness_evidence_mf1a_strict_bundle_v1"
MANIFEST_SCHEMA_VERSION = "correctness_evidence_mf1a_strict_manifest_v1"
RECORD_SCHEMA_VERSION = "correctness_evidence_mf1a_strict_case_v1"
BUNDLE_SCHEMA_VERSION = "correctness_evidence_mf1a_strict_bundle_v1"
ARTIFACT_FILENAME = "mf1a_strict_bundle.json"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "mf1a" / "strict"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / ARTIFACT_FILENAME
ROUTE = PHASE31_RUNTIME_PATH_CHANNEL_NATIVE
MAX_PARTITION_QUBITS = 2

CLAIM_BOUNDARY = (
    "Provisional strict-route slice evidence for phase31_channel_native at anchors 4, "
    "6, 8, and 10 with max_partition_qubits 2; not the frozen M-F1a milestone "
    "denominator. No complete M-F1a, external-protocol, Aer, energy, or "
    "frozen-matrix claim."
)

_FROZEN_MANIFEST_CELLS: tuple[dict[str, Any], ...] = (
    {
        "anchor_qbits": 4,
        "workload": "phase31_local_support_q4_spectator_embedding_smoke",
        "route": ROUTE,
        "max_partition_qubits": MAX_PARTITION_QUBITS,
    },
    {
        "anchor_qbits": 6,
        "workload": "mf1a_strict_spectator_embed_q6",
        "route": ROUTE,
        "max_partition_qubits": MAX_PARTITION_QUBITS,
    },
    {
        "anchor_qbits": 8,
        "workload": "mf1a_strict_spectator_embed_q8",
        "route": ROUTE,
        "max_partition_qubits": MAX_PARTITION_QUBITS,
    },
    {
        "anchor_qbits": 10,
        "workload": "mf1a_strict_spectator_embed_q10",
        "route": ROUTE,
        "max_partition_qubits": MAX_PARTITION_QUBITS,
    },
)

_NO_PRIOR = object()

STRICT_REGENERATION_ALLOWLIST = (
    "cases[0].provenance.implementation_revision",
    "cases[1].provenance.implementation_revision",
    "cases[2].provenance.implementation_revision",
    "cases[3].provenance.implementation_revision",
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
    allowlist = STRICT_REGENERATION_ALLOWLIST
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
        raise ValueError("M-F1a strict manifest must match the four frozen cells exactly")


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
    return [{"partition_index": record.partition_index} for record in result.partitions]


def mf1a_strict_spectator_operation_specs(qbit_num: int) -> list[dict[str, Any]]:
    if qbit_num < 2 or qbit_num % 2 != 0:
        raise ValueError(f"qbit_num must be an even integer >= 2, got {qbit_num}")
    specs: list[dict[str, Any]] = []
    for pair_index in range(qbit_num // 2):
        control = 2 * pair_index
        target = control + 1
        gate_index = 4 * pair_index + 2
        specs.extend(
            (
                {"kind": "gate", "name": "U3", "target_qbit": control, "param_count": 3},
                {"kind": "gate", "name": "U3", "target_qbit": target, "param_count": 3},
                {
                    "kind": "gate",
                    "name": "CNOT",
                    "target_qbit": target,
                    "control_qbit": control,
                    "param_count": 0,
                },
                {
                    "kind": "noise",
                    "name": "amplitude_damping",
                    "target_qbit": target,
                    "source_gate_index": gate_index,
                    "fixed_value": 0.05,
                    "param_count": 0,
                },
                {
                    "kind": "noise",
                    "name": "phase_damping",
                    "target_qbit": control,
                    "source_gate_index": gate_index,
                    "fixed_value": 0.07,
                    "param_count": 0,
                },
                {"kind": "gate", "name": "U3", "target_qbit": control, "param_count": 3},
            )
        )
    return specs


def build_mf1a_strict_spectator_descriptor_set(qbit_num: int):
    surface = build_canonical_planner_surface_from_operation_specs(
        qbit_num=qbit_num,
        source_type="structured_family_builder",
        workload_id=f"mf1a_strict_spectator_embed_q{qbit_num}",
        operation_specs=mf1a_strict_spectator_operation_specs(qbit_num),
    )
    return build_partition_descriptor_set(surface, max_partition_qubits=MAX_PARTITION_QUBITS)


def _channel_native_partition_count(
    fused_regions: list[dict[str, Any]], partition_count: int
) -> int:
    count = 0
    for partition_index in range(partition_count):
        regions = [
            region
            for region in fused_regions
            if region.get("partition_index") == partition_index
            and region.get("candidate_kind") == "channel_native_motif"
            and region.get("classification") == "actually_fused"
        ]
        if len(regions) == 1:
            count += 1
    return count


def _build_cell_descriptor(cell: dict[str, Any]):
    anchor_qbits = cell["anchor_qbits"]
    if anchor_qbits == 4:
        return workloads.build_phase31_microcase_descriptor_set(
            "phase31_local_support_q4_spectator_embedding_smoke",
            max_partition_qubits=MAX_PARTITION_QUBITS,
        )
    if anchor_qbits in (6, 8, 10):
        return build_mf1a_strict_spectator_descriptor_set(anchor_qbits)
    raise ValueError(f"Unsupported strict anchor {anchor_qbits}")


def _seed_policy_for_cell(cell: dict[str, Any]) -> str:
    del cell
    return "deterministic_workload_no_random_seed"


def _build_realization(result, descriptor_set) -> dict[str, Any]:
    partitions = _serialize_partitions(result)
    fused_regions = _serialize_fused_regions(result.fused_regions)
    partition_count = len(descriptor_set.partitions)
    channel_native_partition_count = _channel_native_partition_count(
        fused_regions, partition_count
    )
    return {
        "requested_path": result.requested_runtime_path,
        "realized_path": result.runtime_path,
        "partition_count": partition_count,
        "exact_output_present": result.exact_output_present,
        "channel_native_partition_count": channel_native_partition_count,
        "partitions": partitions,
        "fused_regions": fused_regions,
    }


def build_cases(*, provenance: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    manifest = build_manifest()
    validate_manifest_cells(manifest["cells"])
    run_provenance = capture_provenance() if provenance is None else provenance
    cases: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        descriptor_set = _build_cell_descriptor(cell)
        parameters = build_initial_parameters(descriptor_set.parameter_count)
        result = execute_partitioned_density_channel_native(descriptor_set, parameters)
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


def _motif_reason_is_valid(reason: Any) -> bool:
    if not isinstance(reason, str) or not reason.startswith("channel_native_motif_kraus_count_"):
        return False
    suffix = reason.removeprefix("channel_native_motif_kraus_count_")
    return suffix.isdigit() and int(suffix) >= 1


def _are_ints(values: list[Any]) -> bool:
    return all(isinstance(value, int) and not isinstance(value, bool) for value in values)


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

    for partition in partitions:
        if set(partition.keys()) != {"partition_index"}:
            return False

    indices = [partition.get("partition_index") for partition in partitions]
    if not _are_ints(indices) or sorted(indices) != list(range(partition_count)):
        return False

    for region in fused_regions:
        if region.get("candidate_kind") != "channel_native_motif":
            return False
        if region.get("classification") != "actually_fused":
            return False
        if not _motif_reason_is_valid(region.get("reason")):
            return False

    region_indices = [region.get("partition_index") for region in fused_regions]
    if not _are_ints(region_indices) or sorted(region_indices) != list(
        range(partition_count)
    ):
        return False

    if realization.get("channel_native_partition_count") != partition_count:
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
