#!/usr/bin/env python3
"""M-F1a provisional q4 baseline exactness tracer."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy

from benchmarks.density_matrix.correctness_evidence.common import DEFAULT_OUTPUT_ROOT
from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters
from benchmarks.density_matrix.planner_surface.common import build_phase2_continuity_vqe
from squander.density_matrix import DensityMatrix
from squander.density_matrix import _density_matrix_cpp
from squander.partitioning.noisy_planner import (
    build_phase3_continuity_partition_descriptor_set,
)
from squander.partitioning.noisy_runtime import (
    PHASE3_RUNTIME_PATH_BASELINE,
    execute_partitioned_density,
    execute_sequential_density_reference,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SUITE_NAME = "correctness_evidence_mf1a_q4_baseline_bundle_v1"
MANIFEST_SCHEMA_VERSION = "correctness_evidence_mf1a_q4_baseline_manifest_v1"
RECORD_SCHEMA_VERSION = "correctness_evidence_mf1a_q4_baseline_case_v1"
BUNDLE_SCHEMA_VERSION = "correctness_evidence_mf1a_q4_baseline_bundle_v1"
ARTIFACT_FILENAME = "mf1a_q4_baseline_bundle.json"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "mf1a" / "q4_baseline"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / ARTIFACT_FILENAME
REGENERATION_COMMAND = (
    "conda run -n qgd --no-capture-output python "
    "benchmarks/density_matrix/correctness_evidence/validation_pipeline.py"
)
WORKLOAD = "phase2_xxz_hea_q4_continuity"
ROUTE = "partitioned_density_descriptor_baseline"
ANCHOR_QBITS = 4
MAX_PARTITION_QUBITS = 2
MF1A_QA001_MATRIX_TOL = 1e-10
MF1A_QA001_LAMBDA_MIN_FLOOR = -1e-12

_NO_PRIOR = object()
_QA001_VALUE_KEYS = (
    "frobenius_norm_diff",
    "max_abs_diff",
    "trace_abs_deviation",
    "lambda_min",
)
_QA001_REGENERATION_TOLERANCES = {
    "frobenius_norm_diff": 1e-10,
    "max_abs_diff": 1e-10,
    "trace_abs_deviation": 1e-10,
    "lambda_min": 1e-12,
}


def _density_array(value: DensityMatrix | np.ndarray) -> np.ndarray:
    if isinstance(value, DensityMatrix):
        return np.asarray(value.to_numpy(), dtype=np.complex128)
    return np.asarray(value, dtype=np.complex128)


def _default_eigenvalues(candidate: DensityMatrix | np.ndarray) -> np.ndarray:
    density = (
        candidate
        if isinstance(candidate, DensityMatrix)
        else DensityMatrix.from_numpy(np.asarray(candidate, dtype=np.complex128))
    )
    return np.asarray(density.eigenvalues(), dtype=np.float64)


def evaluate_mf1a_qa001(
    candidate: DensityMatrix | np.ndarray,
    reference: DensityMatrix | np.ndarray,
    *,
    eigenvalues_fn: Callable[[DensityMatrix | np.ndarray], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate only the frozen QA-001 predicate, in its frozen failure order."""
    candidate_array = _density_array(candidate)
    reference_array = _density_array(reference)
    finite_entries = bool(
        np.isfinite(candidate_array.real).all()
        and np.isfinite(candidate_array.imag).all()
    )

    delta = candidate_array - reference_array
    frobenius = float(np.linalg.norm(delta))
    max_abs = float(np.max(np.abs(delta)))
    trace_deviation = float(abs(np.trace(candidate_array) - 1.0))
    finite_residuals = bool(
        np.isfinite([frobenius, max_abs, trace_deviation]).all()
    )

    result: dict[str, Any] = {
        "finite_entries_pass": finite_entries,
        "finite_residuals_pass": finite_residuals,
        "frobenius_norm_diff": frobenius,
        "frobenius_norm_diff_pass": bool(
            finite_residuals and frobenius <= MF1A_QA001_MATRIX_TOL
        ),
        "max_abs_diff": max_abs,
        "max_abs_diff_pass": bool(
            finite_residuals and max_abs <= MF1A_QA001_MATRIX_TOL
        ),
        "trace_abs_deviation": trace_deviation,
        "trace_abs_deviation_pass": bool(
            finite_residuals and trace_deviation <= MF1A_QA001_MATRIX_TOL
        ),
        "eigensolver_pass": False,
        "finite_eigenvalues_pass": False,
        "lambda_min": None,
        "lambda_min_pass": False,
        "qa001_pass": False,
        "first_failure": None,
    }

    if finite_entries and finite_residuals:
        try:
            eigenvalues = np.asarray(
                (eigenvalues_fn or _default_eigenvalues)(candidate),
                dtype=np.float64,
            )
            result["eigensolver_pass"] = True
            result["finite_eigenvalues_pass"] = bool(
                eigenvalues.size > 0 and np.isfinite(eigenvalues).all()
            )
            if result["finite_eigenvalues_pass"]:
                result["lambda_min"] = float(np.min(eigenvalues))
                result["lambda_min_pass"] = bool(
                    result["lambda_min"] >= MF1A_QA001_LAMBDA_MIN_FLOOR
                )
        except Exception:
            result["eigensolver_pass"] = False

    checks = (
        ("finite_entries", result["finite_entries_pass"]),
        ("finite_residuals", result["finite_residuals_pass"]),
        ("frobenius_norm_diff", result["frobenius_norm_diff_pass"]),
        ("max_abs_diff", result["max_abs_diff_pass"]),
        ("trace_abs_deviation", result["trace_abs_deviation_pass"]),
        ("eigensolver", result["eigensolver_pass"]),
        ("finite_eigenvalues", result["finite_eigenvalues_pass"]),
        ("lambda_min", result["lambda_min_pass"]),
    )
    result["first_failure"] = next(
        (name for name, passed in checks if not passed), None
    )
    result["qa001_pass"] = result["first_failure"] is None
    return result


def build_manifest() -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "cells": [
            {
                "anchor_qbits": ANCHOR_QBITS,
                "workload": WORKLOAD,
                "route": ROUTE,
                "max_partition_qubits": MAX_PARTITION_QUBITS,
            }
        ],
    }


def validate_manifest_cells(cells: list[dict[str, Any]]) -> None:
    if cells != build_manifest()["cells"]:
        raise ValueError("M-F1a tracer manifest must contain exactly one reviewed cell")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, str]:
    try:
        display_path = path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        display_path = str(path.resolve())
    return {"path": display_path, "sha256": _sha256(path)}


def _git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def capture_provenance() -> dict[str, Any]:
    """Capture the checkout and runtime identity before this bundle is written."""
    revision = _git_output("rev-parse", "HEAD")
    status_lines = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    dirty_paths = sorted(line[3:] for line in status_lines if len(line) > 3)
    extension_path = Path(_density_matrix_cpp.__file__).resolve()
    environment = {
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    dependencies = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "squander": metadata.version("squander"),
    }
    clean_start = not dirty_paths
    complete = bool(
        revision
        and environment["conda_default_env"] == "qgd"
        and environment["conda_prefix"]
        and environment["python_executable"]
        and extension_path.is_file()
    )
    return {
        "implementation_revision": revision,
        "clean_start": clean_start,
        "dirty_paths": dirty_paths,
        "command": REGENERATION_COMMAND,
        "environment": environment,
        "dependencies": dependencies,
        "extension_identities": [_identity(extension_path)],
        "input_artifact_identities": [],
        "provenance_pass": bool(clean_start and complete),
    }


def build_cases(*, provenance: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    manifest = build_manifest()
    validate_manifest_cells(manifest["cells"])
    run_provenance = capture_provenance() if provenance is None else provenance

    vqe, _, _ = build_phase2_continuity_vqe(ANCHOR_QBITS)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(
        vqe, max_partition_qubits=MAX_PARTITION_QUBITS
    )
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    result = execute_partitioned_density(
        descriptor_set, parameters, allow_fusion=False
    )
    reference = execute_sequential_density_reference(descriptor_set, parameters)
    qa001 = evaluate_mf1a_qa001(result.density_matrix, reference)

    return [
        {
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "route": ROUTE,
            "anchor_qbits": ANCHOR_QBITS,
            "workload": WORKLOAD,
            "planner_setting": {
                "max_partition_qubits": descriptor_set.max_partition_qubits
            },
            "parameters": parameters.tolist(),
            "seed_policy": "deterministic_workload_no_random_seed",
            "realization": {
                "requested_path": result.requested_runtime_path,
                "realized_path": result.runtime_path,
                "partition_count": result.partition_count,
                "exact_output_present": result.exact_output_present,
                "actual_fused_execution": result.actual_fused_execution,
                "fused_region_count": result.fused_region_count,
                "fused_region_classifications": [
                    region.classification for region in result.fused_regions
                ],
            },
            "qa001": qa001,
            "milestone_counted": False,
            "completeness_claim": False,
            "claim_boundary": (
                "This is the q4 baseline tracer only; it makes no complete M-F1a, "
                "external-protocol, Aer, energy, or frozen-matrix claim."
            ),
            "provenance": run_provenance,
        }
    ]


def _route_realization_pass(case: dict[str, Any]) -> bool:
    realization = case["realization"]
    return bool(
        case["route"] == ROUTE
        and case["anchor_qbits"] == ANCHOR_QBITS
        and case["workload"] == WORKLOAD
        and case["planner_setting"]["max_partition_qubits"]
        == MAX_PARTITION_QUBITS
        and realization["requested_path"] == PHASE3_RUNTIME_PATH_BASELINE
        and realization["realized_path"] == PHASE3_RUNTIME_PATH_BASELINE
        and realization["partition_count"] > 1
        and realization["exact_output_present"] is True
        and realization["actual_fused_execution"] is False
        and realization["fused_region_count"] == 0
        and "actually_fused"
        not in realization["fused_region_classifications"]
    )


def _load_prior_bundle() -> dict[str, Any] | None:
    if not DEFAULT_OUTPUT_PATH.exists():
        return None
    try:
        return json.loads(DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": BUNDLE_SCHEMA_VERSION, "invalid_prior_bundle": True}


def _regeneration_result(
    current_case: dict[str, Any], prior_bundle: dict[str, Any] | None
) -> dict[str, Any]:
    if prior_bundle is None:
        return {"prior_present": False, "pass": True, "first_mismatch": None}
    prior_cases = prior_bundle.get("cases")
    if (
        prior_bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or not isinstance(prior_cases, list)
        or len(prior_cases) != 1
    ):
        return {
            "prior_present": True,
            "pass": False,
            "first_mismatch": "bundle_structure",
        }
    prior_case = prior_cases[0]
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
                "first_mismatch": f"cases[0].{key}",
            }
    for key in ("extension_identities", "input_artifact_identities"):
        if current_case["provenance"].get(key) != prior_case.get("provenance", {}).get(
            key
        ):
            return {
                "prior_present": True,
                "pass": False,
                "first_mismatch": f"cases[0].provenance.{key}",
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
        if current_case["provenance"].get(key) != prior_case.get("provenance", {}).get(
            key
        ):
            return {
                "prior_present": True,
                "pass": False,
                "first_mismatch": f"cases[0].provenance.{key}",
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
            "first_mismatch": "cases[0].qa001.categorical",
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
                "first_mismatch": f"cases[0].qa001.{key}",
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
        exact_set_pass = True
    except (KeyError, ValueError):
        exact_set_pass = False

    route_pass = bool(
        exact_set_pass and len(cases) == 1 and _route_realization_pass(cases[0])
    )
    qa001_pass = bool(
        exact_set_pass and len(cases) == 1 and cases[0]["qa001"]["qa001_pass"]
    )
    provenance_pass = bool(
        exact_set_pass
        and len(cases) == 1
        and cases[0]["provenance"]["provenance_pass"]
    )
    selected_prior = _load_prior_bundle() if prior_bundle is _NO_PRIOR else prior_bundle
    regeneration = (
        _regeneration_result(cases[0], selected_prior)
        if exact_set_pass and len(cases) == 1
        else {
            "prior_present": selected_prior is not None,
            "pass": False,
            "first_mismatch": "manifest_exact_set",
        }
    )
    gates = (
        ("manifest_exact_set", exact_set_pass),
        ("route_realization", route_pass),
        ("qa001", qa001_pass),
        ("provenance", provenance_pass),
        ("regeneration", regeneration["pass"]),
    )
    first_failure = next((name for name, passed in gates if not passed), None)
    status = "pass" if first_failure is None else "fail"
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
        "summary": {
            "total_cases": len(cases),
            "qa001_passes": sum(
                bool(case.get("qa001", {}).get("qa001_pass")) for case in cases
            ),
            "milestone_counted_cases": 0,
            "completeness_claim": False,
            "first_failure": first_failure,
        },
        "regeneration": regeneration,
        "non_counted_context": non_counted_context or {},
        "cases": cases,
    }
