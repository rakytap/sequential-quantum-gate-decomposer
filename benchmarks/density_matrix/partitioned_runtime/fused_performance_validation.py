#!/usr/bin/env python3
"""Fused performance validation: timing, memory, and threshold-or-diagnosis closure.

Benchmarks representative required structured cases with real fused coverage and
closes the threshold-or-diagnosis rule for publication-style evidence.

Run with:
    python benchmarks/density_matrix/partitioned_runtime/fused_performance_validation.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.partitioned_runtime.common import (
    PHASE3_RUNTIME_DENSITY_TOL,
    execute_fused_with_reference,
)
from benchmarks.density_matrix.partitioned_runtime.fusion_case_selection import (
    iter_fusion_structured_cases,
)
from benchmarks.density_matrix.planner_surface.common import build_software_metadata
from squander.partitioning.noisy_runtime import (
    execute_partitioned_density,
    execute_partitioned_density_fused,
)

SUITE_NAME = "phase3_partitioned_runtime_fused_performance"
ARTIFACT_FILENAME = "fused_performance_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "partitioned_runtime"
    / "fused_performance"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "runtime_schema_version",
    "software",
    "summary",
    "cases",
)
REPETITIONS = 3


def _benchmark_case(metadata: dict, descriptor_set, parameters) -> dict:
    fused_result, _, density_metrics = execute_fused_with_reference(
        descriptor_set, parameters
    )
    if not fused_result.actual_fused_execution:
        return {
            "case_name": metadata["workload_id"],
            "case_kind": metadata["case_kind"],
            "qbit_num": metadata["qbit_num"],
            "actual_fused_execution": False,
            "threshold_or_diagnosis_pass": False,
            "reason": "no_real_fused_coverage",
            "metadata": dict(metadata),
        }

    baseline_runtime_ms: list[float] = []
    fused_runtime_ms: list[float] = []
    baseline_peak_rss_kb: list[int] = []
    fused_peak_rss_kb: list[int] = []
    for _ in range(REPETITIONS):
        baseline_result = execute_partitioned_density(descriptor_set, parameters)
        fused_repeat_result = execute_partitioned_density_fused(descriptor_set, parameters)
        baseline_runtime_ms.append(baseline_result.runtime_ms)
        fused_runtime_ms.append(fused_repeat_result.runtime_ms)
        baseline_peak_rss_kb.append(baseline_result.peak_rss_kb)
        fused_peak_rss_kb.append(fused_repeat_result.peak_rss_kb)

    baseline_median_runtime_ms = float(statistics.median(baseline_runtime_ms))
    fused_median_runtime_ms = float(statistics.median(fused_runtime_ms))
    baseline_median_peak_rss_kb = int(statistics.median(baseline_peak_rss_kb))
    fused_median_peak_rss_kb = int(statistics.median(fused_peak_rss_kb))
    speedup = (
        baseline_median_runtime_ms / fused_median_runtime_ms
        if fused_median_runtime_ms > 0.0
        else 0.0
    )
    memory_reduction = (
        (baseline_median_peak_rss_kb - fused_median_peak_rss_kb)
        / baseline_median_peak_rss_kb
        if baseline_median_peak_rss_kb > 0
        else 0.0
    )
    correctness_pass = (
        density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        and density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        and fused_result.rho_is_valid
    )
    positive_threshold_pass = correctness_pass and (
        speedup >= 1.2 or memory_reduction >= 0.15
    )
    diagnosis_reasons: list[str] = []
    if not positive_threshold_pass:
        if fused_result.deferred_region_count > 0:
            diagnosis_reasons.append("limited_fused_coverage_due_to_noise_boundaries")
        if fused_result.supported_unfused_region_count > 0:
            diagnosis_reasons.append("supported_islands_left_unfused")
        if fused_median_runtime_ms >= baseline_median_runtime_ms:
            diagnosis_reasons.append("python_fused_kernel_overhead_or_short_unitary_islands")
        if fused_median_peak_rss_kb >= baseline_median_peak_rss_kb:
            diagnosis_reasons.append("no_peak_memory_reduction_on_representative_case")
        if not diagnosis_reasons:
            diagnosis_reasons.append("no_representative_threshold_gain_observed")
    return {
        "case_name": metadata["workload_id"],
        "case_kind": metadata["case_kind"],
        "family_name": metadata["family_name"],
        "qbit_num": metadata["qbit_num"],
        "noise_pattern": metadata["noise_pattern"],
        "runtime_schema_version": fused_result.runtime_schema_version,
        "runtime_path": fused_result.runtime_path,
        "actual_fused_execution": fused_result.actual_fused_execution,
        "fused_region_count": fused_result.fused_region_count,
        "supported_unfused_region_count": fused_result.supported_unfused_region_count,
        "deferred_region_count": fused_result.deferred_region_count,
        "fused_gate_count": fused_result.fused_gate_count,
        "frobenius_norm_diff": density_metrics["frobenius_norm_diff"],
        "max_abs_diff": density_metrics["max_abs_diff"],
        "correctness_pass": correctness_pass,
        "baseline_median_runtime_ms": baseline_median_runtime_ms,
        "fused_median_runtime_ms": fused_median_runtime_ms,
        "baseline_median_peak_rss_kb": baseline_median_peak_rss_kb,
        "fused_median_peak_rss_kb": fused_median_peak_rss_kb,
        "speedup": speedup,
        "memory_reduction": memory_reduction,
        "positive_threshold_pass": positive_threshold_pass,
        "diagnosis_reasons": diagnosis_reasons,
        "threshold_or_diagnosis_pass": correctness_pass
        and (positive_threshold_pass or len(diagnosis_reasons) > 0),
        "metadata": dict(metadata),
    }


def build_cases() -> list[dict]:
    selected_by_qubits: dict[int, dict] = {}
    for metadata, descriptor_set, parameters in iter_fusion_structured_cases():
        if metadata["qbit_num"] in selected_by_qubits:
            continue
        case = _benchmark_case(metadata, descriptor_set, parameters)
        if case["actual_fused_execution"]:
            selected_by_qubits[metadata["qbit_num"]] = case
        if set(selected_by_qubits) == {8, 10}:
            break
    missing = {8, 10} - set(selected_by_qubits)
    if missing:
        raise RuntimeError(
            "Missing representative fused benchmark cases for qubits: {}".format(
                sorted(missing)
            )
        )
    return [selected_by_qubits[8], selected_by_qubits[10]]


def build_artifact_bundle(cases: list[dict]) -> dict:
    threshold_pass_cases = sum(case["positive_threshold_pass"] for case in cases)
    threshold_or_diagnosis_passes = sum(
        case["threshold_or_diagnosis_pass"] for case in cases
    )
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if threshold_or_diagnosis_passes == len(cases)
        and len(cases) == 2
        and (
            threshold_pass_cases >= 1
            or all(case["diagnosis_reasons"] for case in cases if not case["positive_threshold_pass"])
        )
        else "fail",
        "runtime_schema_version": cases[0]["runtime_schema_version"],
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "representative_qubits": [8, 10],
            "positive_threshold_pass_cases": threshold_pass_cases,
            "threshold_or_diagnosis_passes": threshold_or_diagnosis_passes,
            "diagnosis_only_cases": sum(
                (not case["positive_threshold_pass"]) and case["threshold_or_diagnosis_pass"]
                for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Fused performance bundle missing required fields: {}".format(
                ", ".join(missing)
            )
        )
    return bundle


def write_artifact_bundle(bundle: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ARTIFACT_FILENAME
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the fused performance bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case console output.",
    )
    args = parser.parse_args(argv)

    cases = build_cases()
    bundle = build_artifact_bundle(cases)
    output_path = write_artifact_bundle(bundle, output_dir=args.output_dir)

    if not args.quiet:
        for case in cases:
            print(
                "{case_name}: speedup={speedup:.3f}, memory_reduction={memory_reduction:.3f}, positive={positive_threshold_pass}, diagnosis={diagnosis_reasons}".format(
                    **case
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
