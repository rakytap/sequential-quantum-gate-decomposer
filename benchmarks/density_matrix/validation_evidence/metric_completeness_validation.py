#!/usr/bin/env python3
"""Validation: metric-completeness gate.

Builds the phase-level metric-completeness gate from the delivered validation
evidence layers:
- local correctness bundle,
- workflow baseline bundle,
- trace-and-anchor bundle plus raw trace artifact.

The resulting bundle is intentionally a thin validation-evidence layer:
- it checks required internal-consistency fields on mandatory microcases,
- it checks required execution and performance fields on mandatory workflow
  cases,
- it checks the delivered trace artifact for required runtime and stability
  metadata,
- and it fails explicitly when required metrics are missing.

Run with:
    python benchmarks/density_matrix/validation_evidence/metric_completeness_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import (
    EXACT_REGIME_PARAMETER_SET_COUNT,
    EXACT_REGIME_WORKFLOW_QUBITS,
)
from benchmarks.density_matrix.validation_evidence.local_correctness_validation import (
    run_validation as run_local_correctness_validation,
)
from benchmarks.density_matrix.validation_evidence.trace_anchor_validation import (
    PRIMARY_BACKEND,
    REFERENCE_BACKEND,
    run_validation as run_trace_anchor_validation,
)

SUITE_NAME = "metric_completeness_validation"
ARTIFACT_FILENAME = "metric_completeness_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "validation_evidence"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "backend",
    "reference_backend",
    "requirements",
    "thresholds",
    "software",
    "summary",
    "required_artifacts",
)
REQUIRED_MICRO_FIELDS = (
    "status",
    "energy_pass",
    "density_valid_pass",
    "trace_pass",
    "observable_pass",
)
REQUIRED_WORKFLOW_FIELDS = (
    "status",
    "workflow_completed",
    "energy_pass",
    "density_valid_pass",
    "trace_pass",
    "observable_pass",
    "bridge_supported_pass",
    "total_case_runtime_ms",
    "process_peak_rss_kb",
)
REQUIRED_TRACE_FIELDS = (
    "status",
    "workflow_completed",
    "bridge_supported_pass",
    "total_trace_runtime_ms",
    "process_peak_rss_kb",
    "required_validation_trace",
)


def build_requirement_metadata():
    return {
        "micro_required_fields": list(REQUIRED_MICRO_FIELDS),
        "workflow_required_fields": list(REQUIRED_WORKFLOW_FIELDS),
        "trace_required_fields": list(REQUIRED_TRACE_FIELDS),
        "required_bundle_sources": [
            "local_correctness_validation",
            "workflow_baseline_validation",
            "trace_anchor_validation",
        ],
    }


def _missing_fields(payload, required_fields):
    return [field for field in required_fields if field not in payload]


def _build_missing_field_map(cases, required_fields):
    missing_by_case = {}
    for case in cases:
        missing = _missing_fields(case, required_fields)
        if missing:
            missing_by_case[case["case_name"]] = missing
    return missing_by_case


def build_artifact_bundle(
    local_correctness_bundle,
    workflow_baseline_bundle,
    trace_anchor_bundle,
):
    micro_cases = [dict(case) for case in local_correctness_bundle["cases"]]
    workflow_cases = [dict(case) for case in workflow_baseline_bundle["cases"]]
    trace_artifact = dict(trace_anchor_bundle["trace_artifact"])

    missing_micro_metric_fields_by_case = _build_missing_field_map(
        micro_cases, REQUIRED_MICRO_FIELDS
    )
    missing_workflow_metric_fields_by_case = _build_missing_field_map(
        workflow_cases, REQUIRED_WORKFLOW_FIELDS
    )
    missing_trace_metric_fields = _missing_fields(trace_artifact, REQUIRED_TRACE_FIELDS)

    workflow_cases_with_stable_execution = sum(
        bool(
            case.get("workflow_completed", False)
            and case.get("bridge_supported_pass", False)
            and case.get("status") == "pass"
        )
        for case in workflow_cases
    )
    trace_execution_stability_pass = bool(
        trace_artifact.get("workflow_completed", False)
        and trace_artifact.get("bridge_supported_pass", False)
        and trace_artifact.get("status") == "completed"
    )
    metric_completeness_gate_completed = bool(
        local_correctness_bundle["status"] == "pass"
        and workflow_baseline_bundle["status"] == "pass"
        and trace_anchor_bundle["status"] == "pass"
        and not missing_micro_metric_fields_by_case
        and not missing_workflow_metric_fields_by_case
        and not missing_trace_metric_fields
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if metric_completeness_gate_completed else "fail",
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "requirements": build_requirement_metadata(),
        "thresholds": {
            "micro": dict(local_correctness_bundle["thresholds"]),
            "workflow": dict(workflow_baseline_bundle["thresholds"]),
        },
        "software": dict(workflow_baseline_bundle["software"]),
        "summary": {
            "micro_cases_checked": len(micro_cases),
            "workflow_cases_checked": len(workflow_cases),
            "trace_artifacts_checked": 1,
            "micro_cases_missing_required_metrics": len(
                missing_micro_metric_fields_by_case
            ),
            "workflow_cases_missing_required_metrics": len(
                missing_workflow_metric_fields_by_case
            ),
            "trace_artifacts_missing_required_metrics": len(
                missing_trace_metric_fields
            ),
            "missing_micro_metric_fields_by_case": missing_micro_metric_fields_by_case,
            "missing_workflow_metric_fields_by_case": (
                missing_workflow_metric_fields_by_case
            ),
            "missing_trace_metric_fields": missing_trace_metric_fields,
            "workflow_cases_with_stable_execution": workflow_cases_with_stable_execution,
            "trace_execution_stability_pass": trace_execution_stability_pass,
            "metric_completeness_gate_completed": metric_completeness_gate_completed,
        },
        "required_artifacts": {
            "local_correctness_reference": {
                "suite_name": local_correctness_bundle["suite_name"],
                "status": local_correctness_bundle["status"],
                "summary": local_correctness_bundle["summary"],
            },
            "workflow_baseline_reference": {
                "suite_name": workflow_baseline_bundle["suite_name"],
                "status": workflow_baseline_bundle["status"],
                "summary": workflow_baseline_bundle["summary"],
            },
            "trace_anchor_reference": {
                "suite_name": trace_anchor_bundle["suite_name"],
                "status": trace_anchor_bundle["status"],
                "summary": trace_anchor_bundle["summary"],
            },
        },
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Metric-completeness bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def write_artifact_bundle(output_path: Path, bundle):
    validate_artifact_bundle(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_validation(
    *,
    qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS,
    parameter_set_count: int = EXACT_REGIME_PARAMETER_SET_COUNT,
    verbose=False,
):
    _, _, local_correctness_bundle = run_local_correctness_validation(verbose=verbose)
    workflow_baseline_bundle, _, trace_anchor_bundle = run_trace_anchor_validation(
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
        verbose=verbose,
    )
    bundle = build_artifact_bundle(
        local_correctness_bundle,
        workflow_baseline_bundle,
        trace_anchor_bundle,
    )
    if verbose:
        print(
            "{} [{}] micro_missing={} workflow_missing={} trace_missing={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["micro_cases_missing_required_metrics"],
                bundle["summary"]["workflow_cases_missing_required_metrics"],
                bundle["summary"]["trace_artifacts_missing_required_metrics"],
            )
        )
    return (
        local_correctness_bundle,
        workflow_baseline_bundle,
        trace_anchor_bundle,
        bundle,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the metric-completeness JSON artifact bundle.",
    )
    parser.add_argument(
        "--parameter-set-count",
        type=int,
        default=EXACT_REGIME_PARAMETER_SET_COUNT,
        help="Number of fixed parameter vectors per qubit size.",
    )
    parser.add_argument(
        "--qubit-sizes",
        type=int,
        nargs="*",
        default=list(EXACT_REGIME_WORKFLOW_QUBITS),
        help="Workflow qubit sizes to include.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, _, bundle = run_validation(
        qubit_sizes=tuple(args.qubit_sizes),
        parameter_set_count=args.parameter_set_count,
        verbose=not args.quiet,
    )
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, bundle)
    print(
        "Wrote {} with status {}".format(
            output_path,
            bundle["status"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
