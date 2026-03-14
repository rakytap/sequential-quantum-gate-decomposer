#!/usr/bin/env python3
"""Validation: Task 5 Story 4 metric completeness gate.

Builds the phase-level metric-completeness gate from the already delivered Task
5 evidence layers:
- Task 5 Story 1 local correctness bundle,
- Task 5 Story 2 workflow baseline bundle,
- Task 5 Story 3 trace-and-anchor bundle plus raw trace artifact.

The resulting bundle is intentionally a thin Task 5 layer:
- it checks required internal-consistency fields on mandatory microcases,
- it checks required execution and performance fields on mandatory workflow
  cases,
- it checks the delivered trace artifact for required runtime and stability
  metadata,
- and it fails explicitly when required metrics are missing.

Run with:
    python benchmarks/density_matrix/task5_story4_metric_completeness_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.story2_vqe_density_validation import (
    STORY4_PARAMETER_SET_COUNT,
    STORY4_WORKFLOW_QUBITS,
)
from benchmarks.density_matrix.task5_story1_local_correctness_validation import (
    run_validation as run_story1_validation,
)
from benchmarks.density_matrix.task5_story3_trace_anchor_validation import (
    PRIMARY_BACKEND,
    REFERENCE_BACKEND,
    run_validation as run_story3_validation,
)

SUITE_NAME = "task5_story4_metric_completeness"
ARTIFACT_FILENAME = "story4_metric_completeness_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "phase2_task5"
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
    "required_story5_trace",
)


def build_requirement_metadata():
    return {
        "micro_required_fields": list(REQUIRED_MICRO_FIELDS),
        "workflow_required_fields": list(REQUIRED_WORKFLOW_FIELDS),
        "trace_required_fields": list(REQUIRED_TRACE_FIELDS),
        "required_bundle_sources": [
            "task5_story1_local_correctness",
            "task5_story2_exact_regime_workflow",
            "task5_story3_trace_anchor",
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


def build_artifact_bundle(story1_bundle, story2_bundle, story3_bundle):
    micro_cases = [dict(case) for case in story1_bundle["cases"]]
    workflow_cases = [dict(case) for case in story2_bundle["cases"]]
    trace_artifact = dict(story3_bundle["trace_artifact"])

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
        story1_bundle["status"] == "pass"
        and story2_bundle["status"] == "pass"
        and story3_bundle["status"] == "pass"
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
            "micro": dict(story1_bundle["thresholds"]),
            "workflow": dict(story2_bundle["thresholds"]),
        },
        "software": dict(story2_bundle["software"]),
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
            "story1_local_correctness": {
                "suite_name": story1_bundle["suite_name"],
                "status": story1_bundle["status"],
                "summary": story1_bundle["summary"],
            },
            "story2_workflow_baseline": {
                "suite_name": story2_bundle["suite_name"],
                "status": story2_bundle["status"],
                "summary": story2_bundle["summary"],
            },
            "story3_trace_anchor": {
                "suite_name": story3_bundle["suite_name"],
                "status": story3_bundle["status"],
                "summary": story3_bundle["summary"],
            },
        },
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 5 Story 4 artifact bundle is missing required fields: {}".format(
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
    qubit_sizes=STORY4_WORKFLOW_QUBITS,
    parameter_set_count: int = STORY4_PARAMETER_SET_COUNT,
    verbose=False,
):
    _, _, story1_bundle = run_story1_validation(verbose=verbose)
    story2_bundle, _, story3_bundle = run_story3_validation(
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
        verbose=verbose,
    )
    bundle = build_artifact_bundle(story1_bundle, story2_bundle, story3_bundle)
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
    return story1_bundle, story2_bundle, story3_bundle, bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 5 Story 4 JSON artifact bundle.",
    )
    parser.add_argument(
        "--parameter-set-count",
        type=int,
        default=STORY4_PARAMETER_SET_COUNT,
        help="Number of fixed parameter vectors per qubit size.",
    )
    parser.add_argument(
        "--qubit-sizes",
        type=int,
        nargs="*",
        default=list(STORY4_WORKFLOW_QUBITS),
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
