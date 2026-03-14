#!/usr/bin/env python3
"""Validation: Task 5 Story 3 trace-and-anchor package.

Builds the phase-level Task 5 trace-and-anchor gate from:
- the already passing Task 5 Story 2 workflow baseline, which must contain the
  documented 10-qubit anchor evidence,
- and the canonical bounded 4-qubit optimization trace, kept as a stable raw
  artifact rather than renamed into a second trace identity.

The resulting bundle is intentionally a thin Task 5 layer:
- it keeps the workflow baseline and the bounded trace as separate evidence
  layers,
- it requires both the supported trace and the documented 10-qubit anchor,
- it preserves trace identity and supported-path attribution explicitly,
- and it makes missing trace-or-anchor evidence fail explicitly.

Run with:
    python benchmarks/density_matrix/task5_story3_trace_anchor_validation.py
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
    capture_case,
    run_optimization_trace,
    write_json,
)
from benchmarks.density_matrix.task5_story2_workflow_baseline_validation import (
    PRIMARY_BACKEND,
    REFERENCE_BACKEND,
    run_validation as run_story2_validation,
)

SUITE_NAME = "task5_story3_trace_anchor"
ARTIFACT_FILENAME = "story3_trace_anchor_bundle.json"
TRACE_CASE_NAME = "story2_trace_4q"
TRACE_ARTIFACT_FILENAME = "story2_trace_4q.json"
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
    "trace_artifact",
)


def build_requirement_metadata(
    story2_bundle,
    *,
    qubit_sizes=STORY4_WORKFLOW_QUBITS,
    parameter_set_count: int = STORY4_PARAMETER_SET_COUNT,
    required_trace_case_name: str = TRACE_CASE_NAME,
):
    return {
        "mandatory_workflow_qubits": list(qubit_sizes),
        "fixed_parameter_sets_per_size": parameter_set_count,
        "documented_anchor_qubit": 10,
        "required_trace_case_name": required_trace_case_name,
        "required_bundle_sources": [
            "task5_story2_exact_regime_workflow",
            "canonical_bounded_optimization_trace",
        ],
        "workflow_baseline_status": story2_bundle["status"],
    }


def validate_trace_artifact(trace_artifact):
    required_fields = (
        "case_name",
        "status",
        "workflow_completed",
        "bridge_supported_pass",
        "support_tier",
        "case_purpose",
        "counts_toward_mandatory_baseline",
        "required_story5_trace",
        "optimizer",
        "parameter_count",
        "total_trace_runtime_ms",
        "process_peak_rss_kb",
    )
    missing_fields = [field for field in required_fields if field not in trace_artifact]
    if missing_fields:
        raise ValueError(
            "Task 5 Story 3 trace artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def build_artifact_bundle(
    story2_bundle,
    trace_result,
    *,
    qubit_sizes=STORY4_WORKFLOW_QUBITS,
    parameter_set_count: int = STORY4_PARAMETER_SET_COUNT,
):
    trace_artifact = dict(trace_result)
    validate_trace_artifact(trace_artifact)

    documented_10q_anchor_case_names = sorted(
        case["case_name"]
        for case in story2_bundle["cases"]
        if case["qbit_num"] == 10
    )
    workflow_baseline_completed = bool(
        story2_bundle["summary"]["workflow_baseline_completed"]
    )
    documented_10q_anchor_present = bool(
        story2_bundle["summary"]["documented_10q_anchor_present"]
    )
    required_trace_present = bool(trace_artifact.get("required_story5_trace"))
    required_trace_completed = bool(
        trace_artifact["status"] == "completed"
        and trace_artifact.get("workflow_completed", False)
    )
    required_trace_bridge_supported = bool(
        trace_artifact.get("bridge_supported_pass", False)
    )
    trace_and_anchor_gate_completed = bool(
        workflow_baseline_completed
        and documented_10q_anchor_present
        and required_trace_present
        and required_trace_completed
        and required_trace_bridge_supported
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if trace_and_anchor_gate_completed else "fail",
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "requirements": build_requirement_metadata(
            story2_bundle,
            qubit_sizes=qubit_sizes,
            parameter_set_count=parameter_set_count,
            required_trace_case_name=trace_artifact["case_name"],
        ),
        "thresholds": dict(story2_bundle["thresholds"]),
        "software": dict(story2_bundle["software"]),
        "summary": {
            "workflow_baseline_completed": workflow_baseline_completed,
            "documented_10q_anchor_present": documented_10q_anchor_present,
            "documented_10q_anchor_case_names": documented_10q_anchor_case_names,
            "required_trace_case_name": trace_artifact["case_name"],
            "required_trace_present": required_trace_present,
            "required_trace_completed": required_trace_completed,
            "required_trace_bridge_supported": required_trace_bridge_supported,
            "required_trace_counts_toward_mandatory_baseline": trace_artifact.get(
                "counts_toward_mandatory_baseline"
            ),
            "trace_and_anchor_gate_completed": trace_and_anchor_gate_completed,
        },
        "required_artifacts": {
            "story2_workflow_baseline": {
                "suite_name": story2_bundle["suite_name"],
                "status": story2_bundle["status"],
                "summary": story2_bundle["summary"],
            }
        },
        "trace_artifact": trace_artifact,
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 5 Story 3 artifact bundle is missing required fields: {}".format(
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
    _, story2_bundle = run_story2_validation(
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
        verbose=verbose,
    )
    trace_result = capture_case(TRACE_CASE_NAME, run_optimization_trace)
    bundle = build_artifact_bundle(
        story2_bundle,
        trace_result,
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
    )
    if verbose:
        print(
            "{} [{}] 10q_anchor={} trace={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["documented_10q_anchor_present"],
                bundle["summary"]["required_trace_completed"],
            )
        )
    return story2_bundle, trace_result, bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 5 Story 3 JSON artifacts.",
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
    _, trace_result, bundle = run_validation(
        qubit_sizes=tuple(args.qubit_sizes),
        parameter_set_count=args.parameter_set_count,
        verbose=not args.quiet,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_bundle(args.output_dir / ARTIFACT_FILENAME, bundle)
    write_json(args.output_dir / TRACE_ARTIFACT_FILENAME, trace_result)
    print(
        "Wrote {} and {} with status {}".format(
            args.output_dir / ARTIFACT_FILENAME,
            args.output_dir / TRACE_ARTIFACT_FILENAME,
            bundle["status"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
