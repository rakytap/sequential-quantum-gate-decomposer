#!/usr/bin/env python3
"""Validation: required local-noise workflow bundle.

Reuses the canonical workflow-scale exactness harness and emits the required
local-noise artifact pair:
- one required-local-noise workflow bundle across the mandatory exact regime,
- one bounded required-baseline optimization trace.

Run with:
    python benchmarks/density_matrix/noise_support/required_local_noise_workflow_validation.py
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
    PRIMARY_BACKEND,
    REFERENCE_BACKEND,
    EXACT_REGIME_PARAMETER_SET_COUNT,
    EXACT_REGIME_WORKFLOW_QUBITS,
    build_exact_regime_workflow_bundle,
    capture_case,
    run_optimization_trace,
    run_exact_regime_workflow_matrix,
    write_json,
)

SUITE_NAME = "required_local_noise_workflow"
WORKFLOW_BUNDLE_FILENAME = "required_local_noise_workflow_bundle.json"
TRACE_ARTIFACT_FILENAME = "required_local_noise_trace_4q.json"
TRACE_CASE_NAME = "required_local_noise_trace_4q"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "noise_support"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "backend",
    "reference_backend",
    "thresholds",
    "software",
    "summary",
    "cases",
)


def build_artifact_bundle(
    workflow_results,
    trace_result,
    *,
    qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS,
    parameter_set_count: int = EXACT_REGIME_PARAMETER_SET_COUNT,
):
    bundle = build_exact_regime_workflow_bundle(
        workflow_results,
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
        trace_result=trace_result,
    )
    bundle = dict(bundle)
    bundle["suite_name"] = SUITE_NAME
    bundle["summary"] = dict(bundle["summary"])
    bundle["summary"].update(
        {
            "required_trace_case_name": trace_result["case_name"],
            "required_trace_completed": trace_result["status"] == "completed",
            "required_trace_bridge_supported": trace_result.get(
                "bridge_supported_pass", False
            ),
            "required_trace_present": bool(trace_result.get("required_validation_trace")),
            "required_trace_counts_toward_mandatory_baseline": trace_result.get(
                "counts_toward_mandatory_baseline"
            ),
        }
    )
    validate_artifact_bundle(bundle, trace_result=trace_result)
    return bundle


def validate_artifact_bundle(bundle, *, trace_result):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Required local-noise workflow bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    if bundle["summary"]["required_cases"] != bundle["summary"]["total_cases"]:
        raise ValueError(
            "Required local-noise workflow bundle mixes non-required cases into the mandatory matrix"
        )

    if not trace_result.get("required_validation_trace", False):
        raise ValueError("Required local-noise trace is missing required_validation_trace")

    if trace_result.get("support_tier") != "required":
        raise ValueError("Required local-noise trace is not marked as required support")

    if trace_result.get("status") != "completed":
        raise ValueError("Required local-noise trace did not complete successfully")


def write_artifact_bundle(output_path: Path, bundle, *, trace_result):
    validate_artifact_bundle(bundle, trace_result=trace_result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_validation(
    *,
    qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS,
    parameter_set_count: int = EXACT_REGIME_PARAMETER_SET_COUNT,
    verbose=False,
):
    workflow_results = run_exact_regime_workflow_matrix(
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
    )
    trace_result = capture_case(TRACE_CASE_NAME, run_optimization_trace)
    bundle = build_artifact_bundle(
        workflow_results,
        trace_result,
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
    )
    if verbose:
        print(
            "{} [{}] required cases={}/{} trace={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["required_passed_cases"],
                bundle["summary"]["required_cases"],
                trace_result["status"],
            )
        )
    return workflow_results, trace_result, bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the required local-noise workflow JSON artifacts.",
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
    _, trace_result, bundle = run_validation(
        qubit_sizes=tuple(args.qubit_sizes),
        parameter_set_count=args.parameter_set_count,
        verbose=not args.quiet,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_bundle(
        args.output_dir / WORKFLOW_BUNDLE_FILENAME,
        bundle,
        trace_result=trace_result,
    )
    write_json(args.output_dir / TRACE_ARTIFACT_FILENAME, trace_result)
    print(
        "Wrote {} and {} with status {} ({}/{})".format(
            args.output_dir / WORKFLOW_BUNDLE_FILENAME,
            args.output_dir / TRACE_ARTIFACT_FILENAME,
            bundle["status"],
            bundle["summary"]["required_passed_cases"],
            bundle["summary"]["required_cases"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
