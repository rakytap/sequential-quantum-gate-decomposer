#!/usr/bin/env python3
"""Validation: publication-ready validation bundle.

Builds the top-level validation-evidence manifest by assembling the delivered
artifacts into one reproducible, machine-checkable package. The bundle
preserves the canonical raw trace artifact explicitly while keeping the other
validation-evidence bundles as first-class top-level evidence layers.

Run with:
    python benchmarks/density_matrix/validation_evidence/validation_evidence_publication_bundle.py
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
    build_software_metadata,
    capture_case,
    get_git_revision,
    run_optimization_trace,
    write_json,
)
from benchmarks.density_matrix.validation_evidence.local_correctness_validation import (
    ARTIFACT_FILENAME as LOCAL_CORRECTNESS_ARTIFACT_FILENAME,
    run_validation as run_local_correctness_validation,
    write_artifact_bundle as write_local_correctness_bundle_file,
)
from benchmarks.density_matrix.validation_evidence.workflow_baseline_validation import (
    ARTIFACT_FILENAME as WORKFLOW_BASELINE_ARTIFACT_FILENAME,
    run_validation as run_workflow_baseline_validation,
    write_artifact_bundle as write_workflow_baseline_bundle_file,
)
from benchmarks.density_matrix.validation_evidence.trace_anchor_validation import (
    ARTIFACT_FILENAME as TRACE_ANCHOR_ARTIFACT_FILENAME,
    TRACE_CASE_NAME,
    TRACE_ARTIFACT_FILENAME as TRACE_ARTIFACT_FILENAME,
    build_artifact_bundle as build_trace_anchor_bundle,
    write_artifact_bundle as write_trace_anchor_bundle_file,
)
from benchmarks.density_matrix.validation_evidence.metric_completeness_validation import (
    ARTIFACT_FILENAME as METRIC_COMPLETENESS_ARTIFACT_FILENAME,
    build_artifact_bundle as build_metric_completeness_bundle,
    write_artifact_bundle as write_metric_completeness_bundle_file,
)
from benchmarks.density_matrix.validation_evidence.interpretation_validation import (
    ARTIFACT_FILENAME as INTERPRETATION_ARTIFACT_FILENAME,
    build_artifact_bundle as build_validation_interpretation_bundle,
    write_artifact_bundle as write_validation_interpretation_bundle_file,
)
from benchmarks.density_matrix.noise_support.optional_noise_classification_validation import (
    build_artifact_bundle as build_optional_noise_bundle,
    run_validation as run_optional_noise_validation,
)
from benchmarks.density_matrix.noise_support.unsupported_noise_validation import (
    build_artifact_bundle as build_unsupported_noise_bundle,
    run_validation as run_unsupported_noise_validation,
)

SUITE_NAME = "validation_evidence_publication"
ARTIFACT_FILENAME = "validation_evidence_publication_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "validation_evidence"
)
BUNDLE_FIELDS = (
    "suite_name",
    "status",
    "backend",
    "reference_backend",
    "software",
    "provenance",
    "summary",
    "artifacts",
)


def _build_artifact_entry(
    *,
    artifact_id,
    artifact_class,
    mandatory,
    path,
    status,
    expected_statuses,
    purpose,
    generation_command,
    summary,
):
    return {
        "artifact_id": artifact_id,
        "artifact_class": artifact_class,
        "mandatory": mandatory,
        "path": path,
        "status": status,
        "expected_statuses": list(expected_statuses),
        "purpose": purpose,
        "generation_command": generation_command,
        "summary": dict(summary),
    }


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_validation_evidence_publication_bundle(
    output_dir: Path,
    *,
    local_correctness_bundle,
    workflow_baseline_bundle,
    trace_anchor_bundle,
    trace_artifact,
    metric_completeness_bundle,
    interpretation_bundle,
):
    output_dir = Path(output_dir)
    local_correctness_command = (
        f"python benchmarks/density_matrix/validation_evidence/local_correctness_validation.py "
        f"--output-dir {output_dir}"
    )
    workflow_baseline_command = (
        f"python benchmarks/density_matrix/validation_evidence/workflow_baseline_validation.py "
        f"--output-dir {output_dir}"
    )
    trace_anchor_command = (
        f"python benchmarks/density_matrix/validation_evidence/trace_anchor_validation.py "
        f"--output-dir {output_dir}"
    )
    metric_completeness_command = (
        f"python benchmarks/density_matrix/validation_evidence/metric_completeness_validation.py "
        f"--output-dir {output_dir}"
    )
    interpretation_command = (
        f"python benchmarks/density_matrix/validation_evidence/interpretation_validation.py "
        f"--output-dir {output_dir}"
    )
    publication_bundle_command = (
        f"python benchmarks/density_matrix/validation_evidence/validation_evidence_publication_bundle.py "
        f"--output-dir {output_dir}"
    )

    artifacts = [
        _build_artifact_entry(
            artifact_id="local_correctness_bundle",
            artifact_class="local_correctness_baseline_bundle",
            mandatory=True,
            path=LOCAL_CORRECTNESS_ARTIFACT_FILENAME,
            status=local_correctness_bundle["status"],
            expected_statuses=["pass"],
            purpose="Phase-level local correctness gate for the mandatory 1 to 3 qubit micro-validation matrix.",
            generation_command=local_correctness_command,
            summary={
                "required_cases": local_correctness_bundle["summary"]["required_cases"],
                "required_passed_cases": local_correctness_bundle["summary"][
                    "required_passed_cases"
                ],
                "required_pass_rate": local_correctness_bundle["summary"][
                    "required_pass_rate"
                ],
                "stable_case_ids_present": local_correctness_bundle["summary"][
                    "stable_case_ids_present"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="workflow_baseline_bundle",
            artifact_class="workflow_exact_regime_baseline_bundle",
            mandatory=True,
            path=WORKFLOW_BASELINE_ARTIFACT_FILENAME,
            status=workflow_baseline_bundle["status"],
            expected_statuses=["pass"],
            purpose="Phase-level 4/6/8/10 workflow-scale exact-regime baseline.",
            generation_command=workflow_baseline_command,
            summary={
                "required_cases": workflow_baseline_bundle["summary"]["required_cases"],
                "required_passed_cases": workflow_baseline_bundle["summary"][
                    "required_passed_cases"
                ],
                "required_pass_rate": workflow_baseline_bundle["summary"][
                    "required_pass_rate"
                ],
                "documented_10q_anchor_present": workflow_baseline_bundle["summary"][
                    "documented_10q_anchor_present"
                ],
                "stable_case_ids_present": workflow_baseline_bundle["summary"][
                    "stable_case_ids_present"
                ],
                "stable_parameter_set_ids_present": workflow_baseline_bundle["summary"][
                    "stable_parameter_set_ids_present"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="trace_anchor_bundle",
            artifact_class="trace_and_anchor_bundle",
            mandatory=True,
            path=TRACE_ANCHOR_ARTIFACT_FILENAME,
            status=trace_anchor_bundle["status"],
            expected_statuses=["pass"],
            purpose="Phase-level bounded trace plus documented 10-qubit anchor package.",
            generation_command=trace_anchor_command,
            summary={
                "documented_10q_anchor_present": trace_anchor_bundle["summary"][
                    "documented_10q_anchor_present"
                ],
                "required_trace_case_name": trace_anchor_bundle["summary"][
                    "required_trace_case_name"
                ],
                "required_trace_present": trace_anchor_bundle["summary"][
                    "required_trace_present"
                ],
                "required_trace_completed": trace_anchor_bundle["summary"][
                    "required_trace_completed"
                ],
                "required_trace_bridge_supported": trace_anchor_bundle["summary"][
                    "required_trace_bridge_supported"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="optimization_trace_4q",
            artifact_class="raw_trace_artifact",
            mandatory=True,
            path=TRACE_ARTIFACT_FILENAME,
            status=trace_artifact["status"],
            expected_statuses=["completed"],
            purpose="Canonical raw bounded optimization trace linked from the trace-and-anchor bundle.",
            generation_command=trace_anchor_command,
            summary={
                "case_name": trace_artifact["case_name"],
                "workflow_completed": trace_artifact["workflow_completed"],
                "bridge_supported_pass": trace_artifact["bridge_supported_pass"],
                "required_validation_trace": trace_artifact["required_validation_trace"],
            },
        ),
        _build_artifact_entry(
            artifact_id="metric_completeness_bundle",
            artifact_class="metric_completeness_bundle",
            mandatory=True,
            path=METRIC_COMPLETENESS_ARTIFACT_FILENAME,
            status=metric_completeness_bundle["status"],
            expected_statuses=["pass"],
            purpose="Phase-level metric-completeness gate for mandatory supported evidence.",
            generation_command=metric_completeness_command,
            summary={
                "micro_cases_missing_required_metrics": metric_completeness_bundle[
                    "summary"
                ][
                    "micro_cases_missing_required_metrics"
                ],
                "workflow_cases_missing_required_metrics": metric_completeness_bundle[
                    "summary"
                ][
                    "workflow_cases_missing_required_metrics"
                ],
                "trace_artifacts_missing_required_metrics": metric_completeness_bundle[
                    "summary"
                ][
                    "trace_artifacts_missing_required_metrics"
                ],
                "metric_completeness_gate_completed": metric_completeness_bundle[
                    "summary"
                ][
                    "metric_completeness_gate_completed"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="interpretation_bundle",
            artifact_class="interpretation_guardrail_bundle",
            mandatory=True,
            path=INTERPRETATION_ARTIFACT_FILENAME,
            status=interpretation_bundle["status"],
            expected_statuses=["pass"],
            purpose="Phase-level interpretation guardrails preventing optional, unsupported, or incomplete evidence from inflating the main claim.",
            generation_command=interpretation_command,
            summary={
                "mandatory_artifacts_complete": interpretation_bundle["summary"][
                    "mandatory_artifacts_complete"
                ],
                "optional_evidence_supplemental": interpretation_bundle["summary"][
                    "optional_evidence_supplemental"
                ],
                "unsupported_evidence_negative_only": interpretation_bundle["summary"][
                    "unsupported_evidence_negative_only"
                ],
                "main_validation_claim_completed": interpretation_bundle["summary"][
                    "main_validation_claim_completed"
                ],
            },
        ),
    ]

    mandatory_artifacts = [artifact for artifact in artifacts if artifact["mandatory"]]
    present_count = 0
    status_match_count = 0
    for artifact in mandatory_artifacts:
        if (output_dir / artifact["path"]).exists():
            present_count += 1
        if artifact["status"] in artifact["expected_statuses"]:
            status_match_count += 1

    bundle_status = (
        "pass"
        if present_count == len(mandatory_artifacts)
        and status_match_count == len(mandatory_artifacts)
        else "fail"
    )
    bundle = {
        "suite_name": SUITE_NAME,
        "status": bundle_status,
        "backend": local_correctness_bundle["backend"],
        "reference_backend": local_correctness_bundle["reference_backend"],
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": publication_bundle_command,
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
        },
        "summary": {
            "mandatory_artifact_count": len(mandatory_artifacts),
            "present_artifact_count": present_count,
            "status_match_count": status_match_count,
            "missing_artifact_count": len(mandatory_artifacts) - present_count,
            "mismatched_status_count": len(mandatory_artifacts) - status_match_count,
            "raw_trace_reference_pass": trace_anchor_bundle["summary"][
                "required_trace_case_name"
            ]
            == trace_artifact["case_name"],
        },
        "artifacts": artifacts,
    }
    validate_validation_evidence_publication_bundle(bundle, output_dir)
    return bundle


def validate_validation_evidence_publication_bundle(bundle, bundle_dir: Path):
    missing_fields = [field for field in BUNDLE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Validation-evidence publication bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    artifact_ids = {artifact["artifact_id"] for artifact in bundle["artifacts"]}
    required_ids = {
        "local_correctness_bundle",
        "workflow_baseline_bundle",
        "trace_anchor_bundle",
        "optimization_trace_4q",
        "metric_completeness_bundle",
        "interpretation_bundle",
    }
    missing_ids = required_ids - artifact_ids
    if missing_ids:
        raise ValueError(
            "Validation-evidence publication bundle is missing required artifact IDs: {}".format(
                ", ".join(sorted(missing_ids))
            )
        )

    required_summary_fields = {
        "local_correctness_bundle": {
            "required_cases",
            "required_passed_cases",
            "required_pass_rate",
            "stable_case_ids_present",
        },
        "workflow_baseline_bundle": {
            "required_cases",
            "required_passed_cases",
            "required_pass_rate",
            "documented_10q_anchor_present",
            "stable_case_ids_present",
            "stable_parameter_set_ids_present",
        },
        "trace_anchor_bundle": {
            "documented_10q_anchor_present",
            "required_trace_case_name",
            "required_trace_present",
            "required_trace_completed",
            "required_trace_bridge_supported",
        },
        "optimization_trace_4q": {
            "case_name",
            "workflow_completed",
            "bridge_supported_pass",
            "required_validation_trace",
        },
        "metric_completeness_bundle": {
            "micro_cases_missing_required_metrics",
            "workflow_cases_missing_required_metrics",
            "trace_artifacts_missing_required_metrics",
            "metric_completeness_gate_completed",
        },
        "interpretation_bundle": {
            "mandatory_artifacts_complete",
            "optional_evidence_supplemental",
            "unsupported_evidence_negative_only",
            "main_validation_claim_completed",
        },
    }

    for artifact in bundle["artifacts"]:
        artifact_path = bundle_dir / artifact["path"]
        if artifact["mandatory"] and not artifact_path.exists():
            raise ValueError(
                f"Validation-evidence publication bundle is missing artifact file: {artifact['path']}"
            )
        if artifact["status"] not in artifact["expected_statuses"]:
            raise ValueError(
                "Validation-evidence publication artifact {} has unexpected status {}".format(
                    artifact["artifact_id"], artifact["status"]
                )
            )

        payload = _load_json(artifact_path)
        required_fields = required_summary_fields[artifact["artifact_id"]]
        target = payload.get("summary", payload)
        missing_summary_fields = [
            field for field in required_fields if field not in target
        ]
        if missing_summary_fields:
            raise ValueError(
                "Artifact {} is missing required summary fields: {}".format(
                    artifact["artifact_id"], ", ".join(missing_summary_fields)
                )
            )


def write_validation_evidence_publication_bundle(output_path: Path, bundle):
    validate_validation_evidence_publication_bundle(bundle, output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate_validation_evidence_publication_bundle(
    output_dir: Path,
    *,
    qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS,
    parameter_set_count: int = EXACT_REGIME_PARAMETER_SET_COUNT,
    verbose=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, local_correctness_bundle = run_local_correctness_validation(verbose=verbose)
    write_local_correctness_bundle_file(
        output_dir / LOCAL_CORRECTNESS_ARTIFACT_FILENAME,
        local_correctness_bundle,
    )

    _, workflow_baseline_bundle = run_workflow_baseline_validation(
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
        verbose=verbose,
    )
    write_workflow_baseline_bundle_file(
        output_dir / WORKFLOW_BASELINE_ARTIFACT_FILENAME,
        workflow_baseline_bundle,
    )

    trace_artifact = capture_case(TRACE_CASE_NAME, run_optimization_trace)
    trace_anchor_bundle = build_trace_anchor_bundle(
        workflow_baseline_bundle,
        trace_artifact,
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
    )
    write_trace_anchor_bundle_file(
        output_dir / TRACE_ANCHOR_ARTIFACT_FILENAME,
        trace_anchor_bundle,
    )
    write_json(output_dir / TRACE_ARTIFACT_FILENAME, trace_artifact)

    metric_completeness_bundle = build_metric_completeness_bundle(
        local_correctness_bundle,
        workflow_baseline_bundle,
        trace_anchor_bundle,
    )
    write_metric_completeness_bundle_file(
        output_dir / METRIC_COMPLETENESS_ARTIFACT_FILENAME,
        metric_completeness_bundle,
    )

    required_local_noise_bundle, required_local_noise_micro_bundle, optional_results = (
        run_optional_noise_validation(verbose=verbose)
    )
    optional_bundle = build_optional_noise_bundle(
        required_local_noise_bundle,
        required_local_noise_micro_bundle,
        optional_results,
    )
    unsupported_results = run_unsupported_noise_validation(verbose=verbose)
    unsupported_bundle = build_unsupported_noise_bundle(unsupported_results)
    interpretation_bundle = build_validation_interpretation_bundle(
        metric_completeness_bundle,
        optional_bundle,
        unsupported_bundle,
    )
    write_validation_interpretation_bundle_file(
        output_dir / INTERPRETATION_ARTIFACT_FILENAME,
        interpretation_bundle,
    )

    bundle = build_validation_evidence_publication_bundle(
        output_dir,
        local_correctness_bundle=local_correctness_bundle,
        workflow_baseline_bundle=workflow_baseline_bundle,
        trace_anchor_bundle=trace_anchor_bundle,
        trace_artifact=trace_artifact,
        metric_completeness_bundle=metric_completeness_bundle,
        interpretation_bundle=interpretation_bundle,
    )
    write_validation_evidence_publication_bundle(output_dir / ARTIFACT_FILENAME, bundle)
    return bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the validation-evidence publication JSON artifacts.",
    )
    parser.add_argument(
        "--parameter-set-count",
        type=int,
        default=EXACT_REGIME_PARAMETER_SET_COUNT,
        help="Number of fixed parameter vectors per workflow qubit size.",
    )
    parser.add_argument(
        "--qubit-sizes",
        type=int,
        nargs="*",
        default=list(EXACT_REGIME_WORKFLOW_QUBITS),
        help="Workflow qubit sizes to include in the mandatory baseline.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = generate_validation_evidence_publication_bundle(
        args.output_dir,
        qubit_sizes=tuple(args.qubit_sizes),
        parameter_set_count=args.parameter_set_count,
        verbose=not args.quiet,
    )
    print(
        "Wrote {} with status {} ({}/{})".format(
            args.output_dir / ARTIFACT_FILENAME,
            bundle["status"],
            bundle["summary"]["status_match_count"],
            bundle["summary"]["mandatory_artifact_count"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
