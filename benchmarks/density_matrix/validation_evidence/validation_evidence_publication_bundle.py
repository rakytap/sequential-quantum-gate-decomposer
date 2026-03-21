#!/usr/bin/env python3
"""Validation: Task 5 Story 6 publication-ready validation bundle.

Builds the top-level Task 5 manifest by assembling the delivered artifacts from
Stories 1 to 5 into one reproducible, machine-checkable package. The bundle
preserves the canonical raw trace artifact explicitly while keeping the other
Task 5 story bundles as first-class top-level evidence layers.

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
    ARTIFACT_FILENAME as STORY1_ARTIFACT_FILENAME,
    run_validation as run_story1_validation,
    write_artifact_bundle as write_story1_bundle_file,
)
from benchmarks.density_matrix.validation_evidence.workflow_baseline_validation import (
    ARTIFACT_FILENAME as STORY2_ARTIFACT_FILENAME,
    run_validation as run_story2_validation,
    write_artifact_bundle as write_story2_bundle_file,
)
from benchmarks.density_matrix.validation_evidence.trace_anchor_validation import (
    ARTIFACT_FILENAME as STORY3_ARTIFACT_FILENAME,
    TRACE_CASE_NAME,
    TRACE_ARTIFACT_FILENAME as STORY3_TRACE_ARTIFACT_FILENAME,
    build_artifact_bundle as build_story3_bundle,
    write_artifact_bundle as write_story3_bundle_file,
)
from benchmarks.density_matrix.validation_evidence.metric_completeness_validation import (
    ARTIFACT_FILENAME as STORY4_ARTIFACT_FILENAME,
    build_artifact_bundle as build_story4_bundle,
    write_artifact_bundle as write_story4_bundle_file,
)
from benchmarks.density_matrix.validation_evidence.interpretation_validation import (
    ARTIFACT_FILENAME as STORY5_ARTIFACT_FILENAME,
    build_artifact_bundle as build_exact_density_validation_bundle,
    write_artifact_bundle as write_exact_density_validation_bundle_file,
)
from benchmarks.density_matrix.noise_support.optional_noise_classification_validation import (
    build_artifact_bundle as build_task4_story3_bundle,
    run_validation as run_task4_story3_validation,
)
from benchmarks.density_matrix.noise_support.unsupported_noise_validation import (
    build_artifact_bundle as build_task4_story4_bundle,
    run_validation as run_task4_story4_validation,
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


def build_planner_calibration_story6_bundle(
    output_dir: Path,
    *,
    story1_bundle,
    story2_bundle,
    story3_bundle,
    story3_trace_result,
    story4_bundle,
    story5_bundle,
):
    output_dir = Path(output_dir)
    story1_command = (
        f"python benchmarks/density_matrix/validation_evidence/local_correctness_validation.py "
        f"--output-dir {output_dir}"
    )
    story2_command = (
        f"python benchmarks/density_matrix/validation_evidence/workflow_baseline_validation.py "
        f"--output-dir {output_dir}"
    )
    story3_command = (
        f"python benchmarks/density_matrix/validation_evidence/trace_anchor_validation.py "
        f"--output-dir {output_dir}"
    )
    story4_command = (
        f"python benchmarks/density_matrix/validation_evidence/metric_completeness_validation.py "
        f"--output-dir {output_dir}"
    )
    story5_command = (
        f"python benchmarks/density_matrix/validation_evidence/interpretation_validation.py "
        f"--output-dir {output_dir}"
    )
    story6_command = (
        f"python benchmarks/density_matrix/validation_evidence/validation_evidence_publication_bundle.py "
        f"--output-dir {output_dir}"
    )

    artifacts = [
        _build_artifact_entry(
            artifact_id="local_correctness_bundle",
            artifact_class="local_correctness_baseline_bundle",
            mandatory=True,
            path=STORY1_ARTIFACT_FILENAME,
            status=story1_bundle["status"],
            expected_statuses=["pass"],
            purpose="Phase-level local correctness gate for the mandatory 1 to 3 qubit micro-validation matrix.",
            generation_command=story1_command,
            summary={
                "required_cases": story1_bundle["summary"]["required_cases"],
                "required_passed_cases": story1_bundle["summary"][
                    "required_passed_cases"
                ],
                "required_pass_rate": story1_bundle["summary"]["required_pass_rate"],
                "stable_case_ids_present": story1_bundle["summary"][
                    "stable_case_ids_present"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="workflow_baseline_bundle",
            artifact_class="workflow_exact_regime_baseline_bundle",
            mandatory=True,
            path=STORY2_ARTIFACT_FILENAME,
            status=story2_bundle["status"],
            expected_statuses=["pass"],
            purpose="Phase-level 4/6/8/10 workflow-scale exact-regime baseline.",
            generation_command=story2_command,
            summary={
                "required_cases": story2_bundle["summary"]["required_cases"],
                "required_passed_cases": story2_bundle["summary"][
                    "required_passed_cases"
                ],
                "required_pass_rate": story2_bundle["summary"]["required_pass_rate"],
                "documented_10q_anchor_present": story2_bundle["summary"][
                    "documented_10q_anchor_present"
                ],
                "stable_case_ids_present": story2_bundle["summary"][
                    "stable_case_ids_present"
                ],
                "stable_parameter_set_ids_present": story2_bundle["summary"][
                    "stable_parameter_set_ids_present"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="trace_anchor_bundle",
            artifact_class="trace_and_anchor_bundle",
            mandatory=True,
            path=STORY3_ARTIFACT_FILENAME,
            status=story3_bundle["status"],
            expected_statuses=["pass"],
            purpose="Phase-level bounded trace plus documented 10-qubit anchor package.",
            generation_command=story3_command,
            summary={
                "documented_10q_anchor_present": story3_bundle["summary"][
                    "documented_10q_anchor_present"
                ],
                "required_trace_case_name": story3_bundle["summary"][
                    "required_trace_case_name"
                ],
                "required_trace_present": story3_bundle["summary"][
                    "required_trace_present"
                ],
                "required_trace_completed": story3_bundle["summary"][
                    "required_trace_completed"
                ],
                "required_trace_bridge_supported": story3_bundle["summary"][
                    "required_trace_bridge_supported"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="optimization_trace_4q",
            artifact_class="raw_trace_artifact",
            mandatory=True,
            path=STORY3_TRACE_ARTIFACT_FILENAME,
            status=story3_trace_result["status"],
            expected_statuses=["completed"],
            purpose="Canonical raw bounded optimization trace linked from the Task 5 Story 3 bundle.",
            generation_command=story3_command,
            summary={
                "case_name": story3_trace_result["case_name"],
                "workflow_completed": story3_trace_result["workflow_completed"],
                "bridge_supported_pass": story3_trace_result["bridge_supported_pass"],
                "required_validation_trace": story3_trace_result["required_validation_trace"],
            },
        ),
        _build_artifact_entry(
            artifact_id="metric_completeness_bundle",
            artifact_class="metric_completeness_bundle",
            mandatory=True,
            path=STORY4_ARTIFACT_FILENAME,
            status=story4_bundle["status"],
            expected_statuses=["pass"],
            purpose="Phase-level metric-completeness gate for mandatory supported evidence.",
            generation_command=story4_command,
            summary={
                "micro_cases_missing_required_metrics": story4_bundle["summary"][
                    "micro_cases_missing_required_metrics"
                ],
                "workflow_cases_missing_required_metrics": story4_bundle["summary"][
                    "workflow_cases_missing_required_metrics"
                ],
                "trace_artifacts_missing_required_metrics": story4_bundle["summary"][
                    "trace_artifacts_missing_required_metrics"
                ],
                "metric_completeness_gate_completed": story4_bundle["summary"][
                    "metric_completeness_gate_completed"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="interpretation_bundle",
            artifact_class="interpretation_guardrail_bundle",
            mandatory=True,
            path=STORY5_ARTIFACT_FILENAME,
            status=story5_bundle["status"],
            expected_statuses=["pass"],
            purpose="Phase-level interpretation guardrails preventing optional, unsupported, or incomplete evidence from inflating the main claim.",
            generation_command=story5_command,
            summary={
                "mandatory_artifacts_complete": story5_bundle["summary"][
                    "mandatory_artifacts_complete"
                ],
                "optional_evidence_supplemental": story5_bundle["summary"][
                    "optional_evidence_supplemental"
                ],
                "unsupported_evidence_negative_only": story5_bundle["summary"][
                    "unsupported_evidence_negative_only"
                ],
                "main_phase2_claim_completed": story5_bundle["summary"][
                    "main_phase2_claim_completed"
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
        "backend": story1_bundle["backend"],
        "reference_backend": story1_bundle["reference_backend"],
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": story6_command,
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
        },
        "summary": {
            "mandatory_artifact_count": len(mandatory_artifacts),
            "present_artifact_count": present_count,
            "status_match_count": status_match_count,
            "missing_artifact_count": len(mandatory_artifacts) - present_count,
            "mismatched_status_count": len(mandatory_artifacts) - status_match_count,
            "raw_trace_reference_pass": story3_bundle["summary"][
                "required_trace_case_name"
            ]
            == story3_trace_result["case_name"],
        },
        "artifacts": artifacts,
    }
    validate_planner_calibration_story6_bundle(bundle, output_dir)
    return bundle


def validate_planner_calibration_story6_bundle(bundle, bundle_dir: Path):
    missing_fields = [field for field in BUNDLE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 5 Story 6 bundle is missing required fields: {}".format(
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
            "Task 5 Story 6 bundle is missing required artifact IDs: {}".format(
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
            "main_phase2_claim_completed",
        },
    }

    for artifact in bundle["artifacts"]:
        artifact_path = bundle_dir / artifact["path"]
        if artifact["mandatory"] and not artifact_path.exists():
            raise ValueError(
                f"Task 5 Story 6 bundle is missing artifact file: {artifact['path']}"
            )
        if artifact["status"] not in artifact["expected_statuses"]:
            raise ValueError(
                "Task 5 Story 6 artifact {} has unexpected status {}".format(
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


def write_planner_calibration_story6_bundle(output_path: Path, bundle):
    validate_planner_calibration_story6_bundle(bundle, output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate_story6_bundle(
    output_dir: Path,
    *,
    qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS,
    parameter_set_count: int = EXACT_REGIME_PARAMETER_SET_COUNT,
    verbose=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, story1_bundle = run_story1_validation(verbose=verbose)
    write_story1_bundle_file(output_dir / STORY1_ARTIFACT_FILENAME, story1_bundle)

    _, story2_bundle = run_story2_validation(
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
        verbose=verbose,
    )
    write_story2_bundle_file(output_dir / STORY2_ARTIFACT_FILENAME, story2_bundle)

    story3_trace_result = capture_case(TRACE_CASE_NAME, run_optimization_trace)
    story3_bundle = build_story3_bundle(
        story2_bundle,
        story3_trace_result,
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
    )
    write_story3_bundle_file(output_dir / STORY3_ARTIFACT_FILENAME, story3_bundle)
    write_json(output_dir / STORY3_TRACE_ARTIFACT_FILENAME, story3_trace_result)

    story4_bundle = build_story4_bundle(story1_bundle, story2_bundle, story3_bundle)
    write_story4_bundle_file(output_dir / STORY4_ARTIFACT_FILENAME, story4_bundle)

    task4_story3_story1_bundle, task4_story3_story2_bundle, optional_results = (
        run_task4_story3_validation(verbose=verbose)
    )
    optional_bundle = build_task4_story3_bundle(
        task4_story3_story1_bundle,
        task4_story3_story2_bundle,
        optional_results,
    )
    unsupported_results = run_task4_story4_validation(verbose=verbose)
    unsupported_bundle = build_task4_story4_bundle(unsupported_results)
    story5_bundle = build_exact_density_validation_bundle(
        story4_bundle,
        optional_bundle,
        unsupported_bundle,
    )
    write_exact_density_validation_bundle_file(output_dir / STORY5_ARTIFACT_FILENAME, story5_bundle)

    bundle = build_planner_calibration_story6_bundle(
        output_dir,
        story1_bundle=story1_bundle,
        story2_bundle=story2_bundle,
        story3_bundle=story3_bundle,
        story3_trace_result=story3_trace_result,
        story4_bundle=story4_bundle,
        story5_bundle=story5_bundle,
    )
    write_planner_calibration_story6_bundle(output_dir / ARTIFACT_FILENAME, bundle)
    return bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 5 Story 6 JSON artifacts.",
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
    bundle = generate_story6_bundle(
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
