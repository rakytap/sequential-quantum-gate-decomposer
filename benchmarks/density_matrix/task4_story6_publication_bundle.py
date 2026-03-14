#!/usr/bin/env python3
"""Validation: Task 4 Story 6 publication-ready evidence bundle.

Builds the top-level Task 4 manifest by assembling the delivered artifacts from
Stories 1 to 5 into one reproducible, machine-checkable package.

Run with:
    python benchmarks/density_matrix/task4_story6_publication_bundle.py
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
    PRIMARY_BACKEND,
    REFERENCE_BACKEND,
    STORY4_PARAMETER_SET_COUNT,
    STORY4_WORKFLOW_QUBITS,
    build_software_metadata,
    get_git_revision,
    write_json,
)
from benchmarks.density_matrix.task4_story1_required_local_noise_validation import (
    ARTIFACT_FILENAME as STORY1_ARTIFACT_FILENAME,
    write_artifact_bundle as write_story1_bundle_file,
)
from benchmarks.density_matrix.task4_story2_required_local_noise_micro_validation import (
    ARTIFACT_FILENAME as STORY2_ARTIFACT_FILENAME,
    write_artifact_bundle as write_story2_bundle_file,
)
from benchmarks.density_matrix.task4_story3_optional_noise_classification_validation import (
    ARTIFACT_FILENAME as STORY3_ARTIFACT_FILENAME,
    build_artifact_bundle as build_story3_bundle,
    run_validation as run_story3_validation,
    write_artifact_bundle as write_story3_bundle_file,
)
from benchmarks.density_matrix.task4_story4_unsupported_noise_validation import (
    ARTIFACT_FILENAME as STORY4_ARTIFACT_FILENAME,
    build_artifact_bundle as build_story4_bundle,
    run_validation as run_story4_validation,
    write_artifact_bundle as write_story4_bundle_file,
)
from benchmarks.density_matrix.task4_story5_required_local_noise_workflow_validation import (
    TRACE_ARTIFACT_FILENAME as STORY5_TRACE_ARTIFACT_FILENAME,
    WORKFLOW_BUNDLE_FILENAME as STORY5_WORKFLOW_BUNDLE_FILENAME,
    run_validation as run_story5_validation,
    write_artifact_bundle as write_story5_workflow_bundle_file,
)

SUITE_NAME = "task4_story6_publication_evidence"
ARTIFACT_FILENAME = "task4_story6_publication_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "phase2_task4"
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


def build_task4_story6_bundle(
    output_dir: Path,
    *,
    story1_bundle,
    story2_bundle,
    story3_bundle,
    story4_bundle,
    story5_workflow_bundle,
    story5_trace_result,
):
    output_dir = Path(output_dir)
    story1_command = (
        f"python benchmarks/density_matrix/task4_story1_required_local_noise_validation.py "
        f"--output-dir {output_dir}"
    )
    story2_command = (
        f"python benchmarks/density_matrix/task4_story2_required_local_noise_micro_validation.py "
        f"--output-dir {output_dir}"
    )
    story3_command = (
        f"python benchmarks/density_matrix/task4_story3_optional_noise_classification_validation.py "
        f"--output-dir {output_dir}"
    )
    story4_command = (
        f"python benchmarks/density_matrix/task4_story4_unsupported_noise_validation.py "
        f"--output-dir {output_dir}"
    )
    story5_command = (
        f"python benchmarks/density_matrix/task4_story5_required_local_noise_workflow_validation.py "
        f"--output-dir {output_dir}"
    )
    story6_command = (
        f"python benchmarks/density_matrix/task4_story6_publication_bundle.py "
        f"--output-dir {output_dir}"
    )

    artifacts = [
        _build_artifact_entry(
            artifact_id="task4_story1_required_local_noise_bundle",
            artifact_class="required_positive_path_bundle",
            mandatory=True,
            path=STORY1_ARTIFACT_FILENAME,
            status=story1_bundle["status"],
            expected_statuses=["pass"],
            purpose="Required positive-path evidence for the three mandatory local-noise models.",
            generation_command=story1_command,
            summary={
                "total_cases": story1_bundle["summary"]["total_cases"],
                "passed_cases": story1_bundle["summary"]["passed_cases"],
                "required_pass_rate": story1_bundle["summary"]["required_pass_rate"],
                "support_tiers_present": story1_bundle["summary"][
                    "support_tiers_present"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="task4_story2_required_local_noise_micro_bundle",
            artifact_class="required_micro_validation_bundle",
            mandatory=True,
            path=STORY2_ARTIFACT_FILENAME,
            status=story2_bundle["status"],
            expected_statuses=["pass"],
            purpose="Mandatory 1 to 3 qubit exact micro-validation for the required local-noise baseline.",
            generation_command=story2_command,
            summary={
                "total_cases": story2_bundle["summary"]["total_cases"],
                "passed_cases": story2_bundle["summary"]["passed_cases"],
                "required_pass_rate": story2_bundle["summary"]["required_pass_rate"],
                "required_noise_models_covered": story2_bundle["summary"][
                    "required_noise_models_covered"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="task4_story3_optional_noise_bundle",
            artifact_class="optional_classification_bundle",
            mandatory=True,
            path=STORY3_ARTIFACT_FILENAME,
            status=story3_bundle["status"],
            expected_statuses=["pass"],
            purpose="Optional-baseline classification that keeps whole-register depolarizing outside the mandatory baseline.",
            generation_command=story3_command,
            summary={
                "required_cases": story3_bundle["summary"]["required_cases"],
                "optional_cases": story3_bundle["summary"]["optional_cases"],
                "optional_pass_rate": story3_bundle["summary"]["optional_pass_rate"],
                "mandatory_baseline_completed": story3_bundle["summary"][
                    "mandatory_baseline_completed"
                ],
                "support_tiers_present": story3_bundle["summary"][
                    "support_tiers_present"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="task4_story4_unsupported_noise_bundle",
            artifact_class="unsupported_noise_bundle",
            mandatory=True,
            path=STORY4_ARTIFACT_FILENAME,
            status=story4_bundle["status"],
            expected_statuses=["pass"],
            purpose="Structured negative evidence for deferred families and invalid Task 4 noise schedule/configuration requests.",
            generation_command=story4_command,
            summary={
                "total_cases": story4_bundle["summary"]["total_cases"],
                "deferred_cases": story4_bundle["summary"]["deferred_cases"],
                "unsupported_cases": story4_bundle["summary"]["unsupported_cases"],
                "unsupported_status_cases": story4_bundle["summary"][
                    "unsupported_status_cases"
                ],
                "boundary_passed_cases": story4_bundle["summary"][
                    "boundary_passed_cases"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="task4_story5_required_local_noise_workflow_bundle",
            artifact_class="required_workflow_bundle",
            mandatory=True,
            path=STORY5_WORKFLOW_BUNDLE_FILENAME,
            status=story5_workflow_bundle["status"],
            expected_statuses=["pass"],
            purpose="Required-local-noise workflow sufficiency bundle across the accepted exact regime.",
            generation_command=story5_command,
            summary={
                "required_cases": story5_workflow_bundle["summary"]["required_cases"],
                "required_passed_cases": story5_workflow_bundle["summary"][
                    "required_passed_cases"
                ],
                "required_pass_rate": story5_workflow_bundle["summary"][
                    "required_pass_rate"
                ],
                "mandatory_baseline_completed": story5_workflow_bundle["summary"][
                    "mandatory_baseline_completed"
                ],
                "unsupported_status_cases": story5_workflow_bundle["summary"][
                    "unsupported_status_cases"
                ],
                "required_trace_case_name": story5_workflow_bundle["summary"][
                    "required_trace_case_name"
                ],
                "required_trace_completed": story5_workflow_bundle["summary"][
                    "required_trace_completed"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="task4_story5_required_local_noise_trace",
            artifact_class="required_workflow_trace",
            mandatory=True,
            path=STORY5_TRACE_ARTIFACT_FILENAME,
            status=story5_trace_result["status"],
            expected_statuses=["completed"],
            purpose="Bounded required-local-noise optimization trace for the anchor workflow.",
            generation_command=story5_command,
            summary={
                "case_name": story5_trace_result["case_name"],
                "support_tier": story5_trace_result["support_tier"],
                "case_purpose": story5_trace_result["case_purpose"],
                "required_story5_trace": story5_trace_result["required_story5_trace"],
                "workflow_completed": story5_trace_result["workflow_completed"],
                "optimizer": story5_trace_result["optimizer"],
                "parameter_count": story5_trace_result["parameter_count"],
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

    workflow_trace_reference_pass = (
        story5_workflow_bundle["summary"]["required_trace_case_name"]
        == story5_trace_result["case_name"]
    )

    bundle_status = (
        "pass"
        if present_count == len(mandatory_artifacts)
        and status_match_count == len(mandatory_artifacts)
        and workflow_trace_reference_pass
        else "fail"
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": bundle_status,
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": story6_command,
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "conda_env_assumption": "qgd",
        },
        "summary": {
            "mandatory_artifact_count": len(mandatory_artifacts),
            "present_artifact_count": present_count,
            "status_match_count": status_match_count,
            "missing_artifact_count": len(mandatory_artifacts) - present_count,
            "mismatched_status_count": len(mandatory_artifacts) - status_match_count,
            "workflow_trace_reference_pass": workflow_trace_reference_pass,
        },
        "artifacts": artifacts,
    }
    validate_task4_story6_bundle(bundle, output_dir)
    return bundle


def validate_task4_story6_bundle(bundle, bundle_dir: Path):
    missing_fields = [field for field in BUNDLE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 4 Story 6 bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    required_ids = {
        "task4_story1_required_local_noise_bundle",
        "task4_story2_required_local_noise_micro_bundle",
        "task4_story3_optional_noise_bundle",
        "task4_story4_unsupported_noise_bundle",
        "task4_story5_required_local_noise_workflow_bundle",
        "task4_story5_required_local_noise_trace",
    }
    artifact_ids = {artifact["artifact_id"] for artifact in bundle["artifacts"]}
    missing_ids = required_ids - artifact_ids
    if missing_ids:
        raise ValueError(
            "Task 4 Story 6 bundle is missing required artifact IDs: {}".format(
                ", ".join(sorted(missing_ids))
            )
        )

    required_payload_fields = {
        "task4_story1_required_local_noise_bundle": {
            "suite_name",
            "status",
            "requirements",
            "software",
            "summary",
            "cases",
        },
        "task4_story2_required_local_noise_micro_bundle": {
            "suite_name",
            "status",
            "requirements",
            "thresholds",
            "software",
            "summary",
            "cases",
        },
        "task4_story3_optional_noise_bundle": {
            "suite_name",
            "status",
            "requirements",
            "thresholds",
            "software",
            "summary",
            "cases",
            "required_artifacts",
        },
        "task4_story4_unsupported_noise_bundle": {
            "suite_name",
            "status",
            "requirements",
            "software",
            "summary",
            "cases",
        },
        "task4_story5_required_local_noise_workflow_bundle": {
            "suite_name",
            "status",
            "thresholds",
            "software",
            "summary",
            "cases",
        },
        "task4_story5_required_local_noise_trace": {
            "case_name",
            "status",
            "support_tier",
            "case_purpose",
            "required_story5_trace",
            "workflow_completed",
            "optimizer",
            "parameter_count",
        },
    }
    required_summary_fields = {
        "task4_story1_required_local_noise_bundle": {
            "total_cases",
            "passed_cases",
            "required_pass_rate",
        },
        "task4_story2_required_local_noise_micro_bundle": {
            "total_cases",
            "passed_cases",
            "required_pass_rate",
            "required_noise_models_covered",
        },
        "task4_story3_optional_noise_bundle": {
            "required_cases",
            "optional_cases",
            "optional_pass_rate",
            "mandatory_baseline_completed",
        },
        "task4_story4_unsupported_noise_bundle": {
            "deferred_cases",
            "unsupported_cases",
            "unsupported_status_cases",
            "boundary_passed_cases",
        },
        "task4_story5_required_local_noise_workflow_bundle": {
            "required_cases",
            "required_passed_cases",
            "required_pass_rate",
            "mandatory_baseline_completed",
            "unsupported_status_cases",
            "required_trace_case_name",
            "required_trace_present",
            "required_trace_completed",
            "required_trace_bridge_supported",
        },
    }

    loaded_payloads = {}
    for artifact in bundle["artifacts"]:
        artifact_path = bundle_dir / artifact["path"]
        if artifact["mandatory"] and not artifact_path.exists():
            raise ValueError(
                "Task 4 Story 6 bundle is missing artifact file: {}".format(
                    artifact["path"]
                )
            )
        if artifact["status"] not in artifact["expected_statuses"]:
            raise ValueError(
                "Task 4 Story 6 artifact {} has unexpected status {}".format(
                    artifact["artifact_id"], artifact["status"]
                )
            )

        payload = _load_json(artifact_path)
        loaded_payloads[artifact["artifact_id"]] = payload

        payload_missing_fields = [
            field
            for field in required_payload_fields[artifact["artifact_id"]]
            if field not in payload
        ]
        if payload_missing_fields:
            raise ValueError(
                "Artifact {} is missing required payload fields: {}".format(
                    artifact["artifact_id"], ", ".join(payload_missing_fields)
                )
            )

        if artifact["artifact_id"] in required_summary_fields:
            summary_missing_fields = [
                field
                for field in required_summary_fields[artifact["artifact_id"]]
                if field not in payload["summary"]
            ]
            if summary_missing_fields:
                raise ValueError(
                    "Artifact {} is missing required summary fields: {}".format(
                        artifact["artifact_id"], ", ".join(summary_missing_fields)
                    )
                )

    workflow_payload = loaded_payloads[
        "task4_story5_required_local_noise_workflow_bundle"
    ]
    trace_payload = loaded_payloads["task4_story5_required_local_noise_trace"]
    if workflow_payload["summary"]["required_trace_case_name"] != trace_payload["case_name"]:
        raise ValueError(
            "Story 5 workflow bundle trace reference does not match the Story 5 trace artifact"
        )


def write_task4_story6_bundle(output_path: Path, bundle):
    validate_task4_story6_bundle(bundle, output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_story6_bundle(
    output_dir: Path,
    *,
    qubit_sizes=STORY4_WORKFLOW_QUBITS,
    parameter_set_count: int = STORY4_PARAMETER_SET_COUNT,
    verbose=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    story1_bundle, story2_bundle, optional_results = run_story3_validation(
        verbose=verbose
    )
    story3_bundle = build_story3_bundle(story1_bundle, story2_bundle, optional_results)
    write_story1_bundle_file(output_dir / STORY1_ARTIFACT_FILENAME, story1_bundle)
    write_story2_bundle_file(output_dir / STORY2_ARTIFACT_FILENAME, story2_bundle)
    write_story3_bundle_file(output_dir / STORY3_ARTIFACT_FILENAME, story3_bundle)

    story4_results = run_story4_validation(verbose=verbose)
    story4_bundle = build_story4_bundle(story4_results)
    write_story4_bundle_file(output_dir / STORY4_ARTIFACT_FILENAME, story4_bundle)

    _, story5_trace_result, story5_workflow_bundle = run_story5_validation(
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
        verbose=verbose,
    )
    write_story5_workflow_bundle_file(
        output_dir / STORY5_WORKFLOW_BUNDLE_FILENAME,
        story5_workflow_bundle,
        trace_result=story5_trace_result,
    )
    write_json(output_dir / STORY5_TRACE_ARTIFACT_FILENAME, story5_trace_result)

    bundle = build_task4_story6_bundle(
        output_dir,
        story1_bundle=story1_bundle,
        story2_bundle=story2_bundle,
        story3_bundle=story3_bundle,
        story4_bundle=story4_bundle,
        story5_workflow_bundle=story5_workflow_bundle,
        story5_trace_result=story5_trace_result,
    )
    write_task4_story6_bundle(output_dir / ARTIFACT_FILENAME, bundle)
    return bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 4 Story 6 JSON artifacts.",
    )
    parser.add_argument(
        "--parameter-set-count",
        type=int,
        default=STORY4_PARAMETER_SET_COUNT,
        help="Number of fixed parameter vectors per workflow qubit size.",
    )
    parser.add_argument(
        "--qubit-sizes",
        type=int,
        nargs="*",
        default=list(STORY4_WORKFLOW_QUBITS),
        help="Workflow qubit sizes to include in the Story 5 prerequisite bundle.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output from prerequisite story generators.",
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
            bundle["summary"]["present_artifact_count"],
            bundle["summary"]["mandatory_artifact_count"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
