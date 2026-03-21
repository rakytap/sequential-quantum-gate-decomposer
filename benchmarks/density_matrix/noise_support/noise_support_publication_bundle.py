#!/usr/bin/env python3
"""Validation: publication-ready noise-support evidence bundle.

Builds the top-level noise-support manifest by assembling the delivered
artifacts into one reproducible, machine-checkable package.

Run with:
    python benchmarks/density_matrix/noise_support/noise_support_publication_bundle.py
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
    build_software_metadata,
    get_git_revision,
    write_json,
)
from benchmarks.density_matrix.noise_support.required_local_noise_validation import (
    ARTIFACT_FILENAME as REQUIRED_LOCAL_NOISE_ARTIFACT_FILENAME,
    write_artifact_bundle as write_required_local_noise_bundle_file,
)
from benchmarks.density_matrix.noise_support.required_local_noise_micro_validation import (
    ARTIFACT_FILENAME as REQUIRED_LOCAL_NOISE_MICRO_ARTIFACT_FILENAME,
    write_artifact_bundle as write_required_local_noise_micro_bundle_file,
)
from benchmarks.density_matrix.noise_support.optional_noise_classification_validation import (
    ARTIFACT_FILENAME as OPTIONAL_NOISE_CLASSIFICATION_ARTIFACT_FILENAME,
    build_artifact_bundle as build_optional_noise_classification_bundle,
    run_validation as run_optional_noise_classification_validation,
    write_artifact_bundle as write_optional_noise_classification_bundle_file,
)
from benchmarks.density_matrix.noise_support.unsupported_noise_validation import (
    ARTIFACT_FILENAME as UNSUPPORTED_NOISE_ARTIFACT_FILENAME,
    build_artifact_bundle as build_unsupported_noise_bundle,
    run_validation as run_unsupported_noise_validation,
    write_artifact_bundle as write_unsupported_noise_bundle_file,
)
from benchmarks.density_matrix.noise_support.required_local_noise_workflow_validation import (
    TRACE_ARTIFACT_FILENAME as REQUIRED_LOCAL_NOISE_TRACE_ARTIFACT_FILENAME,
    WORKFLOW_BUNDLE_FILENAME as REQUIRED_LOCAL_NOISE_WORKFLOW_BUNDLE_FILENAME,
    run_validation as run_required_local_noise_workflow_validation,
    write_artifact_bundle as write_required_local_noise_workflow_bundle_file,
)

SUITE_NAME = "noise_support_publication_evidence"
ARTIFACT_FILENAME = "noise_support_publication_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "noise_support"
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


def build_noise_support_publication_bundle(
    output_dir: Path,
    *,
    required_local_noise_bundle,
    required_local_noise_micro_bundle,
    optional_noise_classification_bundle,
    unsupported_noise_bundle,
    required_local_noise_workflow_bundle,
    required_local_noise_trace_result,
):
    output_dir = Path(output_dir)
    required_local_noise_command = (
        f"python benchmarks/density_matrix/noise_support/required_local_noise_validation.py "
        f"--output-dir {output_dir}"
    )
    required_local_noise_micro_command = (
        f"python benchmarks/density_matrix/noise_support/required_local_noise_micro_validation.py "
        f"--output-dir {output_dir}"
    )
    optional_noise_command = (
        f"python benchmarks/density_matrix/noise_support/optional_noise_classification_validation.py "
        f"--output-dir {output_dir}"
    )
    unsupported_noise_command = (
        f"python benchmarks/density_matrix/noise_support/unsupported_noise_validation.py "
        f"--output-dir {output_dir}"
    )
    required_local_noise_workflow_command = (
        f"python benchmarks/density_matrix/noise_support/required_local_noise_workflow_validation.py "
        f"--output-dir {output_dir}"
    )
    publication_bundle_command = (
        f"python benchmarks/density_matrix/noise_support/noise_support_publication_bundle.py "
        f"--output-dir {output_dir}"
    )

    artifacts = [
        _build_artifact_entry(
            artifact_id="required_local_noise_bundle",
            artifact_class="required_positive_path_bundle",
            mandatory=True,
            path=REQUIRED_LOCAL_NOISE_ARTIFACT_FILENAME,
            status=required_local_noise_bundle["status"],
            expected_statuses=["pass"],
            purpose="Required positive-path evidence for the three mandatory local-noise models.",
            generation_command=required_local_noise_command,
            summary={
                "total_cases": required_local_noise_bundle["summary"]["total_cases"],
                "passed_cases": required_local_noise_bundle["summary"]["passed_cases"],
                "required_pass_rate": required_local_noise_bundle["summary"][
                    "required_pass_rate"
                ],
                "support_tiers_present": required_local_noise_bundle["summary"][
                    "support_tiers_present"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="required_local_noise_micro_bundle",
            artifact_class="required_micro_validation_bundle",
            mandatory=True,
            path=REQUIRED_LOCAL_NOISE_MICRO_ARTIFACT_FILENAME,
            status=required_local_noise_micro_bundle["status"],
            expected_statuses=["pass"],
            purpose="Mandatory 1 to 3 qubit exact micro-validation for the required local-noise baseline.",
            generation_command=required_local_noise_micro_command,
            summary={
                "total_cases": required_local_noise_micro_bundle["summary"][
                    "total_cases"
                ],
                "passed_cases": required_local_noise_micro_bundle["summary"][
                    "passed_cases"
                ],
                "required_pass_rate": required_local_noise_micro_bundle["summary"][
                    "required_pass_rate"
                ],
                "required_noise_models_covered": required_local_noise_micro_bundle[
                    "summary"
                ][
                    "required_noise_models_covered"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="optional_noise_classification_bundle",
            artifact_class="optional_classification_bundle",
            mandatory=True,
            path=OPTIONAL_NOISE_CLASSIFICATION_ARTIFACT_FILENAME,
            status=optional_noise_classification_bundle["status"],
            expected_statuses=["pass"],
            purpose="Optional-baseline classification that keeps whole-register depolarizing outside the mandatory baseline.",
            generation_command=optional_noise_command,
            summary={
                "required_cases": optional_noise_classification_bundle["summary"][
                    "required_cases"
                ],
                "optional_cases": optional_noise_classification_bundle["summary"][
                    "optional_cases"
                ],
                "optional_pass_rate": optional_noise_classification_bundle["summary"][
                    "optional_pass_rate"
                ],
                "mandatory_baseline_completed": optional_noise_classification_bundle[
                    "summary"
                ][
                    "mandatory_baseline_completed"
                ],
                "support_tiers_present": optional_noise_classification_bundle[
                    "summary"
                ][
                    "support_tiers_present"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="unsupported_noise_bundle",
            artifact_class="unsupported_noise_bundle",
            mandatory=True,
            path=UNSUPPORTED_NOISE_ARTIFACT_FILENAME,
            status=unsupported_noise_bundle["status"],
            expected_statuses=["pass"],
            purpose="Structured negative evidence for deferred families and invalid noise schedule/configuration requests.",
            generation_command=unsupported_noise_command,
            summary={
                "total_cases": unsupported_noise_bundle["summary"]["total_cases"],
                "deferred_cases": unsupported_noise_bundle["summary"]["deferred_cases"],
                "unsupported_cases": unsupported_noise_bundle["summary"][
                    "unsupported_cases"
                ],
                "unsupported_status_cases": unsupported_noise_bundle["summary"][
                    "unsupported_status_cases"
                ],
                "boundary_passed_cases": unsupported_noise_bundle["summary"][
                    "boundary_passed_cases"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="required_local_noise_workflow_bundle",
            artifact_class="required_workflow_bundle",
            mandatory=True,
            path=REQUIRED_LOCAL_NOISE_WORKFLOW_BUNDLE_FILENAME,
            status=required_local_noise_workflow_bundle["status"],
            expected_statuses=["pass"],
            purpose="Required-local-noise workflow sufficiency bundle across the accepted exact regime.",
            generation_command=required_local_noise_workflow_command,
            summary={
                "required_cases": required_local_noise_workflow_bundle["summary"][
                    "required_cases"
                ],
                "required_passed_cases": required_local_noise_workflow_bundle[
                    "summary"
                ][
                    "required_passed_cases"
                ],
                "required_pass_rate": required_local_noise_workflow_bundle["summary"][
                    "required_pass_rate"
                ],
                "mandatory_baseline_completed": required_local_noise_workflow_bundle[
                    "summary"
                ][
                    "mandatory_baseline_completed"
                ],
                "unsupported_status_cases": required_local_noise_workflow_bundle[
                    "summary"
                ][
                    "unsupported_status_cases"
                ],
                "required_trace_case_name": required_local_noise_workflow_bundle[
                    "summary"
                ][
                    "required_trace_case_name"
                ],
                "required_trace_completed": required_local_noise_workflow_bundle[
                    "summary"
                ][
                    "required_trace_completed"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="required_local_noise_trace_4q",
            artifact_class="required_workflow_trace",
            mandatory=True,
            path=REQUIRED_LOCAL_NOISE_TRACE_ARTIFACT_FILENAME,
            status=required_local_noise_trace_result["status"],
            expected_statuses=["completed"],
            purpose="Bounded required-local-noise optimization trace for the anchor workflow.",
            generation_command=required_local_noise_workflow_command,
            summary={
                "case_name": required_local_noise_trace_result["case_name"],
                "support_tier": required_local_noise_trace_result["support_tier"],
                "case_purpose": required_local_noise_trace_result["case_purpose"],
                "required_validation_trace": required_local_noise_trace_result[
                    "required_validation_trace"
                ],
                "workflow_completed": required_local_noise_trace_result[
                    "workflow_completed"
                ],
                "optimizer": required_local_noise_trace_result["optimizer"],
                "parameter_count": required_local_noise_trace_result["parameter_count"],
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
        required_local_noise_workflow_bundle["summary"]["required_trace_case_name"]
        == required_local_noise_trace_result["case_name"]
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
            "generation_command": publication_bundle_command,
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
    validate_noise_support_publication_bundle(bundle, output_dir)
    return bundle


def validate_noise_support_publication_bundle(bundle, bundle_dir: Path):
    missing_fields = [field for field in BUNDLE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Noise-support publication bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    required_ids = {
        "required_local_noise_bundle",
        "required_local_noise_micro_bundle",
        "optional_noise_classification_bundle",
        "unsupported_noise_bundle",
        "required_local_noise_workflow_bundle",
        "required_local_noise_trace_4q",
    }
    artifact_ids = {artifact["artifact_id"] for artifact in bundle["artifacts"]}
    missing_ids = required_ids - artifact_ids
    if missing_ids:
        raise ValueError(
            "Noise-support publication bundle is missing required artifact IDs: {}".format(
                ", ".join(sorted(missing_ids))
            )
        )

    required_payload_fields = {
        "required_local_noise_bundle": {
            "suite_name",
            "status",
            "requirements",
            "software",
            "summary",
            "cases",
        },
        "required_local_noise_micro_bundle": {
            "suite_name",
            "status",
            "requirements",
            "thresholds",
            "software",
            "summary",
            "cases",
        },
        "optional_noise_classification_bundle": {
            "suite_name",
            "status",
            "requirements",
            "thresholds",
            "software",
            "summary",
            "cases",
            "required_artifacts",
        },
        "unsupported_noise_bundle": {
            "suite_name",
            "status",
            "requirements",
            "software",
            "summary",
            "cases",
        },
        "required_local_noise_workflow_bundle": {
            "suite_name",
            "status",
            "thresholds",
            "software",
            "summary",
            "cases",
        },
        "required_local_noise_trace_4q": {
            "case_name",
            "status",
            "support_tier",
            "case_purpose",
            "required_validation_trace",
            "workflow_completed",
            "optimizer",
            "parameter_count",
        },
    }
    required_summary_fields = {
        "required_local_noise_bundle": {
            "total_cases",
            "passed_cases",
            "required_pass_rate",
        },
        "required_local_noise_micro_bundle": {
            "total_cases",
            "passed_cases",
            "required_pass_rate",
            "required_noise_models_covered",
        },
        "optional_noise_classification_bundle": {
            "required_cases",
            "optional_cases",
            "optional_pass_rate",
            "mandatory_baseline_completed",
        },
        "unsupported_noise_bundle": {
            "deferred_cases",
            "unsupported_cases",
            "unsupported_status_cases",
            "boundary_passed_cases",
        },
        "required_local_noise_workflow_bundle": {
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
                "Noise-support publication bundle is missing artifact file: {}".format(
                    artifact["path"]
                )
            )
        if artifact["status"] not in artifact["expected_statuses"]:
            raise ValueError(
                "Noise-support publication artifact {} has unexpected status {}".format(
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
        "required_local_noise_workflow_bundle"
    ]
    trace_payload = loaded_payloads["required_local_noise_trace_4q"]
    if workflow_payload["summary"]["required_trace_case_name"] != trace_payload["case_name"]:
        raise ValueError(
            "Required-local-noise workflow bundle trace reference does not match the trace artifact"
        )


def write_noise_support_publication_bundle(output_path: Path, bundle):
    validate_noise_support_publication_bundle(bundle, output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_noise_support_publication_bundle(
    output_dir: Path,
    *,
    qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS,
    parameter_set_count: int = EXACT_REGIME_PARAMETER_SET_COUNT,
    verbose=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_local_noise_bundle, required_local_noise_micro_bundle, optional_results = (
        run_optional_noise_classification_validation(verbose=verbose)
    )
    optional_noise_classification_bundle = build_optional_noise_classification_bundle(
        required_local_noise_bundle,
        required_local_noise_micro_bundle,
        optional_results,
    )
    write_required_local_noise_bundle_file(
        output_dir / REQUIRED_LOCAL_NOISE_ARTIFACT_FILENAME,
        required_local_noise_bundle,
    )
    write_required_local_noise_micro_bundle_file(
        output_dir / REQUIRED_LOCAL_NOISE_MICRO_ARTIFACT_FILENAME,
        required_local_noise_micro_bundle,
    )
    write_optional_noise_classification_bundle_file(
        output_dir / OPTIONAL_NOISE_CLASSIFICATION_ARTIFACT_FILENAME,
        optional_noise_classification_bundle,
    )

    unsupported_noise_results = run_unsupported_noise_validation(verbose=verbose)
    unsupported_noise_bundle = build_unsupported_noise_bundle(
        unsupported_noise_results
    )
    write_unsupported_noise_bundle_file(
        output_dir / UNSUPPORTED_NOISE_ARTIFACT_FILENAME,
        unsupported_noise_bundle,
    )

    _, required_local_noise_trace_result, required_local_noise_workflow_bundle = (
        run_required_local_noise_workflow_validation(
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
        verbose=verbose,
        )
    )
    write_required_local_noise_workflow_bundle_file(
        output_dir / REQUIRED_LOCAL_NOISE_WORKFLOW_BUNDLE_FILENAME,
        required_local_noise_workflow_bundle,
        trace_result=required_local_noise_trace_result,
    )
    write_json(
        output_dir / REQUIRED_LOCAL_NOISE_TRACE_ARTIFACT_FILENAME,
        required_local_noise_trace_result,
    )

    bundle = build_noise_support_publication_bundle(
        output_dir,
        required_local_noise_bundle=required_local_noise_bundle,
        required_local_noise_micro_bundle=required_local_noise_micro_bundle,
        optional_noise_classification_bundle=optional_noise_classification_bundle,
        unsupported_noise_bundle=unsupported_noise_bundle,
        required_local_noise_workflow_bundle=required_local_noise_workflow_bundle,
        required_local_noise_trace_result=required_local_noise_trace_result,
    )
    write_noise_support_publication_bundle(output_dir / ARTIFACT_FILENAME, bundle)
    return bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the noise-support publication JSON artifacts.",
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
        help="Workflow qubit sizes to include in the prerequisite workflow bundle.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output from prerequisite bundle generators.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = generate_noise_support_publication_bundle(
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
