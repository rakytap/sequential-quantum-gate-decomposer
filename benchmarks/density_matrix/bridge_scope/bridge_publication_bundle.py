#!/usr/bin/env python3
"""Publication bundle: bridge evidence.

Assembles the delivered bridge evidence into a single publication-oriented
manifest plus completeness checks.

Run with:
    python benchmarks/density_matrix/bridge_scope/bridge_publication_bundle.py --output-dir <dir>
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
    FIXED_PARAMETER_QUBITS,
    PRIMARY_BACKEND,
    REFERENCE_BACKEND,
    OPTIMIZATION_TRACE_ARTIFACT_FILENAME,
    EXACT_REGIME_WORKFLOW_BUNDLE_FILENAME,
    build_software_metadata,
    build_exact_regime_workflow_bundle,
    capture_case,
    get_git_revision,
    run_fixed_parameter_case,
    run_optimization_trace,
    run_exact_regime_workflow_matrix,
    write_json,
    write_exact_regime_workflow_bundle,
)
from benchmarks.density_matrix.bridge_scope.bridge_validation import (
    ARTIFACT_FILENAME as BRIDGE_MICRO_VALIDATION_BUNDLE_FILENAME,
    build_artifact_bundle as build_bridge_micro_validation_bundle,
    run_validation as run_bridge_micro_validation,
    write_artifact_bundle as write_bridge_micro_validation_bundle,
)
from benchmarks.density_matrix.bridge_scope.unsupported_bridge_validation import (
    ARTIFACT_FILENAME as UNSUPPORTED_BRIDGE_BUNDLE_FILENAME,
    build_artifact_bundle as build_unsupported_bridge_bundle,
    run_validation as run_unsupported_bridge_validation,
    write_artifact_bundle as write_unsupported_bridge_bundle,
)

BRIDGE_PUBLICATION_BUNDLE_FILENAME = "bridge_publication_bundle.json"
BRIDGE_PUBLICATION_BUNDLE_FIELDS = (
    "suite_name",
    "status",
    "backend",
    "reference_backend",
    "software",
    "provenance",
    "summary",
    "artifacts",
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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
    summary=None,
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
        "summary": {} if summary is None else dict(summary),
    }


def build_bridge_publication_bundle(
    output_dir: Path,
    *,
    fixed_results,
    trace_result,
    bridge_micro_validation_bundle,
    unsupported_bridge_bundle,
    workflow_bundle,
):
    output_dir = Path(output_dir)
    supported_fixed_parameter_command = (
        f"python benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py "
        f"--output-dir {output_dir}"
    )
    bridge_micro_validation_command = (
        f"python benchmarks/density_matrix/bridge_scope/bridge_validation.py "
        f"--output-dir {output_dir}"
    )
    unsupported_bridge_command = (
        f"python benchmarks/density_matrix/bridge_scope/unsupported_bridge_validation.py "
        f"--output-dir {output_dir}"
    )
    workflow_command = (
        f"python benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py "
        f"--workflow-bundle --output-dir {output_dir}"
    )
    publication_bundle_command = (
        f"python benchmarks/density_matrix/bridge_scope/bridge_publication_bundle.py "
        f"--output-dir {output_dir}"
    )

    artifacts = [
        _build_artifact_entry(
            artifact_id="bridge_fixed_parameters_4q",
            artifact_class="supported_positive_bridge_case",
            mandatory=True,
            path="fixed_parameters_4q.json",
            status=fixed_results[0]["status"],
            expected_statuses=["completed"],
            purpose="Supported positive 4-qubit bridge artifact for the generated-HEA anchor workflow.",
            generation_command=supported_fixed_parameter_command,
            summary={
                "qbit_num": fixed_results[0]["qbit_num"],
                "absolute_energy_error": fixed_results[0].get("absolute_energy_error"),
                "bridge_source_type": fixed_results[0].get("bridge_source_type"),
                "bridge_operation_count": fixed_results[0].get("bridge_operation_count"),
                "bridge_noise_count": fixed_results[0].get("bridge_noise_count"),
            },
        ),
        _build_artifact_entry(
            artifact_id="bridge_fixed_parameters_6q",
            artifact_class="supported_positive_bridge_case",
            mandatory=True,
            path="fixed_parameters_6q.json",
            status=fixed_results[1]["status"],
            expected_statuses=["completed"],
            purpose="Supported positive 6-qubit bridge artifact for the generated-HEA anchor workflow.",
            generation_command=supported_fixed_parameter_command,
            summary={
                "qbit_num": fixed_results[1]["qbit_num"],
                "absolute_energy_error": fixed_results[1].get("absolute_energy_error"),
                "bridge_source_type": fixed_results[1].get("bridge_source_type"),
                "bridge_operation_count": fixed_results[1].get("bridge_operation_count"),
                "bridge_noise_count": fixed_results[1].get("bridge_noise_count"),
            },
        ),
        _build_artifact_entry(
            artifact_id="bridge_micro_validation_bundle",
            artifact_class="bridge_micro_validation_bundle",
            mandatory=True,
            path=BRIDGE_MICRO_VALIDATION_BUNDLE_FILENAME,
            status=bridge_micro_validation_bundle["status"],
            expected_statuses=["pass"],
            purpose="Canonical local bridge-support validation bundle for 1 to 3 qubit generated-HEA microcases.",
            generation_command=bridge_micro_validation_command,
            summary={
                "total_cases": bridge_micro_validation_bundle["summary"]["total_cases"],
                "passed_cases": bridge_micro_validation_bundle["summary"][
                    "passed_cases"
                ],
                "pass_rate": bridge_micro_validation_bundle["summary"]["pass_rate"],
                "microcase_qubits": bridge_micro_validation_bundle["requirements"][
                    "microcase_qubits"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="bridge_exact_regime_workflow_bundle",
            artifact_class="workflow_exact_regime_bundle",
            mandatory=True,
            path=EXACT_REGIME_WORKFLOW_BUNDLE_FILENAME,
            status=workflow_bundle["status"],
            expected_statuses=["pass"],
            purpose="Canonical workflow-scale bridge bundle across the mandatory 4/6/8/10 exact regime.",
            generation_command=workflow_command,
            summary={
                "total_cases": workflow_bundle["summary"]["total_cases"],
                "passed_cases": workflow_bundle["summary"]["passed_cases"],
                "pass_rate": workflow_bundle["summary"]["pass_rate"],
                "unsupported_cases": workflow_bundle["summary"]["unsupported_cases"],
                "bridge_supported_cases": workflow_bundle["summary"]["bridge_supported_cases"],
                "documented_10q_anchor_present": workflow_bundle["summary"][
                    "documented_10q_anchor_present"
                ],
                "supported_trace_completed": workflow_bundle["summary"][
                    "supported_trace_completed"
                ],
                "supported_trace_case_name": workflow_bundle["summary"][
                    "supported_trace_case_name"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="bridge_optimization_trace_4q",
            artifact_class="supported_optimization_trace",
            mandatory=True,
            path=OPTIMIZATION_TRACE_ARTIFACT_FILENAME,
            status=trace_result["status"],
            expected_statuses=["completed"],
            purpose="Supported bounded 4-qubit optimization trace proving the full workflow crosses the bridge.",
            generation_command=workflow_command,
            summary={
                "case_name": trace_result.get("case_name"),
                "optimizer": trace_result.get("optimizer"),
                "parameter_count": trace_result.get("parameter_count"),
                "workflow_completed": trace_result.get("workflow_completed"),
                "bridge_supported_pass": trace_result.get("bridge_supported_pass"),
                "initial_energy": trace_result.get("initial_energy"),
                "final_energy": trace_result.get("final_energy"),
            },
        ),
        _build_artifact_entry(
            artifact_id="unsupported_bridge_bundle",
            artifact_class="unsupported_bridge_bundle",
            mandatory=True,
            path=UNSUPPORTED_BRIDGE_BUNDLE_FILENAME,
            status=unsupported_bridge_bundle["status"],
            expected_statuses=["pass"],
            purpose="Canonical representative unsupported-bridge bundle covering source, lowering, insertion, and noise-type failures.",
            generation_command=unsupported_bridge_command,
            summary={
                "required_categories": unsupported_bridge_bundle["requirements"][
                    "required_categories"
                ],
                "total_cases": unsupported_bridge_bundle["summary"]["total_cases"],
                "unsupported_cases": unsupported_bridge_bundle["summary"][
                    "unsupported_cases"
                ],
                "error_match_count": unsupported_bridge_bundle["summary"][
                    "error_match_count"
                ],
                "required_case_count": unsupported_bridge_bundle["summary"][
                    "required_case_count"
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
        "suite_name": "bridge_publication_evidence",
        "status": bundle_status,
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
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
        },
        "artifacts": artifacts,
    }
    validate_bridge_publication_bundle(bundle, output_dir)
    return bundle


def validate_bridge_publication_bundle(bundle, bundle_dir: Path):
    missing_fields = [
        field for field in BRIDGE_PUBLICATION_BUNDLE_FIELDS if field not in bundle
    ]
    if missing_fields:
        raise ValueError(
            "Bridge-publication bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    artifact_ids = {artifact["artifact_id"] for artifact in bundle["artifacts"]}
    required_ids = {
        "bridge_fixed_parameters_4q",
        "bridge_fixed_parameters_6q",
        "bridge_micro_validation_bundle",
        "bridge_exact_regime_workflow_bundle",
        "bridge_optimization_trace_4q",
        "unsupported_bridge_bundle",
    }
    if required_ids - artifact_ids:
        raise ValueError(
            "Bridge-publication bundle is missing required artifact IDs: {}".format(
                ", ".join(sorted(required_ids - artifact_ids))
            )
        )

    required_summary_keys = {
        "bridge_micro_validation_bundle": {
            "total_cases",
            "passed_cases",
            "pass_rate",
            "microcase_qubits",
        },
        "bridge_exact_regime_workflow_bundle": {
            "total_cases",
            "passed_cases",
            "pass_rate",
            "unsupported_cases",
            "bridge_supported_cases",
            "documented_10q_anchor_present",
            "supported_trace_completed",
            "supported_trace_case_name",
        },
        "bridge_optimization_trace_4q": {
            "case_name",
            "optimizer",
            "parameter_count",
            "workflow_completed",
            "bridge_supported_pass",
        },
        "unsupported_bridge_bundle": {
            "required_categories",
            "total_cases",
            "unsupported_cases",
            "error_match_count",
            "required_case_count",
        },
    }

    for artifact in bundle["artifacts"]:
        artifact_path = bundle_dir / artifact["path"]
        if artifact["mandatory"] and not artifact_path.exists():
            raise ValueError(
                f"Bridge-publication bundle is missing artifact file: {artifact['path']}"
            )
        if artifact["status"] not in artifact["expected_statuses"]:
            raise ValueError(
                "Bridge-publication artifact {} has unexpected status {}".format(
                    artifact["artifact_id"], artifact["status"]
                )
            )
        expected_summary_keys = required_summary_keys.get(artifact["artifact_id"])
        if expected_summary_keys and expected_summary_keys - set(artifact["summary"].keys()):
            raise ValueError(
                "Bridge-publication artifact {} is missing summary keys: {}".format(
                    artifact["artifact_id"],
                    ", ".join(sorted(expected_summary_keys - set(artifact["summary"].keys()))),
                )
            )


def write_bridge_publication_bundle(output_path: Path, bundle):
    validate_bridge_publication_bundle(bundle, output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")


def print_bridge_publication_bundle_summary(bundle):
    print("\n" + "=" * 78)
    print(
        "  Bridge Publication Bundle [{} vs {}]".format(
            bundle["backend"], bundle["reference_backend"]
        )
    )
    print("=" * 78)
    print(
        "  Mandatory artifacts present: {}/{}".format(
            bundle["summary"]["present_artifact_count"],
            bundle["summary"]["mandatory_artifact_count"],
        )
    )
    print(
        "  Status matches: {}/{}".format(
            bundle["summary"]["status_match_count"],
            bundle["summary"]["mandatory_artifact_count"],
        )
    )
    print("  Git revision:", bundle["provenance"]["git_revision"])


def generate_bridge_publication_bundle(output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_paths = [
        output_dir / "fixed_parameters_4q.json",
        output_dir / "fixed_parameters_6q.json",
    ]
    if all(path.exists() for path in fixed_paths):
        fixed_results = [_load_json(path) for path in fixed_paths]
    else:
        fixed_results = [
            capture_case(
                f"fixed_parameters_{qbit_num}q",
                lambda qbit_num=qbit_num: run_fixed_parameter_case(qbit_num),
            )
            for qbit_num in FIXED_PARAMETER_QUBITS
        ]
        for result in fixed_results:
            write_json(output_dir / f"{result['case_name']}.json", result)

    trace_path = output_dir / OPTIMIZATION_TRACE_ARTIFACT_FILENAME
    if trace_path.exists():
        trace_result = _load_json(trace_path)
    else:
        trace_result = capture_case("optimization_trace_4q", run_optimization_trace)
        write_json(trace_path, trace_result)

    bridge_micro_validation_path = output_dir / BRIDGE_MICRO_VALIDATION_BUNDLE_FILENAME
    if bridge_micro_validation_path.exists():
        bridge_micro_validation_bundle = _load_json(bridge_micro_validation_path)
    else:
        bridge_micro_validation_results = run_bridge_micro_validation(verbose=False)
        bridge_micro_validation_bundle = build_bridge_micro_validation_bundle(
            bridge_micro_validation_results
        )
        write_bridge_micro_validation_bundle(
            bridge_micro_validation_path, bridge_micro_validation_bundle
        )

    unsupported_bridge_path = output_dir / UNSUPPORTED_BRIDGE_BUNDLE_FILENAME
    if unsupported_bridge_path.exists():
        unsupported_bridge_bundle = _load_json(unsupported_bridge_path)
    else:
        unsupported_bridge_results = run_unsupported_bridge_validation(verbose=False)
        unsupported_bridge_bundle = build_unsupported_bridge_bundle(
            unsupported_bridge_results
        )
        write_unsupported_bridge_bundle(
            unsupported_bridge_path, unsupported_bridge_bundle
        )

    workflow_path = output_dir / EXACT_REGIME_WORKFLOW_BUNDLE_FILENAME
    if workflow_path.exists():
        workflow_bundle = _load_json(workflow_path)
    else:
        workflow_results = run_exact_regime_workflow_matrix()
        workflow_bundle = build_exact_regime_workflow_bundle(
            workflow_results, trace_result=trace_result
        )
        write_exact_regime_workflow_bundle(workflow_path, workflow_bundle)

    bundle = build_bridge_publication_bundle(
        output_dir,
        fixed_results=fixed_results,
        trace_result=trace_result,
        bridge_micro_validation_bundle=bridge_micro_validation_bundle,
        unsupported_bridge_bundle=unsupported_bridge_bundle,
        workflow_bundle=workflow_bundle,
    )
    write_bridge_publication_bundle(output_dir / BRIDGE_PUBLICATION_BUNDLE_FILENAME, bundle)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where bridge publication artifacts will be written.",
    )
    args = parser.parse_args()

    bundle = generate_bridge_publication_bundle(args.output_dir)
    print_bridge_publication_bundle_summary(bundle)

    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
