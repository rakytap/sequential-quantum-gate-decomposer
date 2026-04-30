#!/usr/bin/env python3
"""Validation: validation-evidence interpretation guardrails.

Builds the phase-level interpretation layer from:
- the already passing metric-completeness bundle,
- the optional-noise classification bundle,
- and the unsupported-noise bundle.

The resulting bundle is intentionally a thin validation-evidence layer:
- it computes the main Phase 2 validation claim only from mandatory, complete,
  supported evidence,
- it keeps optional evidence explicitly supplemental,
- it keeps unsupported or deferred evidence explicitly negative,
- and it treats missing mandatory evidence as incomplete rather than as partial
  success.

Run with:
    python benchmarks/density_matrix/validation_evidence/interpretation_validation.py
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
)
from benchmarks.density_matrix.noise_support.optional_noise_classification_validation import (
    build_artifact_bundle as build_optional_noise_bundle,
    run_validation as run_optional_noise_validation,
)
from benchmarks.density_matrix.noise_support.unsupported_noise_validation import (
    build_artifact_bundle as build_unsupported_noise_bundle,
    run_validation as run_unsupported_noise_validation,
)
from benchmarks.density_matrix.noise_support.support_tiers import SUPPORT_TIER_VOCABULARY
from benchmarks.density_matrix.validation_evidence.metric_completeness_validation import (
    PRIMARY_BACKEND,
    REFERENCE_BACKEND,
    run_validation as run_metric_completeness_validation,
)

SUITE_NAME = "validation_evidence_interpretation"
ARTIFACT_FILENAME = "interpretation_bundle.json"
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


def build_requirement_metadata():
    return {
        "support_tier_vocabulary": list(SUPPORT_TIER_VOCABULARY),
        "main_claim_rule": (
            "Only mandatory, complete, supported evidence may close the main "
            "validation-evidence claim."
        ),
        "excluded_evidence_classes": [
            "optional",
            "deferred",
            "unsupported",
            "incomplete",
        ],
        "required_bundle_sources": [
            "metric_completeness_validation",
            "optional_noise_classification",
            "unsupported_noise_boundary",
        ],
    }


def build_artifact_bundle(
    metric_completeness_bundle,
    optional_noise_bundle,
    unsupported_noise_bundle,
):
    incomplete_mandatory_artifacts = []
    if metric_completeness_bundle["status"] != "pass":
        incomplete_mandatory_artifacts.append("metric_completeness_validation")

    for artifact_id, artifact in metric_completeness_bundle["required_artifacts"].items():
        if artifact["status"] != "pass":
            incomplete_mandatory_artifacts.append(artifact_id)

    mandatory_artifacts_complete = not incomplete_mandatory_artifacts
    optional_evidence_supplemental = bool(
        optional_noise_bundle["status"] == "pass"
        and optional_noise_bundle["summary"][
            "optional_cases_count_toward_mandatory_baseline"
        ]
        == 0
    )
    unsupported_evidence_negative_only = bool(
        unsupported_noise_bundle["status"] == "pass"
        and unsupported_noise_bundle["summary"]["unsupported_status_cases"]
        == unsupported_noise_bundle["summary"]["total_cases"]
        and unsupported_noise_bundle["summary"]["mandatory_baseline_case_count"] == 0
    )
    main_validation_claim_completed = bool(
        mandatory_artifacts_complete
        and optional_evidence_supplemental
        and unsupported_evidence_negative_only
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if main_validation_claim_completed else "fail",
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "requirements": build_requirement_metadata(),
        "thresholds": dict(metric_completeness_bundle["thresholds"]),
        "software": build_software_metadata(),
        "summary": {
            "mandatory_artifacts": [
                "local_correctness_reference",
                "workflow_baseline_reference",
                "trace_anchor_reference",
                "exact_regime_metric_completeness",
            ],
            "incomplete_mandatory_artifacts": incomplete_mandatory_artifacts,
            "mandatory_artifacts_complete": mandatory_artifacts_complete,
            "optional_cases": optional_noise_bundle["summary"]["optional_cases"],
            "optional_passed_cases": optional_noise_bundle["summary"][
                "optional_passed_cases"
            ],
            "optional_cases_count_toward_mandatory_baseline": optional_noise_bundle[
                "summary"
            ]["optional_cases_count_toward_mandatory_baseline"],
            "optional_evidence_supplemental": optional_evidence_supplemental,
            "unsupported_status_cases": unsupported_noise_bundle["summary"][
                "unsupported_status_cases"
            ],
            "unsupported_cases": unsupported_noise_bundle["summary"][
                "unsupported_cases"
            ],
            "deferred_cases": unsupported_noise_bundle["summary"]["deferred_cases"],
            "unsupported_evidence_negative_only": unsupported_evidence_negative_only,
            "main_validation_claim_completed": main_validation_claim_completed,
        },
        "required_artifacts": {
            "metric_completeness_validation": {
                "suite_name": metric_completeness_bundle["suite_name"],
                "status": metric_completeness_bundle["status"],
                "summary": metric_completeness_bundle["summary"],
            },
            "optional_noise_reference": {
                "suite_name": optional_noise_bundle["suite_name"],
                "status": optional_noise_bundle["status"],
                "summary": optional_noise_bundle["summary"],
            },
            "unsupported_noise_reference": {
                "suite_name": unsupported_noise_bundle["suite_name"],
                "status": unsupported_noise_bundle["status"],
                "summary": unsupported_noise_bundle["summary"],
            },
        },
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Interpretation bundle is missing required fields: {}".format(
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
    _, _, _, metric_completeness_bundle = run_metric_completeness_validation(
        qubit_sizes=qubit_sizes,
        parameter_set_count=parameter_set_count,
        verbose=verbose,
    )
    required_local_noise_bundle, required_local_noise_micro_bundle, optional_results = (
        run_optional_noise_validation(verbose=verbose)
    )
    optional_noise_bundle = build_optional_noise_bundle(
        required_local_noise_bundle,
        required_local_noise_micro_bundle,
        optional_results,
    )
    unsupported_results = run_unsupported_noise_validation(verbose=verbose)
    unsupported_noise_bundle = build_unsupported_noise_bundle(unsupported_results)
    bundle = build_artifact_bundle(
        metric_completeness_bundle,
        optional_noise_bundle,
        unsupported_noise_bundle,
    )
    if verbose:
        print(
            "{} [{}] mandatory_complete={} optional_supplemental={} unsupported_negative_only={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["mandatory_artifacts_complete"],
                bundle["summary"]["optional_evidence_supplemental"],
                bundle["summary"]["unsupported_evidence_negative_only"],
            )
        )
    return metric_completeness_bundle, optional_noise_bundle, unsupported_noise_bundle, bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the interpretation JSON artifact bundle.",
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
