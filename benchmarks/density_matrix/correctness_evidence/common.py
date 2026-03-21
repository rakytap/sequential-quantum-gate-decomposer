from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.partitioned_runtime.common import (  # noqa: E402
    PHASE3_RUNTIME_DENSITY_TOL,
    PHASE3_RUNTIME_ENERGY_TOL,
)
from benchmarks.density_matrix.planner_calibration.claim_selection import (  # noqa: E402
    build_claim_selection_payload,
)
from benchmarks.density_matrix.planner_calibration.common import (  # noqa: E402
    PLANNER_CANDIDATE_SCHEMA_VERSION,
    build_planner_candidates,
)
from benchmarks.density_matrix.planner_surface.common import (  # noqa: E402
    build_software_metadata,
)

CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION = "phase3_correctness_evidence_record_v1"
CORRECTNESS_EVIDENCE_NEGATIVE_RECORD_SCHEMA_VERSION = "phase3_correctness_evidence_negative_record_v1"
CORRECTNESS_PACKAGE_SCHEMA_VERSION = "phase3_correctness_evidence_package_v1"
CORRECTNESS_EVIDENCE_SUMMARY_SCHEMA_VERSION = "phase3_correctness_evidence_summary_v1"

CORRECTNESS_EVIDENCE_VALIDATION_SLICE_INTERNAL_ONLY = "internal_only"
CORRECTNESS_EVIDENCE_VALIDATION_SLICE_INTERNAL_PLUS_EXTERNAL = "internal_plus_external"
CORRECTNESS_EVIDENCE_REFERENCE_BACKEND_INTERNAL = "sequential_density_descriptor_reference"
CORRECTNESS_EVIDENCE_REFERENCE_BACKEND_EXTERNAL = "qiskit_aer_density_matrix"
CORRECTNESS_EVIDENCE_RUNTIME_CLASS_BASELINE = "plain_partitioned_baseline"

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "correctness_evidence"
)
PLANNER_CALIBRATION_CLAIM_SELECTION_BUNDLE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_calibration"
    / "claim_selection"
    / "claim_selection_bundle.json"
)


def correctness_evidence_output_dir(slice_dir_name: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / slice_dir_name


def write_artifact_bundle(
    bundle: dict[str, Any], output_dir: Path, artifact_filename: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / artifact_filename
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path


@lru_cache(maxsize=1)
def build_selected_candidate() -> dict[str, Any]:
    if PLANNER_CALIBRATION_CLAIM_SELECTION_BUNDLE_PATH.exists():
        claim_payload = json.loads(
            PLANNER_CALIBRATION_CLAIM_SELECTION_BUNDLE_PATH.read_text(encoding="utf-8")
        )
    else:
        claim_payload = build_claim_selection_payload()
    selected = dict(claim_payload["selected_candidate"])
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in build_planner_candidates()
    }
    candidate = candidates_by_id[selected["candidate_id"]]
    selected["candidate_schema_version"] = PLANNER_CANDIDATE_SCHEMA_VERSION
    selected["planner_settings"] = candidate.planner_settings
    selected["claim_selection_schema_version"] = claim_payload.get(
        "schema_version", claim_payload.get("claim_selection_schema_version")
    )
    selected["claim_selection_rule"] = claim_payload.get(
        "claim_selection_rule",
        claim_payload.get("summary", {}).get("claim_selection_rule"),
    )
    selected["selected_candidate_id"] = selected["candidate_id"]
    return selected


def build_package_software_metadata() -> dict[str, Any]:
    metadata = build_software_metadata()
    metadata["planner_calibration_selected_candidate_id"] = build_selected_candidate()["candidate_id"]
    return metadata


# Compatibility aliases for existing semantic imports.
CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_SCHEMA_VERSION = CORRECTNESS_PACKAGE_SCHEMA_VERSION


def build_correctness_evidence_selected_candidate() -> dict[str, Any]:
    return build_selected_candidate()


def build_correctness_evidence_software_metadata() -> dict[str, Any]:
    return build_package_software_metadata()


def base_case_name(workload_id: str) -> str:
    return workload_id


def build_validation_slice(*, external_reference_required: bool) -> str:
    return (
        CORRECTNESS_EVIDENCE_VALIDATION_SLICE_INTERNAL_PLUS_EXTERNAL
        if external_reference_required
        else CORRECTNESS_EVIDENCE_VALIDATION_SLICE_INTERNAL_ONLY
    )
