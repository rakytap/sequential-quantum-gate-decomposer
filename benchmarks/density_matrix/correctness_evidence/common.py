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
    build_task5_claim_selection_payload,
)
from benchmarks.density_matrix.planner_calibration.common import (  # noqa: E402
    TASK5_CANDIDATE_SCHEMA_VERSION,
    build_task5_planner_candidates,
)
from benchmarks.density_matrix.planner_surface.common import (  # noqa: E402
    build_software_metadata,
)

TASK6_CASE_SCHEMA_VERSION = "phase3_task6_case_record_v1"
TASK6_NEGATIVE_RECORD_SCHEMA_VERSION = "phase3_task6_negative_record_v1"
TASK6_CORRECTNESS_PACKAGE_SCHEMA_VERSION = "phase3_task6_correctness_package_v1"
TASK6_SUMMARY_SCHEMA_VERSION = "phase3_task6_summary_consistency_v1"

TASK6_VALIDATION_SLICE_INTERNAL_ONLY = "internal_only"
TASK6_VALIDATION_SLICE_INTERNAL_PLUS_EXTERNAL = "internal_plus_external"
TASK6_REFERENCE_BACKEND_INTERNAL = "sequential_density_descriptor_reference"
TASK6_REFERENCE_BACKEND_EXTERNAL = "qiskit_aer_density_matrix"
TASK6_RUNTIME_CLASS_BASELINE = "plain_partitioned_baseline"

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "phase3_task6"
)
TASK5_CLAIM_SELECTION_BUNDLE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "phase3_task5"
    / "story5_claim_selection"
    / "claim_selection_bundle.json"
)


def task6_story_output_dir(story_dir_name: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / story_dir_name


def write_artifact_bundle(
    bundle: dict[str, Any], output_dir: Path, artifact_filename: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / artifact_filename
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path


@lru_cache(maxsize=1)
def build_task6_selected_candidate() -> dict[str, Any]:
    if TASK5_CLAIM_SELECTION_BUNDLE_PATH.exists():
        claim_payload = json.loads(
            TASK5_CLAIM_SELECTION_BUNDLE_PATH.read_text(encoding="utf-8")
        )
    else:
        claim_payload = build_task5_claim_selection_payload()
    selected = dict(claim_payload["selected_candidate"])
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in build_task5_planner_candidates()
    }
    candidate = candidates_by_id[selected["candidate_id"]]
    selected["candidate_schema_version"] = TASK5_CANDIDATE_SCHEMA_VERSION
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


def build_task6_software_metadata() -> dict[str, Any]:
    metadata = build_software_metadata()
    metadata["task5_selected_candidate_id"] = build_task6_selected_candidate()[
        "candidate_id"
    ]
    return metadata


def base_case_name(workload_id: str) -> str:
    return workload_id


def build_validation_slice(*, external_reference_required: bool) -> str:
    return (
        TASK6_VALIDATION_SLICE_INTERNAL_PLUS_EXTERNAL
        if external_reference_required
        else TASK6_VALIDATION_SLICE_INTERNAL_ONLY
    )
