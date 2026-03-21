#!/usr/bin/env python3
"""Shared helpers for publication-evidence validation bundles."""

from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs" / "density_matrix_project"
PHASE3_ROOT = DOCS_ROOT / "phases" / "phase-3"

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "publication_evidence"
)
CORRECTNESS_EVIDENCE_ARTIFACT_ROOT = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "correctness_evidence"
)
PERFORMANCE_EVIDENCE_ARTIFACT_ROOT = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "performance_evidence"
)

CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_PATH = (
    CORRECTNESS_EVIDENCE_ARTIFACT_ROOT
    / "correctness_package"
    / "correctness_package_bundle.json"
)
CORRECTNESS_EVIDENCE_UNSUPPORTED_BOUNDARY_PATH = (
    CORRECTNESS_EVIDENCE_ARTIFACT_ROOT
    / "unsupported_boundary"
    / "unsupported_boundary_bundle.json"
)
CORRECTNESS_EVIDENCE_SUMMARY_CONSISTENCY_PATH = (
    CORRECTNESS_EVIDENCE_ARTIFACT_ROOT
    / "summary_consistency"
    / "summary_consistency_bundle.json"
)

PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_PATH = (
    PERFORMANCE_EVIDENCE_ARTIFACT_ROOT / "benchmark_package" / "benchmark_package_bundle.json"
)
PERFORMANCE_EVIDENCE_DIAGNOSIS_PATH = (
    PERFORMANCE_EVIDENCE_ARTIFACT_ROOT / "diagnosis" / "diagnosis_bundle.json"
)
PERFORMANCE_EVIDENCE_SENSITIVITY_MATRIX_PATH = (
    PERFORMANCE_EVIDENCE_ARTIFACT_ROOT / "sensitivity_matrix" / "sensitivity_matrix_bundle.json"
)
PERFORMANCE_EVIDENCE_POSITIVE_THRESHOLD_PATH = (
    PERFORMANCE_EVIDENCE_ARTIFACT_ROOT / "positive_threshold" / "positive_threshold_bundle.json"
)
PERFORMANCE_EVIDENCE_SUMMARY_CONSISTENCY_PATH = (
    PERFORMANCE_EVIDENCE_ARTIFACT_ROOT / "summary_consistency" / "summary_consistency_bundle.json"
)

MANDATORY_PUBLICATION_EVIDENCE_DOCS = {
    "phase3_detailed_planning": PHASE3_ROOT / "DETAILED_PLANNING_PHASE_3.md",
    "phase3_adrs": PHASE3_ROOT / "ADRs_PHASE_3.md",
    "phase3_checklist": PHASE3_ROOT / "PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md",
    "phase3_abstract": PHASE3_ROOT / "ABSTRACT_PHASE_3.md",
    "phase3_short_paper": PHASE3_ROOT / "SHORT_PAPER_PHASE_3.md",
    "phase3_short_paper_narrative": PHASE3_ROOT / "SHORT_PAPER_NARRATIVE.md",
    "phase3_paper": PHASE3_ROOT / "PAPER_PHASE_3.md",
    "planner_surface_entry_mini_spec": PHASE3_ROOT / "task-1" / "TASK_1_MINI_SPEC.md",
    "planner_surface_descriptor_mini_spec": PHASE3_ROOT / "task-2" / "TASK_2_MINI_SPEC.md",
    "partitioned_runtime_mini_spec": PHASE3_ROOT / "task-3" / "TASK_3_MINI_SPEC.md",
    "fused_runtime_mini_spec": PHASE3_ROOT / "task-4" / "TASK_4_MINI_SPEC.md",
    "planner_calibration_mini_spec": PHASE3_ROOT / "task-5" / "TASK_5_MINI_SPEC.md",
    "correctness_evidence_mini_spec": PHASE3_ROOT / "task-6" / "TASK_6_MINI_SPEC.md",
    "performance_evidence_mini_spec": PHASE3_ROOT / "task-7" / "TASK_7_MINI_SPEC.md",
    "publication_evidence_mini_spec": PHASE3_ROOT / "task-8" / "TASK_8_MINI_SPEC.md",
    "publication_evidence_stories": PHASE3_ROOT / "task-8" / "TASK_8_STORIES.md",
    "planning_publications": DOCS_ROOT / "planning" / "PUBLICATIONS.md",
    "planning_planning": DOCS_ROOT / "planning" / "PLANNING.md",
    "correctness_evidence_correctness_package_bundle": CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_PATH,
    "correctness_evidence_unsupported_boundary_bundle": CORRECTNESS_EVIDENCE_UNSUPPORTED_BOUNDARY_PATH,
    "correctness_evidence_summary_consistency_bundle": CORRECTNESS_EVIDENCE_SUMMARY_CONSISTENCY_PATH,
    "performance_evidence_benchmark_package_bundle": PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_PATH,
    "performance_evidence_diagnosis_bundle": PERFORMANCE_EVIDENCE_DIAGNOSIS_PATH,
    "performance_evidence_sensitivity_matrix_bundle": PERFORMANCE_EVIDENCE_SENSITIVITY_MATRIX_PATH,
    "performance_evidence_positive_threshold_bundle": PERFORMANCE_EVIDENCE_POSITIVE_THRESHOLD_PATH,
    "performance_evidence_summary_consistency_bundle": PERFORMANCE_EVIDENCE_SUMMARY_CONSISTENCY_PATH,
}

PUBLICATION_SURFACES = {
    "abstract": {
        "doc_id": "phase3_abstract",
        "label": "Phase 3 abstract surface",
    },
    "short_paper": {
        "doc_id": "phase3_short_paper",
        "label": "Phase 3 technical short-paper surface",
    },
    "short_paper_narrative": {
        "doc_id": "phase3_short_paper_narrative",
        "label": "Phase 3 narrative short-paper surface",
    },
    "paper": {
        "doc_id": "phase3_paper",
        "label": "Phase 3 full-paper surface",
    },
}

ROLE_HEADING = "## Publication Surface Role"
CLAIM_HEADING = "## Paper 2 Claim Boundary"

SURFACE_ROLE_PHRASES = {
    "abstract": (
        "This document is the compact conference-abstract surface for the Phase 3 "
        "Paper 2 package."
    ),
    "short_paper": (
        "This document is the compact short-paper surface for the Phase 3 Paper 2 "
        "package."
    ),
    "short_paper_narrative": (
        "This document is the narrative short-paper surface for a general "
        "PhD-conference audience within the Phase 3 Paper 2 package."
    ),
    "paper": (
        "This document is the planning-facing full-paper draft surface for the "
        "Phase 3 Paper 2 package."
    ),
}

PAPER_MAIN_CLAIM = (
    "SQUANDER extends partitioning and limited fusion to exact noisy mixed-state "
    "circuits by making noisy operations first-class planner inputs, preserving "
    "exact gate/noise semantics across partition descriptors and runtime "
    "execution, and validating the resulting partitioned density path on "
    "representative noisy workloads."
)
NARRATIVE_MAIN_CLAIM = (
    "SQUANDER can extend partitioning and limited fusion to exact noisy "
    "mixed-state circuits without reducing noise to planner-external metadata, "
    "while preserving exact semantics and yielding a scientifically useful "
    "benchmarked backend."
)
EVIDENCE_CLOSURE_RULE = (
    "Only mandatory, complete, supported correctness and reproducibility "
    "evidence, plus either measured benefit or benchmark-grounded limitation "
    "reporting, closes the main Paper 2 claim."
)
EVIDENCE_CLOSURE_RULE_ALTERNATIVES = (
    EVIDENCE_CLOSURE_RULE,
    "Only mandatory, complete, supported correctness and reproducibility "
    "evidence, plus either measurable benefit or benchmark-grounded limitation "
    "reporting, closes the main Paper 2 claim.",
)
SUPPORTED_PATH_BOUNDARY_ALTERNATIVES = (
    "The guaranteed Paper 2 path is the canonical noisy mixed-state planner "
    "surface plus the documented exact lowering needed for the frozen Phase 2 "
    "continuity workflow and the structured Phase 3 benchmark families.",
    "The guaranteed Paper 2 path is the canonical noisy mixed-state planner "
    "surface plus the exact lowering needed for the frozen Phase 2 continuity "
    "workflow and the required Phase 3 structured benchmark families.",
    "The guaranteed Paper 2 path is the canonical noisy mixed-state planner "
    "surface plus exact lowering for the frozen Phase 2 continuity workflow and "
    "the required Phase 3 structured benchmark families.",
    "The guaranteed Paper 2 path is the canonical noisy mixed-state planner "
    "surface plus the documented exact lowering required for the frozen Phase 2 "
    "continuity workflow and the required Phase 3 structured benchmark families.",
)
NO_FALLBACK_RULE_ALTERNATIVES = (
    "No silent sequential fallback is part of the Phase 3 contract for any "
    "benchmark that claims partitioned density behavior.",
    "No silent sequential fallback is part of the Phase 3 contract for "
    "benchmarks that claim partitioned density behavior.",
    "No silent substitution of sequential execution is part of the Phase 3 "
    "contract for a benchmark that claims `partitioned_density` behavior.",
)
PHASE_POSITIONING_RULE_ALTERNATIVES = (
    "Paper 2 is the Phase 3 methods and systems milestone in the "
    "density-matrix publication ladder.",
    "Paper 2 is the Phase 3 methods milestone between exact noisy integration "
    "and broader noisy workflow science.",
    "Paper 2 is the Phase 3 noise-aware partitioning and fusion milestone in the "
    "density-matrix publication ladder.",
)
CURRENT_DIAGNOSIS_PHRASE_ALTERNATIVES = (
    "diagnosis-grounded limitation reporting",
    "diagnosis-grounded benchmark evidence",
    "diagnosis-grounded closure",
    "diagnosis branch",
)
BOTTLENECK_PHRASE_ALTERNATIVES = (
    "supported islands left unfused",
    "supported islands still remain unfused",
    "Python-level fused-path overhead",
    "present fused path adds Python-level overhead",
    "present fused path stays slower",
    "fused path stays slower than the sequential reference",
    "peak memory does not improve",
)


def publication_evidence_output_dir(slice_dir_name: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / slice_dir_name


# Compatibility aliases for existing semantic imports.
CORRECTNESS_PACKAGE_PATH = CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_PATH


def write_artifact_bundle(
    bundle: dict[str, Any], output_dir: Path, artifact_filename: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / artifact_filename
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def phrase_present(text: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(text)


def phrase_group_present(text: str, phrases: list[str] | tuple[str, ...]) -> bool:
    return any(phrase_present(text, phrase) for phrase in phrases)


def missing_phrases(text: str, phrases: list[str] | tuple[str, ...]) -> list[str]:
    return [phrase for phrase in phrases if not phrase_present(text, phrase)]


def build_surface_presence(
    surface_id: str,
    *,
    required_headings: list[str] | tuple[str, ...] = (),
    required_phrase_groups: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    forbidden_phrase_groups: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    surface = PUBLICATION_SURFACES[surface_id]
    path = MANDATORY_PUBLICATION_EVIDENCE_DOCS[surface["doc_id"]]
    text = load_text(path)
    missing_headings = [heading for heading in required_headings if heading not in text]
    missing_required_groups = [
        {
            "group_id": group["group_id"],
            "phrases": list(group["phrases"]),
        }
        for group in required_phrase_groups
        if not phrase_group_present(text, group["phrases"])
    ]
    present_forbidden_groups = [
        {
            "group_id": group["group_id"],
            "phrases": list(group["phrases"]),
        }
        for group in forbidden_phrase_groups
        if phrase_group_present(text, group["phrases"])
    ]
    return {
        "surface_id": surface_id,
        "label": surface["label"],
        "doc_id": surface["doc_id"],
        "path": relative_to_repo(path),
        "exists": path.exists(),
        "required_headings": list(required_headings),
        "missing_headings": missing_headings,
        "required_phrase_groups": [
            {
                "group_id": group["group_id"],
                "phrases": list(group["phrases"]),
            }
            for group in required_phrase_groups
        ],
        "missing_required_phrase_groups": missing_required_groups,
        "forbidden_phrase_groups": [
            {
                "group_id": group["group_id"],
                "phrases": list(group["phrases"]),
            }
            for group in forbidden_phrase_groups
        ],
        "present_forbidden_phrase_groups": present_forbidden_groups,
    }


def build_software_metadata() -> dict[str, str]:
    return {"python": sys.version.split()[0]}


def get_git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_or_build_artifact(
    artifact_path: Path,
    *,
    run_validation,
    validate_artifact_bundle,
) -> dict[str, Any]:
    if artifact_path.exists():
        artifact = load_json(artifact_path)
        validate_artifact_bundle(artifact)
        return artifact
    return run_validation(verbose=False)


@lru_cache(maxsize=1)
def load_correctness_evidence_summary_bundle() -> dict[str, Any]:
    return load_json(CORRECTNESS_EVIDENCE_SUMMARY_CONSISTENCY_PATH)


@lru_cache(maxsize=1)
def load_performance_evidence_summary_bundle() -> dict[str, Any]:
    return load_json(PERFORMANCE_EVIDENCE_SUMMARY_CONSISTENCY_PATH)


def correctness_evidence_count_phrase_options() -> tuple[str, ...]:
    count = load_correctness_evidence_summary_bundle()["summary"]["counted_supported_cases"]
    return (f"`{count}` counted supported cases",)


def correctness_evidence_boundary_phrase_options() -> tuple[str, ...]:
    count = load_correctness_evidence_summary_bundle()["summary"]["unsupported_boundary_cases"]
    return (
        f"`{count}` explicit unsupported-boundary cases",
        f"`{count}` explicit unsupported- boundary cases",
        f"`{count}` explicit correctness-evidence boundary cases",
        f"the `{count}` explicit correctness-evidence boundary cases",
        f"`{count}` carried-forward unsupported-boundary cases",
    )


def performance_evidence_count_phrase_options() -> tuple[str, ...]:
    count = load_performance_evidence_summary_bundle()["summary"]["counted_supported_cases"]
    return (
        f"`{count}` counted supported benchmark cases",
        f"`{count}` counted supported cases",
    )


def performance_evidence_review_phrase_options() -> tuple[str, ...]:
    count = load_performance_evidence_summary_bundle()["summary"]["representative_review_cases"]
    return (
        f"`{count}` representative review cases",
        f"representative review set contains `{count}`",
        f"representative review set of `{count}`",
        f"`{count}` primary-seed sparse structured cases",
    )
