#!/usr/bin/env python3
"""Shared helpers for Phase 2 documentation-contract validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs" / "density_matrix_project"
PHASE2_ROOT = DOCS_ROOT / "phases" / "phase-2"
DOCUMENTATION_CONTRACT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "documentation_contract"
)
PHASE2_DOCUMENTATION_INDEX_PATH = PHASE2_ROOT / "PHASE_2_DOCUMENTATION_INDEX.md"

MANDATORY_PHASE2_DOCS = {
    "phase2_documentation_index": PHASE2_DOCUMENTATION_INDEX_PATH,
    "phase2_detailed_planning": PHASE2_ROOT / "DETAILED_PLANNING_PHASE_2.md",
    "phase2_adrs": PHASE2_ROOT / "ADRs_PHASE_2.md",
    "phase2_checklist": PHASE2_ROOT / "PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md",
    "phase2_abstract": PHASE2_ROOT / "ABSTRACT_PHASE_2.md",
    "phase2_short_paper": PHASE2_ROOT / "SHORT_PAPER_PHASE_2.md",
    "phase2_paper": PHASE2_ROOT / "PAPER_PHASE_2.md",
    "backend_selection_task_contract": PHASE2_ROOT / "task-1" / "TASK_1_MINI_SPEC.md",
    "observable_scope_task_contract": PHASE2_ROOT / "task-2" / "TASK_2_MINI_SPEC.md",
    "bridge_scope_task_contract": PHASE2_ROOT / "task-3" / "TASK_3_MINI_SPEC.md",
    "support_matrix_task_contract": PHASE2_ROOT / "task-4" / "TASK_4_MINI_SPEC.md",
    "planner_calibration_task_contract": PHASE2_ROOT / "task-5" / "TASK_5_MINI_SPEC.md",
    "correctness_evidence_task_contract": PHASE2_ROOT / "task-6" / "TASK_6_MINI_SPEC.md",
    "documentation_evidence_task_contract": PHASE2_ROOT / "task-7" / "TASK_7_MINI_SPEC.md",
    "planning_planning": DOCS_ROOT / "planning" / "PLANNING.md",
    "planning_publications": DOCS_ROOT / "planning" / "PUBLICATIONS.md",
    "research_alignment": DOCS_ROOT / "RESEARCH_ALIGNMENT.md",
    "changelog": DOCS_ROOT / "CHANGELOG.md",
    "workflow_publication_bundle": REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "workflow_evidence"
    / "workflow_publication_bundle.json",
}


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def missing_phrases(text: str, phrases: list[str] | tuple[str, ...]) -> list[str]:
    normalized_text = normalize_text(text)
    missing = []
    for phrase in phrases:
        if normalize_text(phrase) not in normalized_text:
            missing.append(phrase)
    return missing


def build_software_metadata():
    return {
        "python": sys.version.split()[0],
    }


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
