#!/usr/bin/env python3
"""Shared helpers for Phase 2 publication claim-package validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs" / "density_matrix_project"
PHASE2_ROOT = DOCS_ROOT / "phases" / "phase-2"
PUBLICATION_CLAIM_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "publication_claim_package"
)
PHASE2_DOCUMENTATION_INDEX_PATH = PHASE2_ROOT / "PHASE_2_DOCUMENTATION_INDEX.md"
WORKFLOW_PUBLICATION_BUNDLE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "workflow_evidence"
    / "workflow_publication_bundle.json"
)
DOCUMENTATION_CONTRACT_BUNDLE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "documentation_contract"
    / "documentation_contract_bundle.json"
)

MANDATORY_PUBLICATION_EVIDENCE_DOCS = {
    "phase2_documentation_index": PHASE2_DOCUMENTATION_INDEX_PATH,
    "phase2_detailed_planning": PHASE2_ROOT / "DETAILED_PLANNING_PHASE_2.md",
    "phase2_adrs": PHASE2_ROOT / "ADRs_PHASE_2.md",
    "phase2_checklist": PHASE2_ROOT / "PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md",
    "phase2_abstract": PHASE2_ROOT / "ABSTRACT_PHASE_2.md",
    "phase2_short_paper": PHASE2_ROOT / "SHORT_PAPER_PHASE_2.md",
    "phase2_short_paper_narrative": PHASE2_ROOT / "SHORT_PAPER_NARRATIVE.md",
    "phase2_paper": PHASE2_ROOT / "PAPER_PHASE_2.md",
    "backend_selection_task_contract": PHASE2_ROOT / "task-1" / "TASK_1_MINI_SPEC.md",
    "observable_scope_task_contract": PHASE2_ROOT / "task-2" / "TASK_2_MINI_SPEC.md",
    "bridge_scope_task_contract": PHASE2_ROOT / "task-3" / "TASK_3_MINI_SPEC.md",
    "support_matrix_task_contract": PHASE2_ROOT / "task-4" / "TASK_4_MINI_SPEC.md",
    "planner_calibration_task_contract": PHASE2_ROOT / "task-5" / "TASK_5_MINI_SPEC.md",
    "correctness_evidence_task_contract": PHASE2_ROOT / "task-6" / "TASK_6_MINI_SPEC.md",
    "documentation_evidence_task_contract": PHASE2_ROOT / "task-7" / "TASK_7_MINI_SPEC.md",
    "publication_claim_task_contract": PHASE2_ROOT / "task-8" / "TASK_8_MINI_SPEC.md",
    "publication_claim_stories": PHASE2_ROOT / "task-8" / "TASK_8_STORIES.md",
    "planning_publications": DOCS_ROOT / "planning" / "PUBLICATIONS.md",
    "planning_planning": DOCS_ROOT / "planning" / "PLANNING.md",
    "research_alignment": DOCS_ROOT / "RESEARCH_ALIGNMENT.md",
    "changelog": DOCS_ROOT / "CHANGELOG.md",
    "workflow_publication_bundle": WORKFLOW_PUBLICATION_BUNDLE_PATH,
    "documentation_contract_bundle": DOCUMENTATION_CONTRACT_BUNDLE_PATH,
}

PUBLICATION_SURFACES = {
    "abstract": {
        "doc_id": "phase2_abstract",
        "label": "Phase 2 abstract surface",
    },
    "short_paper": {
        "doc_id": "phase2_short_paper",
        "label": "implementation-backed compact short-paper surface",
    },
    "short_paper_narrative": {
        "doc_id": "phase2_short_paper_narrative",
        "label": "narrative general-conference short-paper surface",
    },
    "paper": {
        "doc_id": "phase2_paper",
        "label": "implementation-backed full-paper draft surface",
    },
}

ROLE_HEADING = "## Publication Surface Role"
CLAIM_HEADING = "## Paper 1 Claim Boundary"

CANONICAL_MAIN_CLAIM = (
    "SQUANDER's exact density-matrix backend is integrated into one canonical "
    "noisy XXZ VQE workflow through explicit backend selection, exact "
    "Hermitian-energy evaluation via `Re Tr(H*rho)`, a generated-`HEA` bridge, "
    "realistic local-noise support, and a publication-grade validation package."
)
CANONICAL_SUPPORTING_CLAIMS = (
    "explicit backend selection with no silent fallback",
    "exact Hermitian-energy evaluation via `Re Tr(H*rho)`",
    "a generated-`HEA` bridge for the canonical noisy XXZ VQE workflow",
    "realistic local-noise support within the documented Phase 2 support surface",
    "publication-grade validation and reproducibility evidence",
)
CANONICAL_NON_CLAIMS = (
    "density-aware partitioning and fusion are future work for Phase 3, not current Paper 1 results",
    "density-matrix gradients and approximate scaling are future work beyond the current Paper 1 claim",
    "broad noisy-VQA workflow generality beyond the canonical supported path is not a current Paper 1 claim",
    "broad manual circuit parity or full `qgd_Circuit` parity is not a current Paper 1 claim",
    "optimizer-comparison studies and trainability analysis belong to later phases rather than to the delivered Phase 2 result",
)
EVIDENCE_CLOSURE_RULE = (
    "Only mandatory, complete, supported evidence closes the main Paper 1 claim."
)
NO_FALLBACK_RULE = (
    "No implicit `auto` mode or silent fallback is part of the Phase 2 contract."
)
SUPPORTED_PATH_BOUNDARY = (
    "The guaranteed Paper 1 path is the generated-`HEA` VQE-facing density route "
    "rather than broad standalone `NoisyCircuit` capability or full "
    "`qgd_Circuit` parity."
)
EXACT_REGIME_BOUNDARY = (
    "Full end-to-end workflow execution is required at 4 and 6 qubits, "
    "benchmark-ready fixed-parameter evaluation is required at 8 and 10 qubits, "
    "and the documented 10-qubit case is the acceptance anchor for the current "
    "exact regime."
)
PHASE_POSITIONING_RULE = (
    "Paper 1 is the Phase 2 exact noisy backend integration milestone in the "
    "density-matrix publication ladder."
)

SURFACE_ROLE_PHRASES = {
    "abstract": (
        "This document is the compact conference-abstract surface for the Phase 2 "
        "Paper 1 package."
    ),
    "short_paper": (
        "This document is the implementation-backed compact short-paper surface "
        "for the Phase 2 Paper 1 package."
    ),
    "short_paper_narrative": (
        "This document is the narrative short-paper surface for a general "
        "PhD-conference audience within the Phase 2 Paper 1 package."
    ),
    "paper": (
        "This document is the implementation-backed full-paper draft surface for "
        "the Phase 2 Paper 1 package."
    ),
}

SURFACE_REQUIRED_HEADINGS = (
    ROLE_HEADING,
    CLAIM_HEADING,
)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def missing_phrases(text: str, phrases: list[str] | tuple[str, ...]) -> list[str]:
    normalized_text = normalize_text(text)
    return [
        phrase for phrase in phrases if normalize_text(phrase) not in normalized_text
    ]


def present_phrases(text: str, phrases: list[str] | tuple[str, ...]) -> list[str]:
    normalized_text = normalize_text(text)
    return [phrase for phrase in phrases if normalize_text(phrase) in normalized_text]


def build_surface_presence(
    surface_id: str,
    *,
    required_phrases: list[str] | tuple[str, ...] = (),
    forbidden_phrases: list[str] | tuple[str, ...] = (),
    required_headings: list[str] | tuple[str, ...] = (),
) -> dict:
    surface = PUBLICATION_SURFACES[surface_id]
    path = MANDATORY_PUBLICATION_EVIDENCE_DOCS[surface["doc_id"]]
    text = load_text(path)
    missing_required_phrases = missing_phrases(text, required_phrases)
    present_forbidden_phrases = present_phrases(text, forbidden_phrases)
    missing_headings = [heading for heading in required_headings if heading not in text]
    return {
        "surface_id": surface_id,
        "label": surface["label"],
        "doc_id": surface["doc_id"],
        "path": relative_to_repo(path),
        "exists": path.exists(),
        "required_headings": list(required_headings),
        "missing_headings": missing_headings,
        "required_phrases": list(required_phrases),
        "missing_required_phrases": missing_required_phrases,
        "forbidden_phrases": list(forbidden_phrases),
        "present_forbidden_phrases": present_forbidden_phrases,
    }


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
