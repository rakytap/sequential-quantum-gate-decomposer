"""Shared bundle assembly for correctness-evidence validation slices."""

from __future__ import annotations

from typing import Any

from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION,
    build_correctness_evidence_selected_candidate,
    build_correctness_evidence_software_metadata,
)


def assemble_positive_case_bundle(
    suite_name: str,
    status: str,
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "suite_name": suite_name,
        "status": status,
        "record_schema_version": CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION,
        "software": build_correctness_evidence_software_metadata(),
        "selected_candidate": build_correctness_evidence_selected_candidate(),
        "summary": summary,
        "cases": cases,
    }


def assemble_negative_boundary_bundle(
    suite_name: str,
    status: str,
    negative_record_schema_version: str,
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "suite_name": suite_name,
        "status": status,
        "negative_record_schema_version": negative_record_schema_version,
        "software": build_correctness_evidence_software_metadata(),
        "selected_candidate": build_correctness_evidence_selected_candidate(),
        "summary": summary,
        "cases": cases,
    }
