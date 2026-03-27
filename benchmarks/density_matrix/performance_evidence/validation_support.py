"""Shared bundle assembly for performance-evidence validation slices."""

from __future__ import annotations

from typing import Any

from benchmarks.density_matrix.correctness_evidence.common import build_selected_candidate
from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
    build_package_software_metadata,
)


def assemble_record_schema_case_bundle(
    suite_name: str,
    status: str,
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "suite_name": suite_name,
        "status": status,
        "record_schema_version": PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
        "software": build_package_software_metadata(),
        "selected_candidate": build_selected_candidate(),
        "summary": summary,
        "cases": cases,
    }


def assemble_benchmark_matrix_bundle(
    suite_name: str,
    status: str,
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Benchmark matrix inventory bundle has no record_schema_version."""
    return {
        "suite_name": suite_name,
        "status": status,
        "selected_candidate": build_selected_candidate(),
        "software": build_package_software_metadata(),
        "summary": summary,
        "cases": cases,
    }
