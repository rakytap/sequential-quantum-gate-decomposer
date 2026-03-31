#!/usr/bin/env python3
"""Phase 3.1 counted performance matrix validation (P31-S11-E01).

Freezes the full bounded 26-row Phase 3.1 counted performance inventory and
one comparable row schema for every counted case.

Run with:
    python benchmarks/density_matrix/performance_evidence/phase31_counted_matrix_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.case_selection import (
    build_phase31_counted_performance_inventory_cases,
)
from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION,
    performance_evidence_output_dir,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_phase31_decision_summary,
    build_phase31_counted_performance_records,
)
from benchmarks.density_matrix.performance_evidence.validation_support import (
    assemble_record_schema_case_bundle,
)
from benchmarks.density_matrix.validation_scaffold import (
    require_bundle_fields,
    run_case_slice_cli,
)

SUITE_NAME = "performance_evidence_phase31_counted_matrix"
ARTIFACT_FILENAME = "phase31_counted_matrix_bundle.json"
DEFAULT_OUTPUT_DIR = performance_evidence_output_dir("phase31_counted_matrix")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "record_schema_version",
    "software",
    "selected_candidate",
    "summary",
    "cases",
)


def build_cases() -> list[dict]:
    return build_phase31_counted_performance_records()


def build_artifact_bundle(cases: list[dict]) -> dict:
    inventory = build_phase31_counted_performance_inventory_cases()
    case_ids = [case["case_name"] for case in cases]
    inventory_ids = [case["case_name"] for case in inventory]
    decision_summary = build_phase31_decision_summary(cases)
    primary_rows = sum(case["benchmark_slice"] == "phase31_structured_performance" for case in cases)
    control_rows = sum(case["benchmark_slice"] == "phase31_control_performance" for case in cases)
    route_fields_present = all(
        field in case
        for case in cases
        for field in (
            "channel_native_partition_count",
            "phase3_routed_partition_count",
            "channel_native_member_count",
            "phase3_routed_member_count",
            "hybrid_partition_route_records",
        )
    )
    baseline_trio_present = all(
        field in case
        for case in cases
        for field in (
            "sequential_median_runtime_ms",
            "phase3_fused_median_runtime_ms",
            "phase31_hybrid_median_runtime_ms",
            "sequential_median_peak_rss_kb",
            "phase3_fused_median_peak_rss_kb",
            "phase31_hybrid_median_peak_rss_kb",
        )
    )
    build_metadata_present = all(
        field in case
        for case in cases
        for field in (
            "build_policy_id",
            "build_flavor",
            "simd_enabled",
            "tbb_enabled",
            "thread_count",
            "counted_claim_build",
        )
    )
    status = (
        "pass"
        if len(cases) == 26
        and case_ids == inventory_ids
        and len(set(case_ids)) == len(case_ids)
        and primary_rows == 24
        and control_rows == 2
        and route_fields_present
        and baseline_trio_present
        and build_metadata_present
        and decision_summary["inventory_match"]
        and len(decision_summary["break_even_table"]) == 26
        and len(decision_summary["justification_map"]) == 26
        else "fail"
    )
    summary = {
        "total_cases": len(cases),
        "primary_rows": primary_rows,
        "control_rows": control_rows,
        "route_fields_present": route_fields_present,
        "baseline_trio_present": baseline_trio_present,
        "build_metadata_present": build_metadata_present,
        "inventory_match": case_ids == inventory_ids,
        "decision_rows": decision_summary["total_cases"],
        "decision_inventory_match": decision_summary["inventory_match"],
        "phase3_sufficient_rows": decision_summary["phase3_sufficient_rows"],
        "phase31_justified_rows": decision_summary["phase31_justified_rows"],
        "phase31_not_justified_yet_rows": decision_summary["phase31_not_justified_yet_rows"],
        "break_even_table_rows": len(decision_summary["break_even_table"]),
        "justification_map_rows": len(decision_summary["justification_map"]),
    }
    bundle = assemble_record_schema_case_bundle(
        SUITE_NAME,
        status,
        summary,
        cases,
    )
    bundle["record_schema_version"] = PERFORMANCE_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION
    bundle["decision_summary"] = decision_summary
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Phase 3.1 counted matrix bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_cases,
        build_artifact_bundle=build_artifact_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the Phase 3.1 counted matrix bundle into.",
        quiet_report=lambda b: print(
            "total_cases={total_cases}, primary_rows={primary_rows}, control_rows={control_rows}, status={status}".format(
                status=b["status"],
                **b["summary"],
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
