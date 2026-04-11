#!/usr/bin/env python3
"""Phase 3.1 hybrid performance pilot artifact (P31-S09-E01).

Materializes one frozen structured case with sequential, Phase 3 fused, and hybrid
timings, route coverage, and a small decision/diagnosis tag.

Run with:
    python benchmarks/density_matrix/performance_evidence/phase31_hybrid_pilot_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.case_selection import (
    PHASE31_HYBRID_PILOT_WORKLOAD_ID,
    build_phase31_hybrid_pilot_case_context,
)
from benchmarks.density_matrix.performance_evidence.common import (
    performance_evidence_output_dir,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_phase31_hybrid_pilot_record,
)
from benchmarks.density_matrix.performance_evidence.validation_support import (
    assemble_record_schema_case_bundle,
)
from benchmarks.density_matrix.validation_scaffold import (
    require_bundle_fields,
    run_case_slice_cli,
)

SUITE_NAME = "performance_evidence_phase31_hybrid_pilot"
ARTIFACT_FILENAME = "phase31_hybrid_pilot_bundle.json"
DEFAULT_OUTPUT_DIR = performance_evidence_output_dir("phase31_hybrid_pilot")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "record_schema_version",
    "software",
    "selected_candidate",
    "summary",
    "cases",
)


def build_phase31_hybrid_pilot_cases() -> list[dict]:
    return [build_phase31_hybrid_pilot_record(build_phase31_hybrid_pilot_case_context())]


def build_phase31_hybrid_pilot_bundle(cases: list[dict]) -> dict:
    if len(cases) != 1:
        raise ValueError("phase31 hybrid pilot bundle expects exactly one case")
    case = cases[0]
    correctness_pass = bool(
        case.get("phase3_fused_internal_reference_pass")
        and case.get("phase31_hybrid_internal_reference_pass")
    )
    summary = {
        "total_cases": len(cases),
        "pilot_case_name": case["case_name"],
        "pilot_workload_id": PHASE31_HYBRID_PILOT_WORKLOAD_ID,
        "timing_mode": case["timing_mode"],
        "decision_class": case["decision_class"],
        "diagnosis_tag": case["diagnosis_tag"],
        "channel_native_partition_count": case["channel_native_partition_count"],
        "phase3_routed_partition_count": case["phase3_routed_partition_count"],
    }
    bundle = assemble_record_schema_case_bundle(
        SUITE_NAME,
        "pass" if correctness_pass else "fail",
        summary,
        cases,
    )
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Phase 3.1 hybrid pilot bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_phase31_hybrid_pilot_cases,
        build_artifact_bundle=build_phase31_hybrid_pilot_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the Phase 3.1 hybrid pilot bundle into.",
        quiet_report=lambda b: print(
            "pilot={pilot_case_name}, decision={decision_class}, diagnosis={diagnosis_tag}, "
            "cn_partitions={channel_native_partition_count}, p3_partitions={phase3_routed_partition_count}, "
            "status={status}".format(status=b["status"], **b["summary"])
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
