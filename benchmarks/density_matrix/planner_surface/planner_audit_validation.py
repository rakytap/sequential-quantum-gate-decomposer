#!/usr/bin/env python3
"""Planner audit surface validation.

Emits audit-oriented planner records for one continuity-anchor case, one
microcase, and one structured-family case. Validates that the shared canonical
planner schema carries explicit provenance and qubit-support audit metadata.

Run with:
    python benchmarks/density_matrix/planner_surface/planner_audit_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_surface.common import (
    build_phase2_continuity_vqe,
    build_software_metadata,
)
from benchmarks.density_matrix.planner_surface.workloads import (
    iter_microcase_surfaces,
    iter_structured_surfaces,
)
from squander.partitioning.noisy_planner import (
    build_bridge_overlap_report,
    build_phase3_continuity_planner_surface,
    build_planner_audit_record,
    phase3_entry_route_for_source_type,
)

SUITE_NAME = "phase3_planner_surface_planner_audit"
ARTIFACT_FILENAME = "planner_audit_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "planner_surface"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "schema_version",
    "software",
    "summary",
    "cases",
)


def build_cases() -> list[dict]:
    continuity_vqe, _, _ = build_phase2_continuity_vqe(4)
    continuity_surface = build_phase3_continuity_planner_surface(continuity_vqe)
    continuity_audit = build_planner_audit_record(
        continuity_surface, metadata={"case_kind": "continuity"}
    )
    continuity_audit.update(
        build_bridge_overlap_report(continuity_surface, continuity_vqe.describe_density_bridge())
    )
    continuity_audit["case_name"] = continuity_surface.workload_id

    micro_metadata, micro_surface = next(iter(iter_microcase_surfaces()))
    micro_audit = build_planner_audit_record(micro_surface, metadata=micro_metadata)
    micro_audit["case_name"] = micro_surface.workload_id

    structured_metadata, structured_surface = next(iter(iter_structured_surfaces()))
    structured_audit = build_planner_audit_record(
        structured_surface, metadata=structured_metadata
    )
    structured_audit["case_name"] = structured_surface.workload_id

    return [continuity_audit, micro_audit, structured_audit]


def build_artifact_bundle(cases: list[dict]) -> dict:
    passed_cases = sum(bool(case["summary"]["operation_count"] > 0) for case in cases)
    bridge_overlap_passes = sum(case.get("bridge_overlap_pass", True) for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if passed_cases == len(cases) and bridge_overlap_passes == len(cases)
        else "fail",
        "schema_version": "phase3_canonical_noisy_planner_v1",
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "bridge_overlap_passes": bridge_overlap_passes,
            "continuity_cases": sum(
                case["metadata"].get("case_kind") == "continuity" for case in cases
            ),
            "methods_cases": sum(
                case["metadata"].get("case_kind") != "continuity" for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Planner audit bundle missing required fields: {}".format(", ".join(missing))
        )
    return bundle


def write_artifact_bundle(bundle: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ARTIFACT_FILENAME
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the planner audit bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case console output.",
    )
    args = parser.parse_args(argv)

    cases = build_cases()
    bundle = build_artifact_bundle(cases)
    output_path = write_artifact_bundle(bundle, output_dir=args.output_dir)

    if not args.quiet:
        for case in cases:
            print(
                "{case_name}: ops={operation_count}, route={entry_route}".format(
                    case_name=case["case_name"],
                    operation_count=case["summary"]["operation_count"],
                    entry_route=phase3_entry_route_for_source_type(
                        case["provenance"]["source_type"]
                    ),
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
