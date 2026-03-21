from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

from benchmarks.density_matrix.planner_calibration.bundle import (
    build_planner_calibration_calibration_bundle_payload,
)
from benchmarks.density_matrix.planner_calibration.claim_selection import (
    PLANNER_CALIBRATION_CLAIM_STATUS_COMPARISON,
)

PLANNER_CALIBRATION_BOUNDARY_SCHEMA_VERSION = "phase3_planner_calibration_claim_boundary_v1"


@lru_cache(maxsize=1)
def _build_planner_calibration_boundary_payload_cached() -> dict:
    bundle_payload = build_planner_calibration_calibration_bundle_payload()
    selected_candidate = bundle_payload["selected_candidate"]
    comparison_cases = [
        case
        for case in bundle_payload["cases"]
        if case["claim_status"] == PLANNER_CALIBRATION_CLAIM_STATUS_COMPARISON
    ]
    return {
        "schema_version": PLANNER_CALIBRATION_BOUNDARY_SCHEMA_VERSION,
        "supported_claim": {
            "candidate_id": selected_candidate["candidate_id"],
            "planner_family": selected_candidate["planner_family"],
            "planner_variant": selected_candidate["planner_variant"],
            "max_partition_qubits": selected_candidate["max_partition_qubits"],
            "claim_scope": "benchmark_calibrated_span_budget_candidate",
        },
        "comparison_baselines": bundle_payload["comparison_candidate_ids"],
        "comparison_baseline_cases": comparison_cases,
        "diagnosis_only_cases": [],
        "approximation_areas": [
            {
                "category": "bounded_candidate_surface",
                "description": "Planner calibration tunes the current noisy planner's span-budget settings rather than a broader family of already-implemented noisy planner variants.",
            },
            {
                "category": "bounded_workload_matrix",
                "description": "The supported claim is calibrated only on the frozen continuity, microcase, and structured Phase 3 workload inventory.",
            },
            {
                "category": "no_global_optimality_claim",
                "description": "The selected candidate is benchmark-calibrated for the frozen matrix but is not claimed as a globally optimal partitioning policy.",
            },
            {
                "category": "optional_fused_signal_scope",
                "description": "Fused-runtime opportunity may inform the signal surface where available, but the supported planner-calibration claim is still rooted in the baseline partitioned-density runtime surface.",
            },
        ],
        "deferred_follow_on_branches": [
            {
                "category": "channel_native_fused_noisy_blocks",
                "description": "Fully channel-native fused noisy blocks remain a benchmark-driven follow-on branch beyond core Phase 3.",
            },
            {
                "category": "broader_noisy_workflow_growth",
                "description": "Broader noisy VQE/VQA workflow growth and gradient-routing features remain outside the minimum planner-calibration claim.",
            },
            {
                "category": "calibration_aware_or_readout_features",
                "description": "Calibration-aware, readout-oriented, and shot-noise workflow features remain deferred beyond the core planner-calibration surface.",
            },
            {
                "category": "approximate_scaling_methods",
                "description": "Approximate scaling branches such as trajectories or MPDO-style methods remain future work rather than part of the supported planner-calibration claim.",
            },
        ],
        "summary": {
            "selected_candidate_id": selected_candidate["candidate_id"],
            "comparison_candidate_ids": bundle_payload["comparison_candidate_ids"],
            "comparison_baseline_case_count": len(comparison_cases),
            "diagnosis_only_case_count": 0,
            "approximation_area_count": 4,
            "deferred_follow_on_branch_count": 4,
        },
    }


def build_planner_calibration_boundary_payload() -> dict:
    return deepcopy(_build_planner_calibration_boundary_payload_cached())
