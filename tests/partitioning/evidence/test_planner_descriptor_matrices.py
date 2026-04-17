"""Benchmark validation matrices for planner descriptor / unsupported-case evidence."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_surface.unsupported_descriptor_validation import (
    build_unsupported_descriptor_cases,
)


def test_unsupported_descriptor_matrix_reports_categories_without_fallback():
    cases = build_unsupported_descriptor_cases()
    expected_categories = {
        "partition_qubits_too_small": "partition_span",
        "dropped_operation": "dropped_operations",
        "hidden_noise_placement": "hidden_noise_placement",
        "incomplete_remapping": "incomplete_remapping",
        "ambiguous_parameter_routing": "ambiguous_parameter_routing",
        "reordering_across_noise_boundaries": "reordering_across_noise_boundaries",
    }

    assert {case["case_name"] for case in cases} == set(expected_categories)
    for case in cases:
        assert case["status"] == "unsupported"
        assert case["unsupported_category"] == expected_categories[case["case_name"]]
        assert case["supported_descriptor_case_recorded"] is False
