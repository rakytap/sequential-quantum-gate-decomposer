"""Benchmark validation matrices for partitioned runtime evidence bundles."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.partitioned_runtime.runtime_audit_validation import (
    build_cases as build_runtime_audit_cases,
)
from benchmarks.density_matrix.partitioned_runtime.runtime_handoff_validation import (
    build_cases as build_runtime_handoff_cases,
)
from benchmarks.density_matrix.partitioned_runtime.runtime_output_validation import (
    build_cases as build_runtime_output_cases,
)
from benchmarks.density_matrix.partitioned_runtime.unsupported_runtime_validation import (
    build_cases as build_unsupported_runtime_cases,
)
from squander.partitioning.noisy_runtime import (
    PHASE3_RUNTIME_PATH_BASELINE,
)


def test_phase3_partitioned_runtime_runtime_handoff_cases_share_shape():
    cases = build_runtime_handoff_cases()
    top_level_keys = {frozenset(case.keys()) for case in cases}
    summary_key_sets = {frozenset(case["summary"].keys()) for case in cases}
    partition_key_sets = {
        frozenset(case["partitions"][0].keys()) for case in cases if case["partitions"]
    }

    assert len(cases) == 3
    assert len(top_level_keys) == 1
    assert len(summary_key_sets) == 1
    assert len(partition_key_sets) == 1


def test_partitioned_runtime_output_cases_share_exact_output_shape():
    cases = build_runtime_output_cases()
    exact_output_key_sets = {frozenset(case["exact_output"].keys()) for case in cases}

    assert len(cases) == 3
    assert len(exact_output_key_sets) == 1
    for case in cases:
        assert case["result_output_pass"] is True
        assert case["exact_output_present"] is True
        assert "shape" in case["exact_output"]
        assert "trace_real" in case["exact_output"]
        assert "density_real" in case["exact_output"]
        assert "density_imag" in case["exact_output"]


def test_phase3_partitioned_runtime_runtime_audit_cases_share_shape():
    cases = build_runtime_audit_cases()
    top_level_keys = {frozenset(case.keys()) for case in cases}
    summary_key_sets = {frozenset(case["summary"].keys()) for case in cases}
    partition_key_sets = {
        frozenset(case["partitions"][0].keys()) for case in cases if case["partitions"]
    }

    assert len(cases) == 3
    assert len(top_level_keys) == 1
    assert len(summary_key_sets) == 1
    assert len(partition_key_sets) == 1
    assert {case["runtime_path"] for case in cases} == {PHASE3_RUNTIME_PATH_BASELINE}
    assert {case["requested_runtime_path"] for case in cases} == {PHASE3_RUNTIME_PATH_BASELINE}


def test_partitioned_runtime_unsupported_cases_have_expected_categories_and_no_fallback():
    cases = build_unsupported_runtime_cases()
    expected_categories = {
        "wrong_requested_mode": "runtime_request",
        "parameter_count_mismatch": "runtime_request",
        "unsupported_gate_name": "unsupported_runtime_operation",
        "unsupported_noise_name": "unsupported_runtime_operation",
        "gate_fixed_value": "descriptor_to_runtime_mismatch",
    }

    assert {case["case_name"] for case in cases} == set(expected_categories)
    for case in cases:
        assert case["status"] == "unsupported"
        assert case["unsupported_category"] == expected_categories[case["case_name"]]
        assert case["fallback_used"] is False
        assert case["supported_runtime_case_recorded"] is False
