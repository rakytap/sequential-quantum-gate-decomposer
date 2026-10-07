"""Negative and schema checks for interop_profile bundle validation."""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.density_matrix.interop_profile.interop_bundle_validation import (
    COUNTED_PAIRS_REQUIRED,
    INTEROP_BATCH_EXCLUSION_NOTE,
    THROUGHPUT_DIVISOR_W6_REQUIRED,
    Z_95,
    assert_mean_o_within_margin,
    spike_count_abs_wrapper_ns_above_20000,
    validate_interop_bundle,
    validate_interop_bundle_w6,
    validate_interop_implementation_paths,
)
from benchmarks.density_matrix.interop_profile.interop_lane import (
    COUNTED_REGENERATION_COMMAND_W6,
    REGENERATION_COMMAND,
)
from benchmarks.density_matrix.interop_profile.validation_pipeline import (
    resolve_interop_output_path,
)


def _minimal_valid_sample() -> dict:
    subtimes = [100, 200, 300, 4000, 500, 100]
    t_lower = subtimes[0] + subtimes[1] + subtimes[2] + subtimes[5] + subtimes[3] + subtimes[4]
    t_public = t_lower + 5000
    return {
        "t_public_ns": t_public,
        "t_lower_ns": t_lower,
        "subtimes_ns": subtimes,
    }


def _one_sided_upper_bound(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = variance**0.5
    bound = mean + Z_95 * std / (len(values) ** 0.5)
    return mean, bound


def _minimal_provenance() -> dict:
    return {
        "implementation_revision": "deadbeef",
        "clean_start": True,
        "dirty_paths": [],
        "command": "conda run -n qgd --no-capture-output python validation_pipeline.py",
        "host": "test-host",
        "cpu_model": "Test CPU",
        "compiler": {
            "executable": "c++",
            "version_line": "g++ test",
            "build_profile": "Release",
        },
        "environment": {
            "conda_default_env": "qgd",
            "conda_prefix": "/tmp/qgd",
            "python_executable": "/tmp/qgd/bin/python",
            "python_version": "3.13.0",
            "host": "test-host",
        },
        "dependencies": {
            "numpy": "2.0.0",
            "scipy": "1.14.0",
            "squander": "0.0.0",
        },
        "extension_identities": [
            {"path": "squander/libqgd.so", "sha256": "a" * 64},
            {"path": "squander/VQA/wrapper.so", "sha256": "b" * 64},
        ],
        "provenance_pass": True,
    }


def _minimal_valid_bundle() -> dict:
    sample = _minimal_valid_sample()
    samples = [copy.deepcopy(sample) for _ in range(COUNTED_PAIRS_REQUIRED)]
    o_values = [
        (row["t_public_ns"] - row["t_lower_ns"]) / row["t_public_ns"] for row in samples
    ]
    mean_o, bound_o = _one_sided_upper_bound(o_values)
    throughput_values = [row["subtimes_ns"][3] / 3072 for row in samples]
    mean_tp, bound_tp = _one_sided_upper_bound(throughput_values)
    wrapper_values = [row["t_public_ns"] - row["t_lower_ns"] for row in samples]
    allocate_values = [
        row["subtimes_ns"][0]
        + row["subtimes_ns"][1]
        + row["subtimes_ns"][2]
        + row["subtimes_ns"][5]
        for row in samples
    ]
    apply_values = [row["subtimes_ns"][3] for row in samples]
    contraction_values = [row["subtimes_ns"][4] for row in samples]

    return {
        "suite": "interop_profile_task1_evqe_4q_v1",
        "operation_count": 12,
        "qbit_num": 4,
        "warmup_pairs": 50,
        "counted_pairs": COUNTED_PAIRS_REQUIRED,
        "milestone_counted": False,
        "clean_start": True,
        "provenance": _minimal_provenance(),
        "claim_boundary": "task-1 tracer",
        "labels": "E-VQE",
        "harness_timer_flag": True,
        "protocol": {
            "pairing": "paired_not_interleaved",
            "public_first_on_even_index": True,
            "warmup_pairs": 50,
            "counted_pairs": COUNTED_PAIRS_REQUIRED,
            "parameter_rule": "linspace",
            "parameter_count": 18,
        },
        "estimator": {
            "name": "arithmetic_mean_O_i",
            "formula": "O_i",
            "one_sided_95": "mean + z*s/sqrt(n)",
            "no_sample_dropped": True,
        },
        "workload": {
            "entry": "Optimization_Problem density_matrix",
            "qbit_num": 4,
            "ansatz": "HEA",
            "hamiltonian_nnz": 40,
            "hamiltonian_csr_sha256": "c" * 64,
        },
        "qa008": {
            "categorical_labels_exact": True,
            "mean_O_absolute_margin": 0.02,
        },
        "components": {
            "mean_wrapper_ns": sum(wrapper_values) / len(wrapper_values),
            "mean_allocate_build_ns": sum(allocate_values) / len(allocate_values),
            "mean_apply_to_ns": sum(apply_values) / len(apply_values),
            "mean_contraction_ns": sum(contraction_values) / len(contraction_values),
        },
        "overhead": {"mean_O": mean_o, "upper_bound_95_O": bound_o},
        "throughput": {
            "divisor": 3072,
            "mean_ns_per_op": mean_tp,
            "upper_bound_95_ns_per_op": bound_tp,
        },
        "samples": samples,
    }


def test_validate_accepts_minimal_fixture():
    validate_interop_bundle(_minimal_valid_bundle())


def test_validate_rejects_fewer_than_1000_pairs():
    bundle = _minimal_valid_bundle()
    bundle["samples"] = bundle["samples"][:10]
    with pytest.raises(ValueError, match="counted_pairs"):
        validate_interop_bundle(bundle)


def test_validate_rejects_qa007_met_in_claim_boundary():
    bundle = _minimal_valid_bundle()
    bundle["claim_boundary"] = "QA-007 met on this row"
    with pytest.raises(ValueError, match="QA-007 met"):
        validate_interop_bundle(bundle)


def test_validate_rejects_qa007_met_anywhere_in_bundle():
    bundle = _minimal_valid_bundle()
    bundle["suite"] = "interop QA-007 met tracer"
    with pytest.raises(ValueError, match="QA-007 met"):
        validate_interop_bundle(bundle)


def test_validate_rejects_wrong_width():
    bundle = _minimal_valid_bundle()
    bundle["qbit_num"] = 6
    with pytest.raises(ValueError, match="width"):
        validate_interop_bundle(bundle)


def test_validate_rejects_broken_partition():
    bundle = _minimal_valid_bundle()
    bundle["samples"][0]["subtimes_ns"] = [1, 1, 1, 1, 1, 1]
    bundle["samples"][0]["t_lower_ns"] = 1_000_000
    with pytest.raises(ValueError, match="partition"):
        validate_interop_bundle(bundle)


def test_validate_rejects_wrong_throughput_divisor():
    bundle = _minimal_valid_bundle()
    bundle["throughput"]["divisor"] = 4096
    with pytest.raises(ValueError, match="divisor"):
        validate_interop_bundle(bundle)


def test_validate_rejects_missing_throughput_mean():
    bundle = _minimal_valid_bundle()
    del bundle["throughput"]["mean_ns_per_op"]
    with pytest.raises(ValueError, match="throughput mean"):
        validate_interop_bundle(bundle)


def test_validate_rejects_missing_throughput_bound():
    bundle = _minimal_valid_bundle()
    del bundle["throughput"]["upper_bound_95_ns_per_op"]
    with pytest.raises(ValueError, match="throughput mean"):
        validate_interop_bundle(bundle)


def test_validate_rejects_attribution_route_label():
    bundle = _minimal_valid_bundle()
    bundle["labels"] = "R-oracle overhead row"
    with pytest.raises(ValueError, match="R-oracle"):
        validate_interop_bundle(bundle)


def test_validate_rejects_batch_entry_label():
    bundle = _minimal_valid_bundle()
    bundle["labels"] = "Optimization_Problem_Batch timed"
    with pytest.raises(ValueError, match="Optimization_Problem_Batch"):
        validate_interop_bundle(bundle)


def test_validate_rejects_second_public_energy_symbol():
    bundle = _minimal_valid_bundle()
    bundle["labels"] = "timed harness_density_lower_energy"
    with pytest.raises(ValueError, match="harness_density_lower_energy"):
        validate_interop_bundle(bundle)


def test_validate_rejects_clean_start_false():
    bundle = _minimal_valid_bundle()
    bundle["clean_start"] = False
    bundle["provenance"]["clean_start"] = False
    with pytest.raises(ValueError, match="clean_start"):
        validate_interop_bundle(bundle)


def test_validate_rejects_incomplete_provenance():
    bundle = _minimal_valid_bundle()
    del bundle["provenance"]["cpu_model"]
    with pytest.raises(ValueError, match="cpu_model"):
        validate_interop_bundle(bundle)


def test_batch_exclusion_note_documents_virtual_dispatch():
    assert "virtual" in INTEROP_BATCH_EXCLUSION_NOTE.lower()
    assert "Optimization_Problem_Batch" in INTEROP_BATCH_EXCLUSION_NOTE


def test_forbidden_partitioning_path_in_change_set():
    with pytest.raises(ValueError, match="partitioning"):
        validate_interop_implementation_paths(
            ["squander/partitioning/noisy_planner.py"]
        )


def test_forbidden_path_check_accepts_allowlisted_only():
    validate_interop_implementation_paths(
        [
            "benchmarks/density_matrix/interop_profile/interop_lane.py",
            "tests/VQE/test_vqe_interop_bundle_validation.py",
        ]
    )


def test_serialized_bundle_scan_catches_qa007_met():
    bundle = _minimal_valid_bundle()
    blob = json.dumps(bundle)
    assert "QA-007 met" not in blob


def _minimal_valid_bundle_w6() -> dict:
    bundle = copy.deepcopy(_minimal_valid_bundle())
    bundle["suite"] = "interop_profile_task2_evqe_6q_v1"
    bundle["qbit_num"] = 6
    bundle["operation_count"] = 18
    bundle["claim_boundary"] = "task-2 tracer row; milestone_counted=false"
    bundle["labels"] = (
        "E-VQE density_matrix harness tracer width 6; no reduction taken on widths 4 and 6"
    )
    bundle["provenance"]["command"] = COUNTED_REGENERATION_COMMAND_W6
    bundle["workload"]["qbit_num"] = 6
    bundle["throughput"]["divisor"] = THROUGHPUT_DIVISOR_W6_REQUIRED
    samples = bundle["samples"]
    throughput_values = [row["subtimes_ns"][3] / THROUGHPUT_DIVISOR_W6_REQUIRED for row in samples]
    o_values = [
        (row["t_public_ns"] - row["t_lower_ns"]) / row["t_public_ns"] for row in samples
    ]
    mean_o, bound_o = _one_sided_upper_bound(o_values)
    mean_tp, bound_tp = _one_sided_upper_bound(throughput_values)
    bundle["overhead"] = {
        "mean_O": mean_o,
        "median_O": sorted(o_values)[len(o_values) // 2],
        "upper_bound_95_O": bound_o,
        "min_O": min(o_values),
        "max_O": max(o_values),
        "spike_count_abs_wrapper_ns_above_20000": spike_count_abs_wrapper_ns_above_20000(
            samples
        ),
    }
    bundle["throughput"]["mean_ns_per_op"] = mean_tp
    bundle["throughput"]["upper_bound_95_ns_per_op"] = bound_tp
    return bundle


def test_validate_w6_accepts_minimal_fixture():
    validate_interop_bundle_w6(_minimal_valid_bundle_w6())


def test_validate_w6_accepts_no_reduction_wording():
    bundle = _minimal_valid_bundle_w6()
    validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_width4_regeneration_command_in_provenance():
    bundle = _minimal_valid_bundle_w6()
    bundle["provenance"]["command"] = REGENERATION_COMMAND
    with pytest.raises(ValueError, match="provenance.command"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_provenance_command_missing_width_flag():
    bundle = _minimal_valid_bundle_w6()
    bundle["provenance"]["command"] = COUNTED_REGENERATION_COMMAND_W6.removesuffix(
        " --width 6"
    )
    with pytest.raises(ValueError, match="provenance.command"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_attribution_route_label():
    bundle = _minimal_valid_bundle_w6()
    bundle["labels"] = "R-base overhead row"
    with pytest.raises(ValueError, match="attribution route label"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_wrong_divisor_3072():
    bundle = _minimal_valid_bundle_w6()
    bundle["throughput"]["divisor"] = 3072
    with pytest.raises(ValueError, match="divisor"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_wrong_divisor_4096():
    bundle = _minimal_valid_bundle_w6()
    bundle["throughput"]["divisor"] = 4096
    with pytest.raises(ValueError, match="divisor"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_wrong_divisor_65536():
    bundle = _minimal_valid_bundle_w6()
    bundle["throughput"]["divisor"] = 65536
    with pytest.raises(ValueError, match="divisor"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_missing_throughput_mean():
    bundle = _minimal_valid_bundle_w6()
    del bundle["throughput"]["mean_ns_per_op"]
    with pytest.raises(ValueError, match="throughput mean"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_missing_throughput_bound():
    bundle = _minimal_valid_bundle_w6()
    del bundle["throughput"]["upper_bound_95_ns_per_op"]
    with pytest.raises(ValueError, match="throughput mean"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_spike_count_mismatch():
    bundle = _minimal_valid_bundle_w6()
    bundle["overhead"]["spike_count_abs_wrapper_ns_above_20000"] = 999
    with pytest.raises(ValueError, match="spike_count"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_a4_kill_claim():
    bundle = _minimal_valid_bundle_w6()
    bundle["labels"] = "A4 kill on this row"
    with pytest.raises(ValueError, match="A4 kill"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_hold_the_line_claim():
    bundle = _minimal_valid_bundle_w6()
    bundle["claim_boundary"] = "CAP-004 hold-the-line label"
    with pytest.raises(ValueError, match="hold-the-line"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_reduction_taken_claim():
    bundle = _minimal_valid_bundle_w6()
    bundle["labels"] = "reduction taken on width 6"
    with pytest.raises(ValueError, match="reduction taken"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_width_8():
    bundle = _minimal_valid_bundle_w6()
    bundle["qbit_num"] = 8
    with pytest.raises(ValueError, match="width"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_qa007_met_in_suite_field():
    bundle = _minimal_valid_bundle_w6()
    bundle["suite"] = "interop QA-007 met row"
    with pytest.raises(ValueError, match="QA-007 met"):
        validate_interop_bundle_w6(bundle)


def _w6_sample_with_negative_overhead_ratio() -> dict:
    sample = _minimal_valid_sample()
    t_lower = 20_000
    t_public = 10_000
    factor = t_lower / sample["t_lower_ns"]
    sub = [int(round(x * factor)) for x in sample["subtimes_ns"]]
    inner = sub[0] + sub[1] + sub[2] + sub[5] + sub[3] + sub[4]
    sub[3] += t_lower - inner
    return {"t_public_ns": t_public, "t_lower_ns": t_lower, "subtimes_ns": sub}


def test_validate_w6_accepts_negative_mean_o():
    bundle = _minimal_valid_bundle_w6()
    sample = _w6_sample_with_negative_overhead_ratio()
    bundle["samples"] = [copy.deepcopy(sample) for _ in range(COUNTED_PAIRS_REQUIRED)]
    samples = bundle["samples"]
    throughput_values = [
        row["subtimes_ns"][3] / THROUGHPUT_DIVISOR_W6_REQUIRED for row in samples
    ]
    o_values = [
        (row["t_public_ns"] - row["t_lower_ns"]) / row["t_public_ns"] for row in samples
    ]
    mean_o, bound_o = _one_sided_upper_bound(o_values)
    mean_tp, bound_tp = _one_sided_upper_bound(throughput_values)
    bundle["overhead"] = {
        "mean_O": mean_o,
        "median_O": sorted(o_values)[len(o_values) // 2],
        "upper_bound_95_O": bound_o,
        "min_O": min(o_values),
        "max_O": max(o_values),
        "spike_count_abs_wrapper_ns_above_20000": spike_count_abs_wrapper_ns_above_20000(
            samples
        ),
    }
    bundle["throughput"]["mean_ns_per_op"] = mean_tp
    bundle["throughput"]["upper_bound_95_ns_per_op"] = bound_tp
    wrapper = [row["t_public_ns"] - row["t_lower_ns"] for row in samples]
    allocate = [
        row["subtimes_ns"][0]
        + row["subtimes_ns"][1]
        + row["subtimes_ns"][2]
        + row["subtimes_ns"][5]
        for row in samples
    ]
    apply_to = [row["subtimes_ns"][3] for row in samples]
    contraction = [row["subtimes_ns"][4] for row in samples]
    bundle["components"] = {
        "mean_wrapper_ns": sum(wrapper) / len(wrapper),
        "mean_allocate_build_ns": sum(allocate) / len(allocate),
        "mean_apply_to_ns": sum(apply_to) / len(apply_to),
        "mean_contraction_ns": sum(contraction) / len(contraction),
    }
    validate_interop_bundle_w6(bundle)


def test_spike_count_pure_function():
    samples = [
        {"t_public_ns": 10_000, "t_lower_ns": 50_000, "subtimes_ns": [1, 1, 1, 1, 1, 1]},
        {"t_public_ns": 10_000, "t_lower_ns": 10_500, "subtimes_ns": [1, 1, 1, 1, 1, 1]},
    ]
    assert spike_count_abs_wrapper_ns_above_20000(samples) == 1


def test_assert_mean_o_within_margin_fixture_pair():
    assert_mean_o_within_margin(0.0, 0.0, margin=0.02)
    assert_mean_o_within_margin(0.02, 0.0, margin=0.02)
    with pytest.raises(ValueError, match="margin"):
        assert_mean_o_within_margin(0.03, 0.0, margin=0.02)


def test_resolve_output_path_refuses_width6_to_task1_name(tmp_path):
    bad = tmp_path / "interop_profile_bundle.json"
    with pytest.raises(ValueError, match="interop_profile_bundle_w6"):
        resolve_interop_output_path(6, bad)


def test_resolve_output_path_refuses_width4_to_w6_name(tmp_path):
    bad = tmp_path / "interop_profile_bundle_w6.json"
    with pytest.raises(ValueError, match="interop_profile_bundle.json"):
        resolve_interop_output_path(4, bad)


def test_resolve_output_path_accepts_width6_tmp_output(tmp_path):
    good = tmp_path / "interop_profile_bundle_w6.json"
    assert resolve_interop_output_path(6, good) == good
