"""Reporting regressions, independent of stochastic synthesis runtimes."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def report(monkeypatch):
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/partitioning/local_synthesis_head_to_head.py"
    )
    spec = importlib.util.spec_from_file_location("local_synthesis_stats_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    # Synthetic timing/count rows exercise aggregation only. Circuit
    # certification and real mismatching exports are covered separately.
    monkeypatch.setattr(module, "matrix_from_qasm", lambda _: np.eye(8))
    return module


def case(identifier):
    return {
        "case_id": identifier, "width": 3, "context": "a2a",
        "dataset": "synthetic", "source_circuit": identifier,
    }


def verified(identifier, method, seconds, wall_seconds=None, cnots=2):
    result = {
        "case_id": identifier, "method": method, "status": "verified",
        "cnots": cnots, "seconds": seconds, "output_qasm": "certificate",
        "topology_verified": True,
    }
    if wall_seconds is not None:
        result["wall_seconds"] = wall_seconds
    return result


def sample_summary(report, missing_wall=False):
    cases = [case(identifier) for identifier in ("a", "b", "osr_only", "bq_only")]
    rows = [
        verified("a", "osr", 2, 3),
        verified("a", "bqskit", 8, None if missing_wall else 12),
        verified("b", "osr", 10, 12),
        verified("b", "bqskit", 10, 13),
        verified("osr_only", "osr", 1000, 1001),
        {"case_id": "osr_only", "method": "bqskit", "status": "timeout"},
        verified("bq_only", "bqskit", 500, 501),
        {"case_id": "bq_only", "method": "osr", "status": "no_solution"},
    ]
    return report.summarize(rows, cases, seed=20260926, tolerance=1e-10)["3q/a2a"]


def test_cluster_bootstrap_matches_partition_weighted_mean(report):
    clusters = {"large": [-2] * 100}
    clusters.update({str(index): [10] for index in range(4)})
    interval = report._clustered_partition_mean_interval(clusters, 20260926)
    rng = np.random.default_rng(20260926)
    indices = rng.integers(0, 5, size=(2000, 5))
    totals = np.array([-200, 10, 10, 10, 10])
    counts = np.array([100, 1, 1, 1, 1])
    expected = np.percentile(
        totals[indices].sum(axis=1) / counts[indices].sum(axis=1), [2.5, 97.5]
    )
    assert interval == pytest.approx(expected)
    # The former unweighted circuit-mean interval was wholly positive,
    # despite a negative displayed partition-weighted mean.
    assert interval[0] < 0
    assert sum(totals) / sum(counts) < 0


def test_cluster_bootstrap_matches_unweighted_for_equal_sizes(report):
    clusters = {"a": [-2, -4], "b": [2, 4], "c": [0, 0]}
    interval = report._clustered_partition_mean_interval(clusters, 17)
    rng = np.random.default_rng(17)
    indices = rng.integers(0, 3, size=(2000, 3))
    expected = np.percentile(np.array([-3, 3, 0])[indices].mean(axis=1), [2.5, 97.5])
    assert interval == pytest.approx(expected)


@pytest.mark.parametrize("clusters", [{}, {"only": [-2, 3]}])
def test_cluster_bootstrap_needs_multiple_source_circuits(report, clusters):
    assert report._clustered_partition_mean_interval(clusters, 17) is None


def test_cluster_bootstrap_is_deterministic_and_replication_invariant(report):
    clusters = {"a": [-2, 1], "b": [3], "c": [0, 4, -1]}
    expected = report._clustered_partition_mean_interval(clusters, 17)
    assert report._clustered_partition_mean_interval(clusters, 17) == expected
    replicated = {key: values * 5 for key, values in clusters.items()}
    assert report._clustered_partition_mean_interval(replicated, 17) == pytest.approx(expected)


def test_summary_uses_matching_cluster_estimand(report):
    cases, rows = [], []
    for source, deltas in [("large", [-2] * 100), *[(str(i), [10]) for i in range(4)]]:
        for index, delta in enumerate(deltas):
            identifier = source + "-" + str(index)
            item = case(identifier)
            item["source_circuit"] = source
            cases.append(item)
            rows.extend([
                verified(identifier, "osr", 1, cnots=64 + delta),
                verified(identifier, "bqskit", 1, cnots=64),
            ])
    summary = report.summarize(rows, cases, 20260926, 1e-10)["3q/a2a"]
    assert summary["mean_cnot_delta_osr_minus_bqskit"] == pytest.approx(-160 / 104)
    assert summary["cnot_delta_ci_estimand"] == "partition_weighted_mean"
    assert summary["circuit_cluster_mean_delta_95pct_ci"][0] < 0


def test_paired_timings_exclude_unmatched_verified_survivors(report):
    summary = sample_summary(report)
    assert summary["median_seconds_on_verified"] == {"osr": 10, "bqskit": 10}
    assert summary["median_seconds_on_paired_verified"] == {"osr": 6, "bqskit": 9}
    assert summary["median_wall_seconds_on_paired_verified"] == {"osr": 7.5, "bqskit": 12.5}
    assert summary["paired_wall_timing_count"] == 2
    assert summary["paired_solver_speedup_count"] == 2
    assert summary["median_paired_solver_speedup_bqskit_over_osr"] == 2.5
    assert summary["median_paired_wall_speedup_bqskit_over_osr"] == pytest.approx((4 + 13 / 12) / 2)
    assert summary["completion_aware_osr_wins"] == 1
    assert summary["completion_aware_osr_losses"] == 1


def test_wall_timings_require_both_values_for_each_pair(report):
    summary = sample_summary(report, missing_wall=True)
    assert summary["paired_wall_timing_count"] == 1
    assert summary["median_wall_seconds_on_paired_verified"] == {"osr": 12, "bqskit": 13}
    assert summary["median_paired_wall_speedup_bqskit_over_osr"] == pytest.approx(13 / 12)


def test_zero_elapsed_time_does_not_produce_infinite_speedup(report):
    rows = [verified("a", method, 0, 0) for method in ("osr", "bqskit")]
    summary = report.summarize(rows, [case("a")], 17, 1e-10)["3q/a2a"]
    assert summary["median_seconds_on_paired_verified"] == {"osr": 0, "bqskit": 0}
    assert summary["paired_solver_speedup_count"] == 0
    assert summary["paired_wall_speedup_count"] == 0
    assert summary["median_paired_solver_speedup_bqskit_over_osr"] is None
    assert summary["median_paired_wall_speedup_bqskit_over_osr"] is None


def test_missing_opponent_has_no_paired_timing(report):
    rows = [verified("a", "osr", 2, 3)]
    summary = report.summarize(rows, [case("a")], 17, 1e-10)["3q/a2a"]
    assert summary["median_seconds_on_paired_verified"] == {"osr": None, "bqskit": None}
    assert summary["paired_wall_timing_count"] == 0


def test_latex_uses_paired_times_and_describes_estimand(report, tmp_path):
    summary = sample_summary(report)
    path = tmp_path / "comparison.tex"
    report.write_tex_table({"3q/a2a": summary}, path, 1e-10)
    tex = path.read_text()
    assert "$6.00/9.00$" in tex
    assert "Paired time OSR/BQSKit" in tex
    assert "partition-weighted mean" in tex
    assert "solver times use only jointly verified targets" in tex
