"""Incomplete head-to-head runs must not manufacture comparative wins."""

import importlib.util
from pathlib import Path
import sys

import pytest


QASM_HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'


@pytest.fixture(scope="module")
def benchmark():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/partitioning/local_synthesis_head_to_head.py"
    )
    spec = importlib.util.spec_from_file_location("osr_summary_regression", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def case(case_id="target"):
    return {
        "case_id": case_id,
        "width": 3,
        "context": "a2a",
        "dataset": "unit_test",
        "source_circuit": case_id,
        "target_sha256": "0" * 64,
        "graph": [[0, 1], [0, 2], [1, 2]],
    }


def result(method, status="verified", case_id="target", cnots=1):
    row = {"case_id": case_id, "method": method, "status": status}
    if status == "verified":
        row.update({
            "cnots": cnots,
            "seconds": 0.1,
            "topology_verified": True,
            "output_qasm": QASM_HEADER + "cx q[0],q[1];\n" * cnots,
        })
    return row


@pytest.mark.parametrize("survivor", ["osr", "bqskit"])
@pytest.mark.parametrize("explicit_not_run", [False, True])
def test_unrun_competitor_is_neither_failure_nor_win(
    benchmark, survivor, explicit_not_run
):
    other = "bqskit" if survivor == "osr" else "osr"
    rows = [result(survivor)]
    if explicit_not_run:
        rows.append(result(other, "not_run"))
    summary = benchmark.summarize(rows, [case()], 42, 1e-10)["3q/a2a"]

    assert summary["paired_attempted"] == 0
    assert summary["paired_verified"] == 0
    assert summary["osr_only_verified"] == 0
    assert summary["bqskit_only_verified"] == 0
    assert summary["completion_aware_osr_wins"] == 0
    assert summary["completion_aware_osr_losses"] == 0
    assert summary["unpaired_verified"][survivor] == 1
    assert summary["unrun_counts"][other] == 1

    audit = benchmark.regression_audit(rows, [case()])
    assert audit[0]["kind"] == "missing_comparison_result"
    assert audit[0]["missing_methods"] == [other]


@pytest.mark.parametrize("survivor", ["osr", "bqskit"])
@pytest.mark.parametrize("failure", ["timeout", "no_solution", "error"])
def test_actual_competitor_failure_remains_a_completion_outcome(
    benchmark, survivor, failure
):
    other = "bqskit" if survivor == "osr" else "osr"
    rows = [result(survivor), result(other, failure)]
    summary = benchmark.summarize(rows, [case()], 42, 1e-10)["3q/a2a"]

    assert summary["paired_attempted"] == 1
    assert summary["unpaired_verified"] == {"osr": 0, "bqskit": 0}
    assert summary["unrun_counts"] == {"osr": 0, "bqskit": 0}
    assert summary["completion_aware_osr_wins"] == int(survivor == "osr")
    assert summary["completion_aware_osr_losses"] == int(survivor == "bqskit")
    assert summary["osr_timeouts_against_verified_bqskit"] == int(
        survivor == "bqskit" and failure == "timeout"
    )
    assert benchmark.regression_audit(rows, [case()])[0]["missing_methods"] == []


def test_fully_paired_counts_and_equivalence_checks_are_preserved(benchmark):
    cases = [case(name) for name in ("osr_timeout", "bq_timeout", "loss", "tie")]
    rows = [
        result("osr", "timeout", "osr_timeout"),
        result("bqskit", case_id="osr_timeout"),
        result("osr", case_id="bq_timeout"),
        result("bqskit", "timeout", "bq_timeout"),
        result("osr", case_id="loss", cnots=3),
        result("bqskit", case_id="loss", cnots=1),
        result("osr", case_id="tie", cnots=1),
        result("bqskit", case_id="tie", cnots=1),
    ]
    summary = benchmark.summarize(rows, cases, 42, 1e-10)["3q/a2a"]

    assert summary["paired_attempted"] == 4
    assert summary["paired_verified"] == 2
    assert summary["osr_wins"] == 0
    assert summary["ties"] == 1
    assert summary["bqskit_wins"] == 1
    assert summary["completion_aware_osr_wins"] == 1
    assert summary["completion_aware_osr_losses"] == 2
    assert summary["max_pairwise_process_infidelity"] <= 1e-10
    assert {entry["kind"] for entry in benchmark.regression_audit(rows, cases)} == {
        "osr_timeout_loss", "bqskit_failure_osr_verified", "osr_cnot_loss"
    }


def test_neither_method_run_is_an_explicit_gap(benchmark):
    summary = benchmark.summarize([], [case()], 42, 1e-10)["3q/a2a"]
    assert summary["unrun_counts"] == {"osr": 1, "bqskit": 1}
    assert summary["completion_aware_osr_wins"] == 0
    assert summary["completion_aware_osr_losses"] == 0
    audit = benchmark.regression_audit([], [case()])
    assert audit[0]["kind"] == "missing_comparison_result"
    assert audit[0]["missing_methods"] == ["osr", "bqskit"]


def test_tex_does_not_count_missing_osr_as_a_synthesis_failure(benchmark, tmp_path):
    rows = [result("bqskit")]
    summary = benchmark.summarize(rows, [case()], 42, 1e-10)
    path = tmp_path / "comparison.tex"
    benchmark.write_tex_table(
        summary, path, 1e-10, benchmark.regression_audit(rows, [case()])
    )
    text = path.read_text(encoding="utf-8")
    assert "3q/a2a & 1 & 0 & 0 & 1 & 0 & 0/0/0" in text
    assert "not synthesis failures or wins" in text
    assert "missing_comparison_result" in text
