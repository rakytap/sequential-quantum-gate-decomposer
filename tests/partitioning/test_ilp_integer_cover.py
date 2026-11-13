"""A timed-out LP relaxation must never become a circuit partition."""
import pytest
import pulp

from squander.partitioning import ilp


def test_fractional_timeout_retries_with_complete_integer_incumbent(monkeypatch):
    parts = [frozenset((0,)), frozenset((1,)), frozenset((2,)), frozenset((0, 1))]
    graph = {0: {1}, 1: {2}, 2: set()}
    calls = []
    real_cbc = ilp.cbc_solver

    class FractionalTimeout(pulp.LpSolver):
        def actualSolve(self, problem, **kwargs):
            solve_status = getattr(pulp, "LpSolveStatus", None)
            if solve_status is not None:
                from pulp.apis.core import clocks

                start = clocks()
            for variable in problem.variables():
                variable.varValue = 0.5
            if solve_status is not None:
                # PuLP 4 returns solve statistics instead of problem.status.
                return self.buildStats(
                    problem, solve_status.TimeLimit, False, start=start
                )
            problem.status = pulp.LpStatusNotSolved
            return problem.status

    def solver(module, **kwargs):
        calls.append(kwargs)
        return FractionalTimeout() if len(calls) == 1 else real_cbc(module, **kwargs)

    monkeypatch.setattr(ilp, "_GUROBI_AVAILABLE", False)
    monkeypatch.setattr(ilp, "_GUROBI_AVAILABILITY_ERROR", RuntimeError("Test CBC fallback"))
    monkeypatch.setattr(ilp, "cbc_solver", solver)
    selected, _ = ilp.ilp_global_optimal(parts, graph, weights=[3, 3, 3, 4])
    assert set(selected) == {2, 3}
    assert calls[1]["warmStart"] is True
    assert calls[1]["threads"] == 1


@pytest.mark.parametrize("values", [(1, 0, 0), (1, 1, 1), (0.5, 0.5, 0.5), (None, 0, 1)])
def test_incomplete_overlapping_and_fractional_covers_are_rejected(values):
    parts = [frozenset((0,)), frozenset((1,)), frozenset((0, 1))]
    assert ilp._binary_partition_cover(values, parts, {0, 1}) is None
