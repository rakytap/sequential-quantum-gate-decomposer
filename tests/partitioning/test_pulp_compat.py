"""Regression tests for the PuLP 3/4 compatibility boundary."""

import pulp

from squander.partitioning.pulp_compat import PulpVariables


def test_nested_binary_variable_dicts_are_bounded():
    problem = pulp.LpProblem("nested_binary_bounds", pulp.LpMinimize)
    variables = PulpVariables(problem, pulp).dicts(
        "binary", (range(2), range(2), range(2)), cat="Binary"
    )

    for first in range(2):
        for second in range(2):
            for third in range(2):
                variable = variables[first][second][third]
                assert variable.lowBound == 0
                assert variable.upBound == 1
