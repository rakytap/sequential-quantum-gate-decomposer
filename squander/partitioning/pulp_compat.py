"""The small PuLP 3/4 API boundary used by partitioning and routing."""


class PulpVariables:
    """Create variables owned by their problem when the PuLP API supports it."""

    def __init__(self, problem, pulp):
        self.problem = problem
        self.pulp = pulp

    def __call__(self, *args, **kwargs):
        add_variable = getattr(self.problem, "add_variable", None)
        if add_variable is not None:
            return add_variable(*args, **kwargs)
        return self.pulp.LpVariable(*args, **kwargs)

    def dicts(self, *args, **kwargs):
        add_variable_dicts = getattr(self.problem, "add_variable_dicts", None)
        if add_variable_dicts is not None:
            return add_variable_dicts(*args, **kwargs)
        return self.pulp.LpVariable.dicts(*args, **kwargs)


class PulpSolveBackend(str):
    """Keep the backend name while carrying PuLP 4's per-solve statistics."""

    def __new__(cls, name, solve_result):
        backend = super().__new__(cls, name)
        backend.solve_result = solve_result
        return backend


def cbc_solver(pulp, **kwargs):
    """PuLP 4 uses COIN_CMD and installs CBC through its cbc extra."""
    solver_type = getattr(pulp, "PULP_CBC_CMD", None)
    if solver_type is None:
        solver_type = pulp.COIN_CMD
    return solver_type(**kwargs)


def pulp_solve_status(pulp, problem, backend):
    """Read status from the solve result, not PuLP 4's removed problem.status."""
    solve_result = getattr(backend, "solve_result", None)
    if hasattr(solve_result, "status_str"):
        return solve_result.status_str
    if solve_result is not None:
        return pulp.LpStatus[int(solve_result)]
    if hasattr(problem, "status"):
        return pulp.LpStatus[problem.status]
    raise RuntimeError("The PuLP solve result is unavailable.")
