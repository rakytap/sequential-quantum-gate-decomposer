# Bounded state-vector (SV) regression lane

Read when a slice could affect state-vector behaviour or when a counted run requires SV
non-regression. Skip VQE unless the change touches VQE or variational code paths.

## Command

Use read-only flags so the run leaves no cache or bytecode in a clean tree:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python -m pytest tests/gates tests/decomposition \
  --ignore=tests/decomposition/test_wide_circuit_optimization.py \
  --deselect tests/decomposition/test_QX2.py::Test_Decomposition::test_N_Qubit_Decomposition_QX2 \
  -p no:cacheprovider -q
```

Adding `tests/VQE` extends coverage when variational paths are in scope.

## Reporting

Run long lanes in the background (see `SKILL.md` § Long runs: tmux). Claim wording:
"no state-vector regression attributable to this slice; known-flaky SV tests listed and
deselected". Never write "all SV tests green".

Timing varies by host; record wall time in the closeout when a counted run depends on it.
