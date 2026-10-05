---
name: test-density-matrix
description: Runs the SQUANDER density matrix test workflow end-to-end. Use when the user asks to test, verify, validate, or benchmark density matrix functionality, including pytest, examples, optional C++ tests, and Qiskit comparison.
---

# Density Matrix Testing

## When to Use

Use this skill when working on:
- `squander/density_matrix/*`
- `squander/partitioning/*`
- `squander/src-cpp/density_matrix/*`
- `squander/src-cpp/variational_quantum_eigensolver/*`
- `tests/density_matrix/*`
- `tests/partitioning/*`
- `tests/VQE/*`
- `examples/density_matrix/*`
- `examples/VQE/*`
- `benchmarks/*` for density matrix validation

Also use before PRs that touch density matrix code or docs.

## Canonical References

- `docs/density_matrix_project/SETUP.md` is the source of truth for setup/testing commands.
- `.cursor/rules/RULE.md` defines project requirements (including C++11 and Python 3.13).
- `pytest.ini` defines test root and marker conventions.

If instructions conflict, follow `docs/density_matrix_project/SETUP.md`.

## Required Environment

Always use the **`qgd` conda environment** for density-matrix and partitioning tests (never rely on the system interpreter).

Interactive shell:

```bash
conda activate qgd
```

Non-interactive runs (agents, CI-style one-liners) should invoke pytest through conda, for example:

```bash
conda run -n qgd --no-capture-output pytest tests/partitioning/ -q
```

## Test Workflow (Default)

Copy and track this checklist:

```text
Density matrix test checklist:
- [ ] 1) Smoke import check
- [ ] 2) Python density matrix tests
- [ ] 3) Example script execution
- [ ] 4) Optional Qiskit validation benchmark
- [ ] 5) Optional C++ density matrix tests
```

### 1) Smoke Import Check

```bash
python -c "from squander.density_matrix import DensityMatrix, NoisyCircuit; print('density-matrix import OK')"
```

### 2) Python Test Suite

Run the density-matrix regression lane (auto-tagged in `tests/conftest.py`):

```bash
pytest -m density_matrix -v
```

Notes:
- `pytest.ini` sets `testpaths = ./tests` and ignores `tests/partitioning/evidence` by default
- markers: `density_matrix` (project regression suite), `slow`
- membership: `tests/density_matrix/`, noisy `tests/partitioning/` (not `test_partition.py`), `tests/VQE/test_VQE.py`

Run only slow density-matrix tests:

```bash
pytest -m "density_matrix and slow" -v
```

Run only non-slow density-matrix tests:

```bash
pytest -m "density_matrix and not slow" -v
```

Optional evidence validators under `tests/partitioning/evidence` (not in default collection):

```bash
pytest tests/partitioning/evidence -o addopts= -m density_matrix -v
```

Examples and benchmark scripts (not pytest modules):

```bash
python examples/density_matrix/basic_usage.py
pytest benchmarks/density_matrix/ -v
```

Run a single test:

```bash
pytest tests/density_matrix/test_density_matrix.py::TestNoisyCircuitNoise::test_depolarizing -v
```

### 3) Example Script

```bash
python examples/density_matrix/basic_usage.py
```

### 4) Optional Qiskit Validation

Requires optional Qiskit dependencies from `SETUP.md`.

```bash
python benchmarks/validate_squander_vs_qiskit.py
```

### 5) Optional C++ Density Matrix Tests

```bash
export QGD_CTEST=1
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"

# QGD_CTEST is consumed at CMake configure time, so force reconfigure.
rm -rf _skbuild
python setup.py build_ext

TEST_BIN=""
for candidate in _skbuild/*/cmake-build/squander/src-cpp/density_matrix/test_density_matrix_cpp; do
  if [ -x "$candidate" ]; then
    TEST_BIN="$candidate"
    break
  fi
done

if [ -z "$TEST_BIN" ]; then
  echo "test_density_matrix_cpp binary not found"
  exit 1
fi

"$TEST_BIN"

unset LDFLAGS LIBRARY_PATH QGD_CTEST
```

## Common Failures and Fixes

See `references/common-failures.md` (SETUP.md remains authoritative).

## Output Format for Reporting Results

When reporting test execution results, use:

```text
Density matrix test report:
- Environment: qgd (active/inactive)
- Smoke import: pass/fail
- Pytest: pass/fail (N passed, M failed, K skipped if available)
- Example script: pass/fail
- Qiskit validation: pass/fail/not run
- C++ tests: pass/fail/not run
- Blocking issues: <list or none>
```

## Bounded state-vector (SV) regression lane

Quantum simulation tests are slow, so run only what the change needs. Command, deselects,
and claim wording: `references/sv-regression-lane.md`.

## KNOWN-FLAKY list

A test is known-flaky only if it is outside the density-matrix code AND its failure is a precision or threshold miss. Known-flaky tests are deselected and recorded, not fixed. Any failure in DM tests or tests/partitioning stops the run and goes to Tech Lead. Every report that deselects a test lists it as a row: test id | failure | date | run | why it isn't attributable to the slice. For the last column, attach `git diff --stat <base> <head>` and show that no squander/ or C++ paths, and nothing the test imports, changed.

| Test id | Failure | Date | Runs |
|---|---|---|---|
| tests/decomposition/test_QX2.py::Test_Decomposition::test_N_Qubit_Decomposition_QX2 | decomposition residual vs 1e-7 threshold: 1.1356e-07 (dirty run), 1.1462e-07 (C1 isolation, FAIL), 9.877e-08 (C1 isolation, PASS) | 2026-10-05 | M-F1a q4 run 1 and C1 |

The test is seeded (random_state=0, optimizer random_seed=1) but still nondeterministic, about 216-236 s per standalone run.

## Snapshot and restore around validation_pipeline.py

After every `validation_pipeline.py` run, restore all eight historical bundles from HEAD,
and never commit them. This standing rule holds until Research Manager revises it; it is
not tied to any open drift work or a particular slice.

Research Manager gate: no C1 and no C2 may include any of the eight, and any commit that
touches them is blocked.

Procedure and which files the pipeline rewrites: `references/validation-pipeline-restore.md`.
Banned in-place evidence CLIs on rocky-squander: same reference § host policy.

## Regeneration acceptance

Milestone-specific regeneration exceptions and revision-only mismatch rules live in that
milestone's ADRs (not in this skill body). Shape and restore discipline:
`references/regeneration-acceptance.md`.

## Long runs: tmux

Run the pipeline and any pytest lane over a few minutes in a detached tmux session named for the tester and run, for example `tmux new-session -d -s tester-<slice>-<step> -c <checkout>`. Send output to a log under /tmp/<run>/ and record the exit code to a file. Never attach to, kill, or reuse sessions you didn't create. Give a time estimate when starting.

## QA-001 exactness predicate (lambda_min witness)

When a counted run cites QA-001, follow `references/qa001-exactness-predicate.md`.
