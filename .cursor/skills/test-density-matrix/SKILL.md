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

### pybind11 not found

```bash
conda activate qgd
pip install pybind11
python setup.py build_ext
```

### TBB headers or libs not found

```bash
conda install -y tbb-devel -c conda-forge
export TBB_INC_DIR=~/.conda/envs/qgd/include
export TBB_LIB_DIR=~/.conda/envs/qgd/lib
rm -rf _skbuild
python setup.py build_ext
```

### `ModuleNotFoundError: No module named 'squander.density_matrix'`

```bash
python setup.py build_ext
python -m pip install -e .
ls squander/density_matrix/_density_matrix_cpp*.so
```

### Qiskit installation issues on Python 3.13

```bash
conda install -y qiskit qiskit-aer -c conda-forge
```

### `./test_standalone/test_density_matrix_cpp: No such file or directory`

The C++ test binary is built under `_skbuild/*/cmake-build/...`, not
`./test_standalone/` in this workflow.

Fix: use the optional C++ test workflow above exactly (including `rm -rf _skbuild`
before build, then execute discovered `_skbuild/*/.../test_density_matrix_cpp` path).

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

Quantum simulation tests are slow, so run only what the change needs. Run this lane when a slice could affect SV behaviour, or when a counted run requires SV non-regression. Skip VQE unless the change touches VQE or variational code paths. Use read-only flags so the run leaves no cache or bytecode in a clean tree:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python -m pytest tests/gates tests/decomposition \
  --ignore=tests/decomposition/test_wide_circuit_optimization.py \
  --deselect tests/decomposition/test_QX2.py::Test_Decomposition::test_N_Qubit_Decomposition_QX2 \
  -p no:cacheprovider -q
```

Reference on rocky-squander (2x EPYC 7542): 805 passed and 1 deselected in about 31 min. With VQE (tests/VQE) added, it was 864 tests in about 38 min. Always run it in the background (see Long runs: tmux). Claim wording: "no state-vector regression attributable to this slice; one known flaky SV test listed and deselected". Never write "all SV tests green".

## KNOWN-FLAKY list

A test is known-flaky only if it is outside the density-matrix code AND its failure is a precision or threshold miss. Known-flaky tests are deselected and recorded, not fixed. Any failure in DM tests or tests/partitioning stops the run and goes to Tech Lead. Every report that deselects a test lists it as a row: test id | failure | date | run | why it isn't attributable to the slice. For the last column, attach `git diff --stat <base> <head>` and show that no squander/ or C++ paths, and nothing the test imports, changed.

| Test id | Failure | Date | Runs |
|---|---|---|---|
| tests/decomposition/test_QX2.py::Test_Decomposition::test_N_Qubit_Decomposition_QX2 | decomposition residual vs 1e-7 threshold: 1.1356e-07 (dirty run), 1.1462e-07 (C1 isolation, FAIL), 9.877e-08 (C1 isolation, PASS) | 2026-10-05 | M-F1a q4 run 1 and C1 |

The test is seeded (random_state=0, optimizer random_seed=1) but still nondeterministic, about 216-236 s per standalone run.

## Snapshot and restore around validation_pipeline.py

`benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` rewrites the q4 bundle AND six tracked sibling bundles: correctness_package, external_correctness, output_integrity, runtime_classification, sequential_correctness, and unsupported_boundary. The rewrites change real content (partition-member records, runtime/RSS, residual values, unsupported_boundary reason strings), not just timestamps, even when every status stays pass. correctness_matrix and summary_consistency were not rewritten on disk.

After every `validation_pipeline.py` run, restore all eight historical bundles from HEAD,
and never commit them. This standing rule holds until Research Manager revises it; it is
not tied to any open drift work or a particular slice.

Research Manager gate: no C1 and no C2 may include any of the eight, and any commit that
touches them is blocked.

Only six of the eight are Phase-3 evidence under ADR-F1A-005.

1. Before running: check that HEAD is the counted revision and that `git status --porcelain --untracked-files=all` is empty. Copy the committed bundle(s) to /tmp/<run>/.
2. After running: copy every rewritten bundle to /tmp/<run>/ with its sha256 value, and diff it against the committed version.
3. Restore each tracked file with `git show <counted-sha>:<path> > <path>`. Never use stash, reset, checkout, or clean. Then confirm `git diff --quiet` and an empty porcelain.
4. Report content drift in the siblings as a finding for Tech Lead. It is not a pass/fail item unless a status changes.

## Rocky-squander: banned in-place evidence runs

On rocky-squander, nobody runs the standalone per-suite CLIs,
`phase31_validation_pipeline.py`, or the `performance_evidence` pipeline, because they
rewrite evidence in place. This holds until a later Research Manager decision.

## Regeneration acceptance (option i)

Regenerating at a commit after the one that produced the committed bundle is EXPECTED to exit 1, with status=fail and summary.first_failure=regeneration, because provenance.implementation_revision changes. Run it once with no env overrides, then accept only if all of these hold:

- The QA-001 metrics (Frobenius, max-abs, |Tr-1|) match the committed bundle within 1e-10, and lambda_min >= -1e-12.
- The route and label fields are identical.
- qa001_pass, provenance_pass, and clean_start are true, dirty_paths is empty, and implementation_revision equals HEAD.
- The other eight suites keep status=pass.
- An unfiltered recursive field diff (regenerated vs committed) shows ONLY these differences: cases[0].provenance.implementation_revision, status pass->fail, summary.first_failure=regeneration, regeneration.prior_present false->true, regeneration.pass true->false, and regeneration.first_mismatch=cases[0].provenance.implementation_revision. Any other difference is a FAIL.

After that, restore every rewritten file to the HEAD bytes, as in the snapshot section. (At C2 a2928bf1 there were exactly 6 differences and 0 unexpected.)

## Long runs: tmux

Run the pipeline and any pytest lane over a few minutes in a detached tmux session named for the tester and run, for example `tmux new-session -d -s tester-<slice>-<step> -c <checkout>`. Send output to a log under /tmp/<run>/ and record the exit code to a file. Never attach to, kill, or reuse sessions you didn't create, such as the build session mf1a-q4-build. Give a time estimate when starting.

## QA-001 exactness predicate (lambda_min witness)

When a counted density-matrix predicate cites QA-001 (ADR-F1A-002, docs/specs/milestones/exactness-reconfirmation/ADRS_EXACTNESS_RECONFIRMATION.md), compute lambda_min(rho) only with the existing DensityMatrix.eigenvalues() contract. LAPACK zheev reads the upper triangle of rho as stored, and the minimum returned eigenvalue is lambda_min. Check finiteness of every entry and residual first. A solver failure or a non-finite eigenvalue fails the row. Do not symmetrize: (rho + rho^dagger)/2 is forbidden. Do not add a Hermiticity gate or cite QA-010. This is an eigensolver convention, not a Hermiticity check. The thresholds are frozen: ||delta rho||_F <= 1e-10, max-abs <= 1e-10, |Tr rho - 1| <= 1e-10, lambda_min >= -1e-12, all against the sequential oracle. Record each value and its pass/fail. Older rho_is_valid, energy, and Aer fields may appear as context but never decide the counted status.

