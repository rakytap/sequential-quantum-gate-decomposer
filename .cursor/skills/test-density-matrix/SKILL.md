---
name: test-density-matrix
description: Runs the SQUANDER density matrix test workflow end-to-end. Use when the user asks to test, verify, validate, or benchmark density matrix functionality, including pytest, examples, optional C++ tests, and Qiskit comparison.
---

# Density Matrix Testing

## When to Use

Use this skill when working on:
- `squander/density_matrix/*`
- `squander/src-cpp/density_matrix/*`
- `tests/density_matrix/*`
- `examples/density_matrix/*`
- `benchmarks/*` for density matrix validation

Also use before PRs that touch density matrix code or docs.

## Canonical References

- `docs/density_matrix_project/SETUP.md` is the source of truth for setup/testing commands.
- `.cursor/rules/RULE.md` defines project requirements (including C++11 and Python 3.13).
- `pytest.ini` defines test root and marker conventions.

If instructions conflict, follow `docs/density_matrix_project/SETUP.md`.

## Required Environment

Always activate the conda environment first:

```bash
conda activate qgd
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

Run full density-matrix tests:

```bash
pytest tests/density_matrix/ -v
```

Notes:
- `pytest.ini` sets `testpaths = ./tests`
- marker available: `slow`
- current test file: `tests/density_matrix/test_density_matrix.py`

Run only slow tests:

```bash
pytest tests/density_matrix/ -v -m slow
```

Run only non-slow tests:

```bash
pytest tests/density_matrix/ -v -m "not slow"
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

