# Tech stack — density-matrix track (current state)

> **Status:** current-state reference · **Owner skill:** `spec-driven-development` ·
> **Scope:** languages, build, environments, test and evidence lanes, and the tool
> conventions an agent must know before changing density-matrix code.
> Install steps and troubleshooting live in `docs/density_matrix_project/SETUP.md`; this file
> names the commands and lanes that evidence rows and closeouts cite.

## Languages, runtimes, build

| Item | Value |
|------|-------|
| C++ | C++11 standard; GCC ≥ 4.8.1 / Intel ≥ 14.0.1 / MSVC; TBB for parallelism, OpenBLAS/LAPACKE |
| Python | 3.13 in the `qgd` conda env (`requires-python >= 3.8` in `pyproject.toml`; 3.10+ needed for the SDD linters) |
| Build | scikit-build + CMake ≥ 3.15 (`CMakeLists.txt` at repo root; density module in `squander/src-cpp/density_matrix/CMakeLists.txt`); pybind11 for bindings |
| Package | `squander` (editable install), version in `pyproject.toml` / `setup.py` |
| Core deps | numpy, scipy, networkx, pybind11; optional qiskit + qiskit-aer, matplotlib |
| Environment | conda env `qgd` (`conda_env_example.yaml`); `TBB_INC_DIR` / `TBB_LIB_DIR` exported before building |

## Commands

All product commands run in the `qgd` environment: interactively after `conda activate qgd`,
non-interactively as `conda run -n qgd --no-capture-output <cmd>`.

| Purpose | Command |
|---------|---------|
| Build the extension | `export TBB_INC_DIR=~/.conda/envs/qgd/include TBB_LIB_DIR=~/.conda/envs/qgd/lib && python setup.py build_ext && python -m pip install -e .` |
| Clean rebuild (after C++/CMake/header changes) | `clean-rebuild` skill (`.cursor/skills/clean-rebuild/SKILL.md`) |
| Smoke import | `python -c "from squander.density_matrix import DensityMatrix, NoisyCircuit"` |
| Example | `python examples/density_matrix/basic_usage.py` |
| Spec linters | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh [--strict] [<milestone dir>]` |

## Test and evidence lanes

Every evidence-matrix row names one of these lanes. `pytest.ini` sets `testpaths = ./tests`,
ignores `tests/partitioning/evidence`, and defines the `slow` marker.

| Lane | Command | Notes |
|------|---------|-------|
| **Fast pytest** | `pytest -m "density_matrix and not slow"` | Default gate for a slice; minutes |
| **Slow pytest** | `pytest -m "density_matrix and slow"` | Long-running cases; run before a closeout |
| **Benchmark tests** | `pytest benchmarks/density_matrix -v` | Validators for the evidence bundles |
| **Correctness evidence pipeline** | `python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | Regenerates counted correctness bundles vs the sequential reference (and Aer where required) |
| **Performance evidence pipeline** | `python benchmarks/density_matrix/performance_evidence/validation_pipeline.py` | Regenerates performance/diagnosis bundles; run correctness first |
| **Phase 3.1 pipeline** | `python benchmarks/density_matrix/correctness_evidence/phase31_validation_pipeline.py` | Frozen channel-native decision-study slice |
| **Qiskit Aer external reference** | `python benchmarks/density_matrix/validate_squander_vs_qiskit.py` | Optional; needs `qiskit`, `qiskit-aer` |
| **C++ unit tests** | `QGD_CTEST=1` at configure time, then run the discovered `_skbuild/*/cmake-build/squander/src-cpp/density_matrix/test_density_matrix_cpp` | Optional; procedure in the `test-density-matrix` skill |
| **CI** | `.github/workflows/ci.yml` — Linux (pip) and Windows (conda) build + `pytest tests/` | Upstream SQUANDER gate; does not run the evidence pipelines |

Evidence artifacts are written under `benchmarks/density_matrix/artifacts/<tree>/`; a
closeout cites the pipeline command and the artifact path it produced.

## Conventions an agent must know

- Never use the system interpreter for tests, benchmarks, or examples; the SDD linters are
  the only stdlib-only scripts and may run with any Python ≥ 3.10.
- Rebuild before testing after any C++ or CMake change; a stale `.so` silently tests old code.
- The sequential `NoisyCircuit` executor is the exact baseline; Qiskit Aer is the external
  reference. Do not loosen a frozen tolerance to make a lane pass.
- The density path is strict: widen a support surface only with a matching negative test
  that pins the rejected input and the error it raises.
- `docs/specs/` is the spec root; `docs/density_matrix_project/archive/` is frozen history.
