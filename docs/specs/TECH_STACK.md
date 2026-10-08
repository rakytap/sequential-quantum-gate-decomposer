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
| **Correctness evidence pipeline** | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | Regenerates the M-F1a correctness bundles against `execute_sequential_density_reference`. Aer and energy are not in the counted set. The optional Aer lane is the row below. |
| **M-F1a exactness fitness lane** | `pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts=""` | Together with the correctness evidence pipeline row; not the fast `pytest -m "density_matrix and not slow"` lane |
| **Performance evidence pipeline** | `python benchmarks/density_matrix/performance_evidence/validation_pipeline.py` | Regenerates performance/diagnosis bundles; run correctness first |
| **Phase 3.1 pipeline** | `python benchmarks/density_matrix/correctness_evidence/phase31_validation_pipeline.py` | Frozen channel-native decision-study slice |
| **Qiskit Aer external reference** | `python benchmarks/density_matrix/validate_squander_vs_qiskit.py` | Optional; needs `qiskit`, `qiskit-aer` |
| **C++ unit tests** | `QGD_CTEST=1` at configure time, then run the discovered `_skbuild/*/cmake-build/squander/src-cpp/density_matrix/test_density_matrix_cpp` | Optional; procedure in the `test-density-matrix` skill |
| **CI** | `.github/workflows/ci.yml` — Linux (pip) and Windows (conda) build + `pytest tests/` | Upstream SQUANDER gate; does not run the evidence pipelines |
| **M-F1a state-vector gate (G-04)** | rocky-local project CI (G-04; Tester recipe; GitHub Actions out of semester path per Zoltán 2026-10-06) at HEAD `031996f4` | The M-F1a state-vector gate is rocky-local project CI (G-04; Tester recipe; GitHub Actions out of semester path per Zoltán 2026-10-06) at HEAD `031996f4`. It is not `.github/workflows/ci.yml` `workflow_dispatch`. G-05 does not run that job. O-10 push at `031996f4` is sync-only. Recorded outcome at write time: pass, `/tmp/mf1a-g04-rocky-ci/REPORT.md` (1067 passed, 1 deselected QX2, exit 0, wall 40m56s; pytest.log `/tmp/mf1a-g04-rocky-ci/pytest.log`). A missing or failed job is not described as green. |
| **Interop profile lane (M-F5a)** | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py` | Sibling lane for E-VQE at 4, 6, and 8 and the attribution routes. Closeout status Shipped at `918a73a4`. R-strict is the ADR-F5A-011 refusal row and is not timed. |
| **M-F5a rocky-local CI (G-09)** | full `pytest tests/` at `2a1dc14e` (the `ci.yml` linux line, plus the standing QX2 deselect, no taskset) | PASS: 1287 passed, 1 skipped, 1 deselected, exit 0, 588 s. Report `/tmp/mf5a-close-ci/REPORT.md` `077c56d6d2896e178304c7740fe3498b447b29b00efce1d067dd83448974574b`. |

Evidence artifacts are written under `benchmarks/density_matrix/artifacts/<tree>/`. M-F1a
artifacts live under `benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/`. The
counted file is `mf1a/counted/mf1a_counted_bundle.json`. The other five directories are the
provisional siblings (`milestone_counted` false): `q4_baseline`, `fused`, `hybrid`, `strict`,
and `baseline`. The eight pre-M-F1a bundles are verified and not written. Docs record
`completeness_claim` true for handoff; the frozen counted bundle JSON may still show
`completeness_claim` false as recorded at generation time (`1b123a9a`; same sentence as
`ARCHITECTURE_OVERVIEW.md`). A closeout cites the pipeline command and the artifact path it
produced.

M-F5a artifacts live under `benchmarks/density_matrix/artifacts/interop_profile/`. Closeout status
is Shipped, not Delivered (`918a73a4`). Measured inventory: E-VQE at 4, 6, and 8, plus timed attribution routes
R-base, R-fused, and R-hybrid. R-strict is a refusal row with diagnosis
`channel_native_noise_presence` (ADR-F5A-011) and carries no timings. UB (one-sided 95% upper bound on O, E-VQE bundles) ≤1.06% at 4/6/8: 1.0589% / 0.2149% / 0.0744%. QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. Exactness (RM N-c): ≤1.2e-16 vs the sequential reference (w4 data only) and ≤5e-16 vs Qiskit Aer 0.17.2. Apply time versus R-base at widths 6 and 8: fused 3.27× and 3.43×, hybrid 6.57× and 8.82×. Current implementation cost, not intrinsic cost; no speed claim. CAP-004 is hold-the-line. There is no reduction. C2 (strict-capable side workload): backlog, possible post-supervisor item, not started; not in M-F5a (Zoltán via PhD Manager and RM, 2026-10-08).
REQ-009 (checklist G-08) is closed per CLOSEOUT and RM ACCEPT 2026-10-08.

## Conventions an agent must know

- Never use the system interpreter for tests, benchmarks, or examples; the SDD linters are
  the only stdlib-only scripts and may run with any Python ≥ 3.10.
- Rebuild before testing after any C++ or CMake change; a stale `.so` silently tests old code.
- The sequential `NoisyCircuit` executor is the exact baseline; Qiskit Aer is the external
  reference. Do not loosen a frozen tolerance to make a lane pass.
- The density path is strict: widen a support surface only with a matching negative test
  that pins the rejected input and the error it raises.
- `docs/specs/` is the spec root; `docs/density_matrix_project/archive/` is frozen history.
